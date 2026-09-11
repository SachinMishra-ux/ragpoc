import os
import sys
import tempfile
import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from dotenv import load_dotenv

# Ensure root directory is in the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Load environment variables
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, "src", ".env"))

from src.embedding_service.document_processor import render_pdf_page_to_base64
from src.embedding_service.embedder import GeminiEmbedder
from src.embedding_service.s3_vector_manager import S3VectorManager
from src.generation_service.gemini_rag_llm import GeminiRAG
from src.generation_service.validation import QueryRequest, QueryResponse, UploadResponse
from src.generation_service.agent import get_agent, FinancialRAGAgent

AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "financial-rag-documents-001")
S3_VECTOR_BUCKET_NAME = (
    os.getenv("S3_VECTOR_BUCKET_NAME")
    or os.getenv("VECTOR_BUCKET_NAME")
    or "financial-rag-vectors"
)
S3_VECTOR_INDEX_NAME = (
    os.getenv("S3_VECTOR_INDEX_NAME")
    or os.getenv("INDEX_NAME")
    or os.getenv("COLLECTION_NAME")
    or "financial-index"
)

# Initialize FastAPI App
app = FastAPI(
    title="Financial RAG Autonomous Agent (Amazon S3 Vectors)",
    description="REST API for answering questions about financial PDFs using a dynamic LangGraph Agent, Amazon S3 Vectors, and SQLite Checkpoint Memory.",
    version="3.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global clients
agent_instance: FinancialRAGAgent | None = None
s3_client = None


@app.on_event("startup")
def startup_event():
    global agent_instance, s3_client
    print("Initializing LangGraph Financial RAG Agent with Amazon S3 Vectors & SQLite Checkpointer...")
    try:
        agent_instance = get_agent()
        s3_client = boto3.client("s3", region_name=AWS_REGION)
        print("All clients and LangGraph agent successfully initialized.")
    except Exception as e:
        print(f"ERROR: Initialization failed during startup: {e}")


@app.options("/query")
def options_query():
    return {}


@app.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
def query_rag(request: QueryRequest):
    global agent_instance

    if not agent_instance:
        try:
            agent_instance = get_agent()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Service clients / LangGraph agent are not initialized: {e}",
            )

    try:
        # Run autonomous LangGraph agent (supports multimodal query with image/screenshot)
        result = agent_instance.run(
            question=request.question,
            thread_id=request.thread_id,
            document_name=request.document_name,
            limit=request.limit,
            image_base64=request.image_base64,
        )

        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            thread_id=result["thread_id"],
            tool_called=result["tool_called"],
            user_image=result.get("user_image"),
            pages_retrieved=result["pages_retrieved"],
            images=result["images"],
            sources=result["sources"],
        )
    except Exception as e:
        print(f"Error during agent execution: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing LangGraph agent: {e}",
        )


@app.get("/threads", status_code=status.HTTP_200_OK)
def get_threads():
    """Lists all distinct conversation threads stored in the SQLite checkpointer."""
    if not agent_instance:
        return {"threads": []}
    try:
        return {"threads": agent_instance.list_threads()}
    except Exception as e:
        return {"threads": [], "error": str(e)}


@app.get("/threads/{thread_id}/history", status_code=status.HTTP_200_OK)
def get_thread_history(thread_id: str):
    """Retrieves conversation history messages for a specific thread_id."""
    if not agent_instance:
        return {"thread_id": thread_id, "messages": []}
    try:
        return {"thread_id": thread_id, "messages": agent_instance.get_history(thread_id)}
    except Exception as e:
        return {"thread_id": thread_id, "messages": [], "error": str(e)}


@app.options("/upload")
def options_upload():
    return {}


@app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_200_OK)
async def upload_document(file: UploadFile = File(...)):
    """Uploads a PDF document to S3, triggering the S3 worker ingestion pipeline."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF documents are supported for ingestion.",
        )

    try:
        s3 = boto3.client("s3", region_name=AWS_REGION)
        s3_key = os.path.basename(file.filename)

        print(f"Uploading {file.filename} to s3://{S3_BUCKET_NAME}/{s3_key}...")
        s3.upload_fileobj(
            file.file,
            S3_BUCKET_NAME,
            s3_key,
            ExtraArgs={"ContentType": "application/pdf"},
        )
        print(f"Successfully uploaded {s3_key} to S3.")

        return UploadResponse(
            status="success",
            filename=file.filename,
            bucket=S3_BUCKET_NAME,
            s3_key=s3_key,
            message=(
                f"Successfully uploaded {file.filename} to s3://{S3_BUCKET_NAME}/{s3_key}. "
                f"Ingestion into Amazon S3 Vectors has been triggered."
            ),
        )
    except ClientError as e:
        print(f"S3 ClientError during upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AWS S3 error: {e}",
        )
    except Exception as e:
        print(f"Upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload document: {e}",
        )


@app.get("/documents", status_code=status.HTTP_200_OK)
def list_indexed_documents():
    """Returns a list of distinct document names available in S3 Vectors / data directory."""
    docs = set()
    data_dir = os.path.join(PROJECT_ROOT, "data")
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if f.lower().endswith(".pdf"):
                docs.add(f)

    if agent_instance and agent_instance.vector_manager:
        try:
            vectors = agent_instance.vector_manager.list_vectors(max_results=100, return_metadata=True)
            for v in vectors:
                doc = v.get("metadata", {}).get("document_name")
                if doc:
                    docs.add(doc)
        except Exception as e:
            print(f"Could not list vectors for document filter: {e}")

    return {"documents": sorted(list(docs))}


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    global agent_instance

    agent_ok = agent_instance is not None
    vectors_ok = agent_ok and agent_instance.vector_manager is not None
    llm_ok = agent_ok and agent_instance.llm is not None

    status_str = "healthy" if (agent_ok and vectors_ok and llm_ok) else "degraded"

    return {
        "status": status_str,
        "details": {
            "agent_initialized": agent_ok,
            "s3_vectors_connected": vectors_ok,
            "vector_bucket": S3_VECTOR_BUCKET_NAME,
            "vector_index": S3_VECTOR_INDEX_NAME,
            "llm_initialized": llm_ok,
            "checkpointer": "SQLite (financial_checkpoints.db)",
        },
    }


@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/ui/")


# Mount UI static folder
ui_dir = os.path.join(PROJECT_ROOT, "ui")
if os.path.exists(ui_dir):
    app.mount("/ui", StaticFiles(directory=ui_dir, html=True), name="ui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
