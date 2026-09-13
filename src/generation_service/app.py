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
from src.generation_service.validation import (
    QueryRequest,
    QueryResponse,
    UploadResponse,
    DeleteDocumentResponse,
    DeleteVectorsRequest,
    DeleteVectorsResponse,
    PurgeIndexResponse,
)
from src.generation_service.agent import get_agent, AcademicRAGAgent

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
    title="Academic & Engineering RAG Autonomous Agent (Amazon S3 Vectors)",
    description="REST API for answering questions about engineering, coding, and academic textbooks using a dynamic LangGraph Agent, Amazon S3 Vectors, and SQLite Checkpoint Memory.",
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
agent_instance: AcademicRAGAgent | None = None
s3_client = None


@app.on_event("startup")
def startup_event():
    global agent_instance, s3_client
    print("Initializing LangGraph Academic & Engineering RAG Agent with Amazon S3 Vectors & SQLite Checkpointer...")
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


def get_vector_manager() -> S3VectorManager:
    global agent_instance
    if agent_instance and agent_instance.vector_manager:
        return agent_instance.vector_manager
    return S3VectorManager(
        vector_bucket_name=S3_VECTOR_BUCKET_NAME,
        index_name=S3_VECTOR_INDEX_NAME,
        region_name=AWS_REGION,
    )


@app.options("/documents/{document_name}")
def options_delete_document(document_name: str):
    return {}


@app.delete("/documents/{document_name}", response_model=DeleteDocumentResponse, status_code=status.HTTP_200_OK)
def delete_document(document_name: str, delete_s3_file: bool = True):
    """
    Deletes all vector embeddings associated with a specific document from Amazon S3 Vectors.
    Optionally deletes the raw PDF from the S3 document bucket and local data directory.
    """
    try:
        vm = get_vector_manager()
        deleted_count = vm.delete_document_vectors(document_name)

        s3_deleted = False
        if delete_s3_file:
            try:
                s3 = boto3.client("s3", region_name=AWS_REGION)
                s3.delete_object(Bucket=S3_BUCKET_NAME, Key=document_name)
                s3_deleted = True
                print(f"🗑️ Deleted s3://{S3_BUCKET_NAME}/{document_name}")
            except Exception as e:
                print(f"ℹ️ Could not delete S3 object s3://{S3_BUCKET_NAME}/{document_name}: {e}")

        # Also remove from local data directory if present
        local_path = os.path.join(PROJECT_ROOT, "data", document_name)
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
                print(f"🗑️ Deleted local file {local_path}")
            except Exception as e:
                print(f"ℹ️ Could not delete local file: {e}")

        return DeleteDocumentResponse(
            status="success",
            document_name=document_name,
            vectors_deleted=deleted_count,
            s3_object_deleted=s3_deleted,
            message=f"Successfully deleted {deleted_count} vector(s) for document '{document_name}' from Amazon S3 Vectors.",
        )
    except Exception as e:
        print(f"Error deleting document '{document_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document '{document_name}': {e}",
        )


@app.options("/vectors")
def options_delete_vectors():
    return {}


@app.delete("/vectors", response_model=DeleteVectorsResponse, status_code=status.HTTP_200_OK)
def delete_vectors(request: DeleteVectorsRequest):
    """
    Deletes specific vector embeddings by key from Amazon S3 Vectors.
    """
    if not request.keys:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'keys' list cannot be empty.",
        )
    try:
        vm = get_vector_manager()
        vm.delete_vectors(request.keys)
        return DeleteVectorsResponse(
            status="success",
            keys_deleted=len(request.keys),
            message=f"Successfully deleted {len(request.keys)} vector(s) from index '{vm.index_name}'.",
        )
    except Exception as e:
        print(f"Error deleting vectors: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete vectors: {e}",
        )


@app.delete("/index", response_model=PurgeIndexResponse, status_code=status.HTTP_200_OK)
def purge_vector_index(confirm: bool = False, recreate: bool = True):
    """
    Purges/deletes the entire Amazon S3 Vector Index.
    Requires query param ?confirm=true.
    If ?recreate=true (default), automatically re-initializes an empty index.
    """
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Safety confirmation required. Provide query parameter ?confirm=true to delete the index.",
        )
    try:
        vm = get_vector_manager()
        vm.delete_index()
        recreated = False
        if recreate:
            vm.ensure_index(dimension=3072)
            recreated = True

        return PurgeIndexResponse(
            status="success",
            index_name=vm.index_name,
            recreated=recreated,
            message=(
                f"Successfully deleted S3 Vector Index '{vm.index_name}'."
                + (" Recreated empty index." if recreated else "")
            ),
        )
    except Exception as e:
        print(f"Error purging S3 Vector Index: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to purge S3 Vector Index: {e}",
        )



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
            "checkpointer": "SQLite (academic_checkpoints.db)",
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
