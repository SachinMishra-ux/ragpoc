import os
import sys
import json
import argparse
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
from src.generation_service.bedrock_rag_llm import BedrockNovaRAG


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query the Academic & Engineering RAG pipeline using Amazon S3 Vectors."
    )
    parser.add_argument(
        "question",
        type=str,
        help="The question you want to ask about the academic/engineering textbooks.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nova",
        choices=["nova", "gemini"],
        help="Inference LLM model: 'nova' (Amazon Nova 2 Lite, default) or 'gemini' (Google Gemini).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of context pages to retrieve from Amazon S3 Vectors.",
    )
    parser.add_argument(
        "--filter-doc",
        type=str,
        default=None,
        help="Filter results to a specific textbook name (e.g., 'ELECTRONIC DEVICES AND CIRCUITS.pdf').",
    )
    parser.add_argument(
        "--filter-page",
        type=int,
        default=None,
        help="Filter results to a specific page number.",
    )
    parser.add_argument(
        "--filter-json",
        type=str,
        default=None,
        help="Custom JSON metadata filter dictionary (e.g. '{\"document_name\": \"JPM_Annual_2023.pdf\"}').",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    vector_bucket = (
        os.getenv("S3_VECTOR_BUCKET_NAME")
        or os.getenv("VECTOR_BUCKET_NAME")
        or "financial-rag-vectors"
    )
    vector_index = (
        os.getenv("S3_VECTOR_INDEX_NAME")
        or os.getenv("INDEX_NAME")
        or os.getenv("COLLECTION_NAME")
        or "financial-index"
    )

    # Construct metadata filter expression
    filter_expr = {}
    if args.filter_json:
        try:
            filter_expr = json.loads(args.filter_json)
        except json.JSONDecodeError as e:
            print(f"Error parsing --filter-json: {e}")
            sys.exit(1)
    if args.filter_doc:
        filter_expr["document_name"] = args.filter_doc
    if args.filter_page is not None:
        filter_expr["page_number"] = args.filter_page
    if not filter_expr:
        filter_expr = None

    print("=" * 60)
    print("Executing Academic & Engineering RAG Query against Amazon S3 Vectors...")
    print(f"Question:      '{args.question}'")
    print(f"Vector Bucket: '{vector_bucket}'")
    print(f"Vector Index:  '{vector_index}'")
    if filter_expr:
        print(f"Metadata Filter: {filter_expr}")
    print("=" * 60)

    # 1. Initialize Clients
    try:
        vector_manager = S3VectorManager(
            vector_bucket_name=vector_bucket,
            index_name=vector_index,
        )
        embedder = GeminiEmbedder(model_name="gemini-embedding-2")
        if args.model == "nova":
            rag_llm = BedrockNovaRAG()
            model_name_display = f"Amazon Nova ({rag_llm.model_name})"
        else:
            rag_llm = GeminiRAG()
            model_name_display = "Google Gemini (gemini-3.1-flash-lite)"
    except Exception as e:
        print(f"Initialization error: {e}")
        sys.exit(1)

    # 2. Embed Query
    try:
        print("Embedding question with Gemini embedding 2...")
        query_embedding = embedder.embed_text(args.question)
    except Exception as e:
        print(f"Failed to generate query embedding: {e}")
        sys.exit(1)

    # 3. Retrieve Context from Amazon S3 Vectors
    try:
        print(f"Retrieving top {args.limit} matching vector(s) from S3 Vectors...")
        results = vector_manager.search(
            query_embedding,
            limit=args.limit,
            filter_expr=filter_expr,
        )

        if not results:
            print("No matching document vectors found in S3 Vectors.")
            sys.exit(0)

        print(f"Successfully retrieved {len(results)} matching vector(s).")
    except Exception as e:
        print(f"Failed to search Amazon S3 Vectors: {e}")
        sys.exit(1)

    # 4. Resolve Context Images
    context_images = []
    print("\nMatched Document Citations:")
    for hit in results:
        meta = hit.get("metadata", {})
        doc_name = meta.get("document_name", "Unknown")
        page_num = meta.get("page_number", 1)
        distance = hit.get("distance")

        print(f"  - Document: {doc_name} | Page: {page_num} | Distance: {distance}")
        if meta.get("text_snippet"):
            print(f"    Snippet: {meta['text_snippet'][:120]}...")

        # Resolve page image from local data folder
        local_path = os.path.join(PROJECT_ROOT, "data", doc_name)
        if os.path.exists(local_path):
            b64_img = render_pdf_page_to_base64(local_path, page_num)
            if b64_img:
                context_images.append(b64_img)

    if not context_images:
        print("\nWarning: Could not render page images from local data folder for LLM input.")
        sys.exit(0)

    # 5. Generate Answer via Selected LLM
    try:
        print(f"\nSubmitting question and {len(context_images)} context page(s) to {model_name_display}...")
        answer = rag_llm.answer_question(args.question, context_images)
        print("\n" + "=" * 60)
        print("Answer:")
        print(answer)
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"Failed to generate answer from LLM: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
