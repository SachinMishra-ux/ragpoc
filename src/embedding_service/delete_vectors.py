import os
import sys
import argparse
from dotenv import load_dotenv

# Ensure root directory is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, "src", ".env"))

from src.embedding_service.s3_vector_manager import S3VectorManager


def main():
    parser = argparse.ArgumentParser(description="Manage and delete vectors in Amazon S3 Vectors")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List stored vectors and documents")
    group.add_argument("--document", type=str, help="Document filename whose vectors should be deleted (e.g. 'ELECTRONIC DEVICES AND CIRCUITS.pdf')")
    group.add_argument("--keys", nargs="+", help="Specific vector key(s) to delete")
    group.add_argument("--delete-index", action="store_true", help="Delete the entire S3 Vector Index")

    args = parser.parse_args()
    vector_manager = S3VectorManager()

    if args.list:
        print(f"Listing vectors from '{vector_manager.vector_bucket_name}/{vector_manager.index_name}'...")
        vectors = vector_manager.list_vectors(max_results=50, return_metadata=True)
        print(f"Found {len(vectors)} sample vector(s):")
        for v in vectors:
            meta = v.get("metadata", {})
            doc = meta.get("document_name", "unknown") if isinstance(meta, dict) else "N/A"
            page = meta.get("page_number", "N/A") if isinstance(meta, dict) else "N/A"
            print(f"  - Key: {v.get('key')} | Doc: {doc} | Page: {page}")

    elif args.document:
        print(f"Target document to delete: {args.document}")
        deleted_count = vector_manager.delete_document_vectors(args.document)
        print(f"Result: {deleted_count} vectors deleted.")

    elif args.keys:
        print(f"Deleting {len(args.keys)} key(s)...")
        vector_manager.delete_vectors(args.keys)
        print(f"Successfully deleted {len(args.keys)} vector(s).")

    elif args.delete_index:
        confirm = input(f"Are you sure you want to delete the ENTIRE index '{vector_manager.index_name}'? (yes/no): ")
        if confirm.strip().lower() == "yes":
            vector_manager.delete_index()
        else:
            print("Operation aborted.")


if __name__ == "__main__":
    main()
