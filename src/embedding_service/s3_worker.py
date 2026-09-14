import os
import sys
import json
import time
import signal
import tempfile
import urllib.parse
from datetime import datetime, timezone
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Ensure root directory and .venv site-packages are in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
venv_site = os.path.join(PROJECT_ROOT, ".venv", "lib", "python3.12", "site-packages")
if os.path.exists(venv_site) and venv_site not in sys.path:
    sys.path.insert(0, venv_site)

# Load environment variables
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, "src", ".env"))

from src.embedding_service.document_processor import (
    iter_document_pages,
    image_to_base64,
    get_document_page_count,
    SUPPORTED_EXTENSIONS,
)
from src.embedding_service.embedder import GeminiEmbedder
from src.embedding_service.s3_vector_manager import S3VectorManager

AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")
SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "academic-rag-documents-493116771407")
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

running = True


def handle_exit(signum, frame):
    global running
    print("\nReceived termination signal. Shutting down S3 worker gracefully...")
    running = False


signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)


def ingest_document_file(doc_path, bucket_name, object_key, vector_manager, embedder, batch_size=5):
    """
    Processes an academic document (PDF, Word, PowerPoint) downloaded from S3 page-by-page/slide-by-slide,
    generates Gemini embedding 2 vectors, and batch upserts into Amazon S3 Vectors with rich filterable metadata.
    """
    try:
        filename = os.path.basename(object_key)
        ext = os.path.splitext(filename)[1].lower()
        file_size = os.path.getsize(doc_path)
        print(f"\n--- Ingesting s3://{bucket_name}/{object_key} (Format: {ext.upper()}) ---")

        total_pages = get_document_page_count(doc_path)
        if total_pages == 0:
            print(f"No pages or slides found in: {filename}")
            return False

        print(f"Streaming {filename} page-by-page / slide-by-slide (Total: {total_pages})...")
        batch_embeddings = []
        batch_images = []
        batch_metadata = []
        processed_count = 0

        for page_num, total, page_img, page_text in iter_document_pages(doc_path, extract_text=True):
            print(f"Embedding page/slide {page_num}/{total} for {filename}...")
            b64_str = image_to_base64(page_img)
            emb = embedder.embed_image(page_img)

            # Rich filterable metadata for S3 Vectors
            meta = {
                "document_name": filename,
                "file_extension": ext,
                "page_number": page_num,
                "total_pages": total,
                "source_bucket": bucket_name,
                "source_key": object_key,
                "source_uri": f"s3://{bucket_name}/{object_key}",
                "source_type": "s3",
                "file_size": file_size,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "text_snippet": page_text[:1200] if page_text else "",
            }

            batch_embeddings.append(emb)
            batch_images.append(b64_str)
            batch_metadata.append(meta)
            processed_count += 1

            # Mild rate-limiting pause to stay within Gemini RPM limits
            time.sleep(1.0)

            if len(batch_embeddings) >= batch_size or page_num == total:
                vector_manager.insert_image_embeddings(
                    embeddings=batch_embeddings,
                    base64_images=batch_images,
                    metadata_list=batch_metadata,
                    batch_size=batch_size,
                )
                print(f"✅ Upserted {len(batch_embeddings)} pages/slides to S3 Vectors (Progress: {page_num}/{total})")
                batch_embeddings = []
                batch_images = []
                batch_metadata = []

        print(
            f"🎉 Successfully completed ingestion for {filename} ({processed_count} pages/slides) "
            f"into S3 Vector Index '{S3_VECTOR_INDEX_NAME}'"
        )
        return True
    except Exception as e:
        print(f"Error during ingestion of {object_key}: {e}")
        return False


# Alias for backward compatibility
ingest_pdf_file = ingest_document_file


# Track recently processed documents to prevent duplicate processing if twin SQS messages arrive
_recently_processed = {}  # (bucket, key) -> timestamp
DEDUPLICATION_WINDOW_SECONDS = 180  # 3 minutes cooldown


def _process_single_s3_object(bucket_name, object_key, s3_client, vector_manager, embedder):
    """Downloads a single object from S3 and triggers multi-format ingestion."""
    ext = os.path.splitext(object_key)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        print(f"Ignoring unsupported file format '{ext}': {object_key}. Supported: {SUPPORTED_EXTENSIONS}")
        return True

    dedup_key = (bucket_name, object_key)
    now = time.time()
    last_processed = _recently_processed.get(dedup_key, 0)
    if now - last_processed < DEDUPLICATION_WINDOW_SECONDS:
        print(f"\n⚡ Skipping duplicate SQS notification for s3://{bucket_name}/{object_key} (already processed {int(now - last_processed)}s ago).")
        return True

    print(f"\nProcessing detected document in S3: s3://{bucket_name}/{object_key}")
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
    os.close(tmp_fd)
    try:
        print(f"Downloading s3://{bucket_name}/{object_key}...")
        s3_client.download_file(bucket_name, object_key, tmp_path)
        print(f"Downloaded to {tmp_path}")

        ingested = ingest_document_file(tmp_path, bucket_name, object_key, vector_manager, embedder)
        if ingested:
            _recently_processed[dedup_key] = time.time()
        return ingested
    except Exception as e:
        print(f"Error downloading or processing s3://{bucket_name}/{object_key}: {e}")
        return False
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def process_message(message_body, s3_client, vector_manager, embedder):
    """
    Parses an SQS message body. Supports:
    1. Standard AWS S3 Event notifications (`{"Records": [{"eventName": "ObjectCreated:...", "s3": {...}}]}`)
    2. Direct API notifications (`{"bucket": "...", "key": "..."}` or `{"files": [...]}`)
    """
    try:
        data = json.loads(message_body)
    except Exception as e:
        print(f"Error parsing SQS message JSON: {e}")
        return True  # Acknowledge unparseable messages to remove them from queue

    if "Event" in data and data["Event"] == "s3:TestEvent":
        print("Received S3 Test Event. Connection verified.")
        return True

    # Check for direct API notification payload
    if "bucket" in data and "key" in data:
        bucket = data["bucket"]
        key = urllib.parse.unquote_plus(data["key"])
        return _process_single_s3_object(bucket, key, s3_client, vector_manager, embedder)

    records = data.get("Records", [])
    if not records:
        print(f"Message contains no recognized records or direct payload: {data}. Skipping.")
        return True

    success = True
    for record in records:
        event_name = record.get("eventName", "")
        if event_name and not event_name.startswith("ObjectCreated:"):
            print(f"Ignoring non-creation event: {event_name}")
            continue

        s3_info = record.get("s3", {})
        bucket_name = s3_info.get("bucket", {}).get("name")
        raw_key = s3_info.get("object", {}).get("key", "")
        object_key = urllib.parse.unquote_plus(raw_key)

        if not bucket_name or not object_key:
            continue

        ok = _process_single_s3_object(bucket_name, object_key, s3_client, vector_manager, embedder)
        if not ok:
            success = False

    return success


def start_worker():
    """Main worker loop that long-polls SQS for S3 event notifications."""
    if not SQS_QUEUE_URL:
        print("ERROR: SQS_QUEUE_URL environment variable is not set. Exiting.")
        sys.exit(1)

    print("=" * 60)
    print("🚀 Starting S3 Ingestion Worker Daemon (Amazon S3 Vectors)")
    print(f"  AWS Region:          {AWS_REGION}")
    print(f"  SQS Queue:           {SQS_QUEUE_URL}")
    print(f"  S3 Vector Bucket:    {S3_VECTOR_BUCKET_NAME}")
    print(f"  S3 Vector Index:     {S3_VECTOR_INDEX_NAME}")
    print("=" * 60)

    # Initialize clients
    sqs_client = boto3.client("sqs", region_name=AWS_REGION)
    s3_client = boto3.client("s3", region_name=AWS_REGION)
    vector_manager = S3VectorManager(
        vector_bucket_name=S3_VECTOR_BUCKET_NAME,
        index_name=S3_VECTOR_INDEX_NAME,
        region_name=AWS_REGION,
    )
    embedder = GeminiEmbedder(model_name="gemini-embedding-2")

    print("\nWaiting for S3 upload events (long-polling SQS 20s)... Press Ctrl+C to stop.")
    while running:
        try:
            response = sqs_client.receive_message(
                QueueUrl=SQS_QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,
                VisibilityTimeout=900,
            )

            messages = response.get("Messages", [])
            if not messages:
                continue

            for message in messages:
                receipt_handle = message["ReceiptHandle"]
                body = message["Body"]

                print("\n📨 Received message from SQS.")
                processed_ok = process_message(body, s3_client, vector_manager, embedder)

                if processed_ok:
                    sqs_client.delete_message(
                        QueueUrl=SQS_QUEUE_URL,
                        ReceiptHandle=receipt_handle,
                    )
                    print("✅ SQS message successfully processed and removed from queue.")
                else:
                    print("⚠️ Ingestion had errors. Message left in queue for retry.")

        except ClientError as e:
            print(f"AWS ClientError: {e}")
            time.sleep(5)
        except Exception as e:
            print(f"Unexpected error in worker loop: {e}")
            time.sleep(5)

    print("S3 Ingestion Worker stopped.")


if __name__ == "__main__":
    start_worker()
