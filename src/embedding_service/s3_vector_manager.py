import os
import uuid
import boto3
from botocore.exceptions import ClientError


class S3VectorManager:
    def __init__(
        self,
        vector_bucket_name: str | None = None,
        index_name: str | None = None,
        region_name: str | None = None,
    ):
        """
        Initializes the Amazon S3 Vectors client and target bucket/index configurations.
        Uses the AWS S3 Vectors service ('s3vectors') supported natively in boto3.
        """
        self.region_name = region_name or os.getenv("AWS_REGION", "eu-north-1")
        self.vector_bucket_name = (
            vector_bucket_name
            or os.getenv("S3_VECTOR_BUCKET_NAME")
            or os.getenv("VECTOR_BUCKET_NAME")
            or "financial-rag-vectors"
        )
        self.index_name = (
            index_name
            or os.getenv("S3_VECTOR_INDEX_NAME")
            or os.getenv("INDEX_NAME")
            or os.getenv("COLLECTION_NAME")
            or "financial-index"
        )
        
        # Initialize boto3 S3 Vectors client
        self.client = boto3.client("s3vectors", region_name=self.region_name)
        # Standard S3 client for document retrieval
        self.s3_client = boto3.client("s3", region_name=self.region_name)

    def ensure_vector_bucket(self):
        """
        Ensures that the S3 Vector Bucket exists.
        Creates it via create_vector_bucket if it does not already exist.
        """
        try:
            self.client.get_vector_bucket(vectorBucketName=self.vector_bucket_name)
            print(f"S3 Vector Bucket '{self.vector_bucket_name}' already exists.")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ["NotFoundException", "ResourceNotFoundException", "404"]:
                print(f"Creating S3 Vector Bucket '{self.vector_bucket_name}' in region '{self.region_name}'...")
                try:
                    self.client.create_vector_bucket(vectorBucketName=self.vector_bucket_name)
                    print(f"✅ S3 Vector Bucket '{self.vector_bucket_name}' created successfully.")
                except ClientError as ce:
                    if ce.response.get("Error", {}).get("Code") != "ConflictException":
                        raise ce
            elif error_code in ["AccessDeniedException", "AccessDenied"]:
                print("\n" + "!" * 70)
                print("❌ AWS IAM AccessDeniedException:")
                print(f"   {e.response.get('Error', {}).get('Message', str(e))}")
                print("\n💡 SOLUTION: Amazon S3 Vectors requires IAM permissions under the 's3vectors' namespace.")
                print("   Standard S3 permissions (s3:*) do not cover s3vectors actions.")
                print("   Please attach an IAM policy granting 's3vectors:*' to your IAM user/role.")
                print("!" * 70 + "\n")
                raise e
            else:
                raise e

    def ensure_index(
        self,
        dimension: int = 1024,
        distance_metric: str = "cosine",
        non_filterable_metadata_keys: list[str] | None = None,
    ):
        """
        Ensures that the S3 Vector Index exists with the specified dimension and distance metric.
        If an existing index has mismatched dimensions, it can be recreated.
        All metadata fields NOT listed in non_filterable_metadata_keys remain 100% filterable by default.
        """
        self.ensure_vector_bucket()

        try:
            index_info = self.client.get_index(
                vectorBucketName=self.vector_bucket_name,
                indexName=self.index_name,
            )
            existing_dim = index_info.get("index", {}).get("dimension")
            if existing_dim != dimension:
                print(
                    f"Dimension mismatch for index '{self.index_name}': "
                    f"existing is {existing_dim}, required is {dimension}."
                )
                print(f"Recreating S3 Vector Index '{self.index_name}' with dimension {dimension}...")
                self.client.delete_index(
                    vectorBucketName=self.vector_bucket_name,
                    indexName=self.index_name,
                )
                self._create_index(dimension, distance_metric, non_filterable_metadata_keys)
            else:
                print(f"S3 Vector Index '{self.index_name}' exists with dimension {dimension}.")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ["NotFoundException", "ResourceNotFoundException", "404"]:
                print(f"Index '{self.index_name}' not found. Creating index...")
                self._create_index(dimension, distance_metric, non_filterable_metadata_keys)
            elif error_code in ["AccessDeniedException", "AccessDenied"]:
                print("\n" + "!" * 70)
                print("❌ AWS IAM AccessDeniedException:")
                print(f"   {e.response.get('Error', {}).get('Message', str(e))}")
                print("\n💡 SOLUTION: Amazon S3 Vectors requires IAM permissions under the 's3vectors' namespace.")
                print("   Please attach an IAM policy granting 's3vectors:*' to your IAM user/role.")
                print("!" * 70 + "\n")
                raise e
            elif error_code == "ConflictException":
                pass
            else:
                raise e

    def _create_index(
        self,
        dimension: int,
        distance_metric: str = "cosine",
        non_filterable_metadata_keys: list[str] | None = None,
    ):
        create_kwargs = {
            "vectorBucketName": self.vector_bucket_name,
            "indexName": self.index_name,
            "dataType": "float32",
            "dimension": dimension,
            "distanceMetric": distance_metric,
        }
        if non_filterable_metadata_keys:
            create_kwargs["metadataConfiguration"] = {
                "nonFilterableMetadataKeys": non_filterable_metadata_keys
            }

        print(f"Creating S3 Vector Index '{self.index_name}' (dim={dimension}, metric={distance_metric})...")
        self.client.create_index(**create_kwargs)
        print(f"✅ S3 Vector Index '{self.index_name}' created successfully.")

    def insert_vectors(self, vectors: list[dict], batch_size: int = 50):
        """
        Batches and writes vectors to the S3 Vector Index using the put_vectors API.
        Each vector item must have:
        - 'key': unique string identifier
        - 'data': {'float32': [float, ...]}
        - 'metadata': dict of filterable and non-filterable metadata
        """
        total_vectors = len(vectors)
        print(f"Inserting {total_vectors} vector(s) into S3 Vectors '{self.index_name}' in batches of {batch_size}...")

        for i in range(0, total_vectors, batch_size):
            batch = vectors[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = ((total_vectors - 1) // batch_size) + 1
            print(f"Uploading batch {batch_num}/{total_batches} ({len(batch)} vectors)...")
            self.client.put_vectors(
                vectorBucketName=self.vector_bucket_name,
                indexName=self.index_name,
                vectors=batch,
            )

        print("✅ Vector insertion complete.")

    def insert_image_embeddings(
        self,
        embeddings: list[list[float]],
        base64_images: list[str],
        metadata_list: list[dict] | None = None,
        batch_size: int = 20,
    ):
        """
        Formats image embeddings and metadata, ensuring dimensions match and batching upserts to S3 Vectors.
        Metadata fields (document_name, page_number, total_pages, source_uri, etc.) are stored as filterable attributes.
        """
        if not embeddings:
            return

        vector_size = len(embeddings[0])
        self.ensure_index(dimension=vector_size)

        vectors = []
        for idx, (emb, b64_img) in enumerate(zip(embeddings, base64_images)):
            extra_meta = metadata_list[idx] if metadata_list and idx < len(metadata_list) else {}
            
            doc_name = extra_meta.get("document_name", "doc")
            page_num = extra_meta.get("page_number", idx + 1)
            clean_doc = doc_name.replace(" ", "_").replace("/", "_")
            vector_key = f"{clean_doc}#page_{page_num}#{uuid.uuid4().hex[:8]}"

            # All metadata fields below are filterable by default in S3 Vectors
            meta = {
                "document_name": str(extra_meta.get("document_name", doc_name)),
                "page_number": int(extra_meta.get("page_number", page_num)),
                "total_pages": int(extra_meta.get("total_pages", len(embeddings))),
                "source_uri": str(extra_meta.get("source_uri", "")),
                "source_type": str(extra_meta.get("source_type", "s3")),
                "file_size": int(extra_meta.get("file_size", 0)),
                "ingested_at": str(extra_meta.get("ingested_at", "")),
            }
            if "source_bucket" in extra_meta and extra_meta["source_bucket"]:
                meta["source_bucket"] = str(extra_meta["source_bucket"])
            if "source_key" in extra_meta and extra_meta["source_key"]:
                meta["source_key"] = str(extra_meta["source_key"])
            if "text_snippet" in extra_meta and extra_meta["text_snippet"]:
                # Filterable metadata string limit per vector is 2KB
                meta["text_snippet"] = str(extra_meta["text_snippet"])[:1200]

            vectors.append(
                {
                    "key": vector_key,
                    "data": {"float32": [float(val) for val in emb]},
                    "metadata": meta,
                }
            )

        self.insert_vectors(vectors, batch_size=batch_size)

    def search(
        self,
        query_embedding: list[float],
        limit: int = 3,
        filter_expr: dict | None = None,
    ) -> list[dict]:
        """
        Executes approximate nearest neighbor search in S3 Vectors with tandem metadata filtering.
        Returns a list of dicts with:
        - 'key': vector key
        - 'distance': float similarity distance
        - 'metadata': metadata dictionary
        """
        print(f"Searching S3 Vectors '{self.index_name}' for top {limit} matches...")
        query_params = {
            "vectorBucketName": self.vector_bucket_name,
            "indexName": self.index_name,
            "queryVector": {"float32": [float(x) for x in query_embedding]},
            "topK": limit,
            "returnMetadata": True,
            "returnDistance": True,
        }

        if filter_expr:
            print(f"Applying metadata filter: {filter_expr}")
            query_params["filter"] = filter_expr

        response = self.client.query_vectors(**query_params)

        results = []
        for hit in response.get("vectors", []):
            results.append(
                {
                    "key": hit.get("key"),
                    "distance": hit.get("distance"),
                    "metadata": hit.get("metadata", {}),
                }
            )

        print(f"Found {len(results)} matching vector(s).")
        return results

    def list_vectors(
        self,
        max_results: int = 50,
        return_metadata: bool = True,
        return_data: bool = False,
    ) -> list[dict]:
        """
        Lists vectors stored inside the S3 Vector Index.
        Returns a list of dicts containing 'key', 'metadata', and optionally 'data'.
        """
        response = self.client.list_vectors(
            vectorBucketName=self.vector_bucket_name,
            indexName=self.index_name,
            maxResults=max_results,
            returnMetadata=return_metadata,
            returnData=return_data,
        )
        return response.get("vectors", [])

    def delete_vectors(self, keys: list[str]):
        """Deletes vectors by key from the index."""
        if not keys:
            return
        # Delete in batches of up to 500 keys
        for i in range(0, len(keys), 500):
            batch = keys[i : i + 500]
            self.client.delete_vectors(
                vectorBucketName=self.vector_bucket_name,
                indexName=self.index_name,
                keys=batch,
            )

    def delete_document_vectors(self, document_name: str) -> int:
        """
        Finds and deletes all vectors belonging to a specific document name.
        Paginates through index vectors, matches by metadata or key prefix, and deletes them.
        Returns the number of deleted vectors.
        """
        clean_doc = document_name.replace(" ", "_").replace("/", "_")
        keys_to_delete = []
        next_token = None

        while True:
            kwargs = {
                "vectorBucketName": self.vector_bucket_name,
                "indexName": self.index_name,
                "maxResults": 500,
                "returnMetadata": True,
            }
            if next_token:
                kwargs["nextToken"] = next_token

            resp = self.client.list_vectors(**kwargs)
            for vec in resp.get("vectors", []):
                meta = vec.get("metadata", {})
                doc = meta.get("document_name") if isinstance(meta, dict) else None
                key = vec.get("key", "")
                if doc == document_name or key.startswith(f"{clean_doc}#") or key.startswith(f"{document_name}#"):
                    keys_to_delete.append(key)

            next_token = resp.get("nextToken")
            if not next_token:
                break

        if keys_to_delete:
            print(f"Deleting {len(keys_to_delete)} vector(s) for document '{document_name}'...")
            self.delete_vectors(keys_to_delete)
            print(f"✅ Deleted {len(keys_to_delete)} vector(s) for '{document_name}'.")
        else:
            print(f"ℹ️ No vectors found for document '{document_name}'.")

        return len(keys_to_delete)

    def delete_index(self):
        """Deletes the entire vector index from the vector bucket."""
        print(f"Deleting S3 Vector Index '{self.index_name}'...")
        self.client.delete_index(
            vectorBucketName=self.vector_bucket_name,
            indexName=self.index_name,
        )
        print(f"✅ Deleted S3 Vector Index '{self.index_name}'.")

