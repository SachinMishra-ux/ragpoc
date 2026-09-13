from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., description="The query question for the academic and engineering documents.")
    thread_id: str | None = Field(
        default=None,
        description="Conversation thread identifier for multi-turn conversation persistence via SQLite checkpointer."
    )
    limit: int = Field(default=3, ge=1, le=10, description="The maximum number of matching pages to retrieve.")
    filter: dict | None = Field(
        default=None,
        description="Optional metadata filter dictionary for Amazon S3 Vectors (e.g., {'document_name': 'ELECTRONIC DEVICES AND CIRCUITS.pdf'})."
    )
    document_name: str | None = Field(
        default=None,
        description="Convenience filter to restrict search to a specific document filename."
    )
    page_number: int | None = Field(
        default=None,
        description="Convenience filter to restrict search to a specific page number."
    )
    image_base64: str | None = Field(
        default=None,
        description="Optional base64-encoded image string (screenshot/chart/table) for multimodal query."
    )


class QueryResponse(BaseModel):
    question: str
    answer: str
    thread_id: str = Field(..., description="The active conversation thread ID.")
    tool_called: bool = Field(
        default=False,
        description="Indicates whether the agent dynamically invoked the S3 Vectors tool or answered directly."
    )
    user_image: str | None = Field(
        default=None,
        description="Optional base64 data URI of the user-provided query image/screenshot."
    )
    pages_retrieved: int = Field(default=0)
    images: list[str] = Field(default=[], description="List of base64 encoded images of matching pages.")
    sources: list[dict] = Field(
        default=[],
        description="List of metadata dictionaries for matched document pages (document name, page number, distance, etc.)."
    )


class UploadResponse(BaseModel):
    status: str = Field(..., description="Upload status, e.g., 'success' or 'error'")
    filename: str = Field(..., description="Original filename of the uploaded file")
    bucket: str = Field(..., description="Target S3 bucket name")
    s3_key: str = Field(..., description="S3 object key")
    message: str = Field(..., description="Descriptive status message")


class DeleteDocumentResponse(BaseModel):
    status: str = Field(..., description="Status of deletion, e.g., 'success'")
    document_name: str = Field(..., description="Name of the document deleted")
    vectors_deleted: int = Field(..., description="Number of vector chunks deleted from S3 Vectors")
    s3_object_deleted: bool = Field(..., description="Whether the raw PDF was deleted from S3 document storage")
    message: str = Field(..., description="Descriptive status message")


class DeleteVectorsRequest(BaseModel):
    keys: list[str] = Field(..., description="List of unique vector keys to delete from S3 Vectors")


class DeleteVectorsResponse(BaseModel):
    status: str = Field(..., description="Status of deletion")
    keys_deleted: int = Field(..., description="Number of vector keys deleted")
    message: str = Field(..., description="Descriptive status message")


class PurgeIndexResponse(BaseModel):
    status: str = Field(..., description="Status of the purge/reset operation")
    index_name: str = Field(..., description="Vector index name")
    recreated: bool = Field(..., description="Whether a fresh index was recreated")
    message: str = Field(..., description="Descriptive status message")

