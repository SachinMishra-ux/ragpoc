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
    model_provider: str = Field(
        default="nova",
        description="LLM provider: 'nova' (Amazon Nova 2 Lite via Bedrock) or 'gemini' (Google Gemini 3.1 Flash Lite)."
    )
    llm_model: str | None = Field(
        default=None,
        description="Specific model identifier to use (e.g., 'amazon.nova-2-lite-v1:0' or 'gemini-3.1-flash-lite'). Defaults to provider's primary model if not specified."
    )


class QueryResponse(BaseModel):
    question: str
    answer: str
    thread_id: str = Field(..., description="The active conversation thread ID.")
    tool_called: bool = Field(
        default=False,
        description="Indicates whether the agent dynamically invoked the S3 Vectors tool or answered directly."
    )
    model_provider: str = Field(
        default="nova",
        description="LLM provider used for inference: 'nova' or 'gemini'."
    )
    model_used: str = Field(
        default="amazon.nova-2-lite-v1:0",
        description="Model identifier used to produce the answer."
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
    file_type: str = Field(default="pdf", description="Document type: pdf, docx, doc, pptx, ppt")
    sqs_queued: bool = Field(default=False, description="Whether message was pushed to SQS queue")
    message: str = Field(..., description="Descriptive status message")


class UploadedFileInfo(BaseModel):
    filename: str = Field(..., description="Original filename of the uploaded file")
    s3_key: str = Field(..., description="S3 object key")
    bucket: str = Field(..., description="Target S3 bucket name")
    file_size: int = Field(default=0, description="Size of file in bytes")
    file_type: str = Field(default="document", description="Document type: pdf, docx, doc, pptx, ppt")
    sqs_queued: bool = Field(default=False, description="Whether message was queued in SQS")
    status: str = Field(default="uploaded", description="Status of the individual file upload")


class BulkUploadResponse(BaseModel):
    status: str = Field(..., description="Overall batch upload status ('success', 'partial', or 'error')")
    total_files: int = Field(..., description="Total number of files received in batch")
    successful_uploads: int = Field(..., description="Number of files successfully uploaded to S3 and queued")
    failed_uploads: int = Field(default=0, description="Number of files that failed upload or validation")
    files: list[UploadedFileInfo] = Field(default=[], description="Detailed breakdown of uploaded files")
    message: str = Field(..., description="Summary status message")


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

