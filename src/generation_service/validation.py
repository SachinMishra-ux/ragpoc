from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., description="The query question for the financial documents.")
    thread_id: str | None = Field(
        default=None,
        description="Conversation thread identifier for multi-turn conversation persistence via SQLite checkpointer."
    )
    limit: int = Field(default=3, ge=1, le=10, description="The maximum number of matching pages to retrieve.")
    filter: dict | None = Field(
        default=None,
        description="Optional metadata filter dictionary for Amazon S3 Vectors (e.g., {'document_name': 'EY_Financial_report_2025.pdf'})."
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
