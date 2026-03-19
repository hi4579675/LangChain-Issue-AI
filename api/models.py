from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class SourceChunk(BaseModel):
    issue_number: int
    content: str
    chunk_type: str
    score: float
    issue_url: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class SyncRequest(BaseModel):
    label: str = "bug"
    max_pages: int = Field(default=10, ge=1, le=100)
