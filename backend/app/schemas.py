from pydantic import BaseModel, Field


class SnippetRequest(BaseModel):
    code: str = Field(min_length=1, max_length=20000)
    language: str = "python"


class PRRequest(BaseModel):
    pr_url: str = Field(examples=["https://github.com/owner/repo/pull/123"])


class HunkResult(BaseModel):
    filename: str
    hunk_index: int
    risk: float
    label: str  # "defect-prone" | "likely clean"
    snippet: str


class FileResult(BaseModel):
    filename: str
    max_risk: float
    hunks: list[HunkResult]


class ReviewResponse(BaseModel):
    overall_risk: float
    files: list[FileResult]
    summary: str | None = None  # LLM-written review; None if no API key configured
    model_version: str
