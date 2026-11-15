from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator

class FaceBox(BaseModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)

class PreprocessRequest(BaseModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    face_box: FaceBox

    @field_validator("end_seconds")
    @classmethod
    def validate_end(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("end_seconds must be greater than 0")
        return value

class UploadVideoResponse(BaseModel):
    video_id: str
    filename: str
    content_type: str | None
    size_bytes: int

JobType = Literal["preprocess", "lipread"]
JobStatus = Literal["not_started", "queued", "running", "completed", "failed"]

class JobStatusResponse(BaseModel):
    video_id: str
    job_type: JobType
    status: JobStatus
    message: str | None = None
    result: dict[str, Any] | None = None
    text: str | None = None
