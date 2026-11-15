from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from .models import JobStatus, JobType

@dataclass
class JobRecord:
    video_id: str
    job_type: JobType
    status: JobStatus = "queued"
    message: str | None = None
    result: dict[str, Any] | None = None
    text: str | None = None
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[tuple[str, JobType], JobRecord] = {}
        self._lock = Lock()

    def set(
        self,
        video_id: str,
        job_type: JobType,
        status: JobStatus,
        message: str | None = None,
        result: dict[str, Any] | None = None,
        text: str | None = None,
    ) -> JobRecord:
        with self._lock:
            record = JobRecord(
                video_id=video_id,
                job_type=job_type,
                status=status,
                message=message,
                result=result,
                text=text,
            )
            self._jobs[(video_id, job_type)] = record
            return record

    def get(self, video_id: str, job_type: JobType) -> JobRecord:
        with self._lock:
            return self._jobs.get(
                (video_id, job_type),
                JobRecord(video_id=video_id, job_type=job_type, status="not_started"),
            )

job_store = JobStore()
