from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import UploadFile

from .config import settings

ALLOWED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".mpeg",
    ".mpg",
}

def create_video_id() -> str:
    return str(uuid4())

def safe_extension(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in ALLOWED_VIDEO_EXTENSIONS:
        return suffix
    return ".mp4"

def video_dir(video_id: str) -> Path:
    return settings.upload_dir / video_id

def processed_video_dir(video_id: str) -> Path:
    return settings.processed_dir / video_id

def input_video_path(video_id: str) -> Path:
    meta = read_metadata(video_id)
    if meta and meta.get("input_path"):
        return Path(meta["input_path"])
    raise FileNotFoundError(f"Video {video_id} was not found")

def mouth_video_path(video_id: str) -> Path:
    return processed_video_dir(video_id) / "mouth_region.mp4"

async def save_upload(file: UploadFile) -> dict[str, Any]:
    video_id = create_video_id()
    target_dir = video_dir(video_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    extension = safe_extension(file.filename or "input.mp4")
    target_path = target_dir / f"input{extension}"

    size = 0
    with target_path.open("wb") as out_file:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            out_file.write(chunk)

    metadata = {
        "video_id": video_id,
        "filename": file.filename or target_path.name,
        "content_type": file.content_type,
        "size_bytes": size,
        "input_path": str(target_path),
    }
    write_metadata(video_id, metadata)
    return metadata

def write_metadata(video_id: str, metadata: dict[str, Any]) -> None:
    target_dir = video_dir(video_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

def read_metadata(video_id: str) -> dict[str, Any] | None:
    metadata_path = video_dir(video_id) / "metadata.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))

def ensure_video_exists(video_id: str) -> None:
    source_path = input_video_path(video_id)
    if not source_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {source_path}")

def reset_processed_dir(video_id: str) -> Path:
    target_dir = processed_video_dir(video_id)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir
