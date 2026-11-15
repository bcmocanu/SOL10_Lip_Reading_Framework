from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from pydantic import ValidationError
from starlette.responses import FileResponse

from core.lipreading import run_lipreading
from core.video_preprocessing import preprocess_video

from .. import storage
from ..config import settings
from ..jobs import JobRecord, job_store
from ..models import FaceBox, JobStatusResponse, PreprocessRequest, UploadVideoResponse

router = APIRouter()

def to_response(record: JobRecord) -> JobStatusResponse:
    return JobStatusResponse(
        video_id=record.video_id,
        job_type=record.job_type,
        status=record.status,
        message=record.message,
        result=record.result,
        text=record.text,
    )

def validate_video_file(file: UploadFile) -> None:
    content_type_is_video = bool(file.content_type and file.content_type.startswith("video/"))
    extension_is_video = Path(file.filename or "").suffix.lower() in storage.ALLOWED_VIDEO_EXTENSIONS

    if not content_type_is_video and not extension_is_video:
        raise HTTPException(status_code=400, detail="Please upload a video file.")

def validate_preprocess_request(payload: PreprocessRequest) -> None:
    duration = payload.end_seconds - payload.start_seconds
    if duration <= 0:
        raise HTTPException(status_code=400, detail="end_seconds must be greater than start_seconds.")
    if duration > settings.max_segment_seconds:
        raise HTTPException(
            status_code=400,
            detail=f"Selected segment must be at most {settings.max_segment_seconds} seconds.",
        )

def parse_face_box(face_box: str) -> FaceBox:
    try:
        return FaceBox.model_validate_json(face_box)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail={"message": "Invalid face_box JSON.", "errors": exc.errors()}) from exc

@router.post("/upload", response_model=UploadVideoResponse)
async def upload_video(file: UploadFile = File(...)) -> UploadVideoResponse:
    validate_video_file(file)
    metadata = await storage.save_upload(file)
    return UploadVideoResponse(**metadata)

@router.post("/preprocess", response_model=JobStatusResponse)
async def upload_and_start_preprocess(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    start_seconds: float = Form(...),
    end_seconds: float = Form(...),
    face_box: str = Form(...),
) -> JobStatusResponse:
    validate_video_file(file)

    payload = PreprocessRequest(
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        face_box=parse_face_box(face_box),
    )
    validate_preprocess_request(payload)

    metadata = await storage.save_upload(file)
    video_id = metadata["video_id"]

    record = job_store.set(video_id, "preprocess", "queued", "Preprocess queued.")
    background_tasks.add_task(run_preprocess_task, video_id, payload)
    return to_response(record)

@router.post("/{video_id}/preprocess", response_model=JobStatusResponse)
def start_preprocess(
    video_id: str,
    payload: PreprocessRequest,
    background_tasks: BackgroundTasks,
) -> JobStatusResponse:
    try:
        storage.ensure_video_exists(video_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    validate_preprocess_request(payload)

    record = job_store.set(video_id, "preprocess", "queued", "Preprocess queued.")
    background_tasks.add_task(run_preprocess_task, video_id, payload)
    return to_response(record)

@router.get("/{video_id}/preprocess-status", response_model=JobStatusResponse)
def get_preprocess_status(video_id: str) -> JobStatusResponse:
    return to_response(job_store.get(video_id, "preprocess"))

@router.get("/{video_id}/mouth-video")
def get_mouth_video(video_id: str) -> FileResponse:
    path = storage.mouth_video_path(video_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Mouth video is not available yet.")
    return FileResponse(path=path, media_type="video/mp4", filename=f"{video_id}_mouth_region.mp4")

@router.post("/{video_id}/lipread", response_model=JobStatusResponse)
def start_lipread(video_id: str, background_tasks: BackgroundTasks) -> JobStatusResponse:
    mouth_video = storage.mouth_video_path(video_id)
    if not mouth_video.exists():
        raise HTTPException(status_code=400, detail="Run preprocess before lip reading.")

    record = job_store.set(video_id, "lipread", "queued", "Lip reading queued.")
    background_tasks.add_task(run_lipread_task, video_id)
    return to_response(record)

@router.get("/{video_id}/lipread-status", response_model=JobStatusResponse)
def get_lipread_status(video_id: str) -> JobStatusResponse:
    return to_response(job_store.get(video_id, "lipread"))

def run_preprocess_task(video_id: str, payload: PreprocessRequest) -> None:
    job_store.set(video_id, "preprocess", "running", "Preprocessing video segment.")
    try:
        source_path = storage.input_video_path(video_id)
        target_dir = storage.reset_processed_dir(video_id)
        output_path = storage.mouth_video_path(video_id)

        result = preprocess_video(
            source_video_path=source_path,
            output_video_path=output_path,
            work_dir=target_dir,
            start_seconds=payload.start_seconds,
            end_seconds=payload.end_seconds,
            face_box=payload.face_box.model_dump(),
        )
        job_store.set(video_id, "preprocess", "completed", "Preprocess completed.", result=result)
    except Exception as exc:  # noqa: BLE001 - task errors must be visible in API status
        job_store.set(video_id, "preprocess", "failed", str(exc))

def run_lipread_task(video_id: str) -> None:
    job_store.set(video_id, "lipread", "running", "Running lip reading module.")
    try:
        mouth_video = storage.mouth_video_path(video_id)
        text = run_lipreading(mouth_video)
        job_store.set(
            video_id,
            "lipread",
            "completed",
            "Lip reading completed.",
            result={"text_length": len(text)},
            text=text,
        )
    except Exception as exc:  # noqa: BLE001 - task errors must be visible in API status
        job_store.set(video_id, "lipread", "failed", str(exc))
