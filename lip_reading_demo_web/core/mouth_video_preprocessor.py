from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import imageio
import numpy as np

from .face_landmark_detector.my_face_aligner import My_Face_Aligner
from .face_landmark_detector.my_face_cropper import My_Face_Cropper
from .face_landmark_detector.my_face_landmark_track_pose_detector import (
    My_Face_Landmark_Track_Pose_Detector,
)

FPS = 25
MOUTH_CROP_SIZE = 96
DEFAULT_DEVICE = "cuda:0"
DEFAULT_MIN_TARGET_IOU = 0.01

class MouthPreprocessingError(RuntimeError):
    """Raised when mouth ROI preprocessing cannot be completed."""

@dataclass(frozen=True)
class FaceBox:
    x: float
    y: float
    width: float
    height: float

    @classmethod
    def from_dict(cls, payload: dict[str, float]) -> "FaceBox":
        return cls(
            x=float(payload["x"]),
            y=float(payload["y"]),
            width=float(payload["width"]),
            height=float(payload["height"]),
        )

    def to_xyxy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.x + self.width, self.y + self.height], dtype=np.float32)

def run_mouth_preprocessing(
    source_video_path: Path,
    output_video_path: Path,
    work_dir: Path,
    start_seconds: float,
    end_seconds: float,
    face_box: dict[str, float],
    model_dir: Path | None = None,
    device: str = DEFAULT_DEVICE,
    min_target_iou: float = DEFAULT_MIN_TARGET_IOU,
) -> dict[str, Any]:
    """Create a normalized 96x96 mouth-region video for the selected face.

    The selected face is identified on the first frame of the selected segment by
    matching the frontend bounding box against the detected faces using IoU.
    The selected track is then followed through the segment, landmarks are
    interpolated when needed, and the final mouth ROI video is written without
    audio at 25 FPS.
    """
    if end_seconds <= start_seconds:
        raise MouthPreprocessingError("end_seconds must be greater than start_seconds.")
    if not source_video_path.exists():
        raise MouthPreprocessingError(f"Source video does not exist: {source_video_path}")

    model_dir = model_dir or Path(__file__).resolve().parent / "models"
    _validate_model_files(model_dir)

    work_dir.mkdir(parents=True, exist_ok=True)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    segment_path = work_dir / "selected_segment_25fps.mp4"
    duration = float(end_seconds - start_seconds)
    _create_25fps_segment(
        source_video_path=source_video_path,
        segment_path=segment_path,
        start_seconds=start_seconds,
        duration_seconds=duration,
    )

    frames = _read_video_frames(segment_path)
    if not frames:
        raise MouthPreprocessingError("No frames were decoded from the selected segment.")

    selected_bbox = FaceBox.from_dict(face_box).to_xyxy()
    detector = My_Face_Landmark_Track_Pose_Detector(model_dir=model_dir, device=device)

    track_id, best_iou, selected_faces = _track_selected_face(
        frames=frames,
        detector=detector,
        selected_bbox=selected_bbox,
        min_target_iou=min_target_iou,
    )

    landmarks = _extract_landmarks_for_track(
        frames=frames,
        detector=detector,
        selected_faces=selected_faces,
    )

    aligner = My_Face_Aligner(mean_face_path=model_dir / "20words_mean_face.npy")
    aligned = aligner.align_landmarks(landmarks)
    if aligned is None:
        raise MouthPreprocessingError("Face alignment failed for the selected track.")
    transformed_landmarks, transform_matrix = aligned

    cropper = My_Face_Cropper(
        crop_width=MOUTH_CROP_SIZE,
        crop_height=MOUTH_CROP_SIZE,
        start_idx=48,
        stop_idx=68,
        convert_gray=False,
    )
    mouth_frames = cropper.crop_faces(transform_matrix, transformed_landmarks, frames)
    if not mouth_frames or any(frame is None for frame in mouth_frames):
        raise MouthPreprocessingError("Mouth ROI cropping failed for one or more frames.")

    _save_video_no_audio(mouth_frames, output_video_path, fps=FPS)

    present_frames = sum(face is not None for face in selected_faces)
    return {
        "mouth_video_path": str(output_video_path),
        "segment_video_path": str(segment_path),
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "duration_seconds": duration,
        "fps": FPS,
        "mouth_crop_size": [MOUTH_CROP_SIZE, MOUTH_CROP_SIZE],
        "target_track_id": int(track_id),
        "target_first_frame_iou": float(best_iou),
        "target_face_presence_frames": int(present_frames),
        "total_frames": int(len(frames)),
        "face_box": face_box,
    }

def _validate_model_files(model_dir: Path) -> None:
    required = [
        "Resnet50_Final.pth",
        "2dfan2.pth",
        "bfm_lms.npy",
        "20words_mean_face.npy",
    ]
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        missing_list = ", ".join(missing)
        raise MouthPreprocessingError(
            f"Missing model files in {model_dir}: {missing_list}. "
            "Create core/models/ and copy the required files there."
        )

def _create_25fps_segment(
    source_video_path: Path,
    segment_path: Path,
    start_seconds: float,
    duration_seconds: float,
) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise MouthPreprocessingError("ffmpeg was not found in PATH.")

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(source_video_path),
        "-ss",
        f"{start_seconds:.3f}",
        "-t",
        f"{duration_seconds:.3f}",
        "-an",
        "-vf",
        f"fps={FPS}",
        "-vsync",
        "cfr",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(segment_path),
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise MouthPreprocessingError(process.stderr[-2500:] or "ffmpeg failed while creating the 25 FPS segment.")

def _read_video_frames(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise MouthPreprocessingError(f"Could not open video segment: {video_path}")

    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(frame)
    cap.release()
    return frames

def _track_selected_face(
    frames: list[np.ndarray],
    detector: My_Face_Landmark_Track_Pose_Detector,
    selected_bbox: np.ndarray,
    min_target_iou: float,
) -> tuple[int, float, list[np.ndarray | None]]:
    selected_track_id: int | None = None
    selected_faces: list[np.ndarray | None] = []
    best_iou = 0.0

    for frame_idx, frame in enumerate(frames):
        frame_info = detector.processAFrame_detect_track(frame)
        faces = frame_info.get("faces", np.empty((0, 15), dtype=np.float32))
        track_ids = frame_info.get("track_ids", [])

        if frame_idx == 0:
            if len(faces) == 0:
                raise MouthPreprocessingError("No face was detected on the first frame of the selected segment.")
            ious = [_iou(face[:4], selected_bbox) for face in faces]
            best_index = int(np.argmax(ious))
            best_iou = float(ious[best_index])
            if best_iou < min_target_iou:
                raise MouthPreprocessingError(
                    f"The selected frontend face box did not match any detected face on the first frame. "
                    f"Best IoU was {best_iou:.4f}."
                )
            selected_track_id = int(track_ids[best_index])

        selected_face = None
        if selected_track_id is not None:
            for idx, track_id in enumerate(track_ids):
                if int(track_id) == selected_track_id and idx < len(faces):
                    selected_face = faces[idx]
                    break
        selected_faces.append(selected_face)

    if selected_track_id is None:
        raise MouthPreprocessingError("Could not select a target face track.")

    if not any(face is not None for face in selected_faces):
        raise MouthPreprocessingError("The selected face track was not present in the selected segment.")

    return selected_track_id, best_iou, selected_faces

def _extract_landmarks_for_track(
    frames: list[np.ndarray],
    detector: My_Face_Landmark_Track_Pose_Detector,
    selected_faces: list[np.ndarray | None],
) -> list[np.ndarray | None]:
    landmarks: list[np.ndarray | None] = []
    for frame, face in zip(frames, selected_faces, strict=True):
        if face is None:
            landmarks.append(None)
            continue

        extra = detector.processAFrame_landmarks_pose(frame, np.asarray([face], dtype=np.float32))
        frame_landmarks = extra.get("landmarks", np.empty((0, 68, 2), dtype=np.float32))
        if len(frame_landmarks) == 0:
            landmarks.append(None)
        else:
            landmarks.append(frame_landmarks[0])

    valid_count = sum(item is not None for item in landmarks)
    if valid_count < 2:
        raise MouthPreprocessingError("Not enough valid landmarks were detected for the selected face track.")
    return landmarks

def _save_video_no_audio(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=["-crf", "16", "-preset", "slow"],
    ) as writer:
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_rgb)

def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in box_b[:4]]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union
