from __future__ import annotations

from pathlib import Path
from typing import Any

from .mouth_video_preprocessor import MouthPreprocessingError, run_mouth_preprocessing

class PreprocessingError(RuntimeError):
    pass

def preprocess_video(
    source_video_path: Path,
    output_video_path: Path,
    work_dir: Path,
    start_seconds: float,
    end_seconds: float,
    face_box: dict[str, float],
) -> dict[str, Any]:
    """Run the SOL10 mouth-region preprocessing pipeline.

    The pipeline converts the selected segment to 25 FPS, selects the face track
    that matches the frontend bounding box on the first frame, computes facial
    landmarks, aligns the face, crops the mouth region to 96x96, and saves a
    no-audio MP4 that can be passed to the lip-reading model.
    """
    try:
        return run_mouth_preprocessing(
            source_video_path=Path(source_video_path),
            output_video_path=Path(output_video_path),
            work_dir=Path(work_dir),
            start_seconds=float(start_seconds),
            end_seconds=float(end_seconds),
            face_box=face_box,
            model_dir=Path(__file__).resolve().parent / "models",
            device="cuda:0",
        )
    except MouthPreprocessingError as exc:
        raise PreprocessingError(str(exc)) from exc
