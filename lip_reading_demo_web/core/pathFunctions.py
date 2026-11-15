from __future__ import annotations

class MyProgress:
    """Small compatibility helper for the original preprocessing scripts."""

    def __init__(self, name: str, maxVal: int = 0, printStep: int = 5) -> None:
        self.name = name
        self.maxVal = maxVal
        self.printStep = max(1, printStep)

    def update_progress(self, value: int) -> None:
        
        return None

def frames_to_timecode(frame_number: int | float, fps: int | float) -> str:
    """Convert a frame index to HH:MM:SS.mmm for debug messages."""
    if fps <= 0:
        return "00:00:00.000"
    total_seconds = float(frame_number) / float(fps)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    millis = int(round((total_seconds - int(total_seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
