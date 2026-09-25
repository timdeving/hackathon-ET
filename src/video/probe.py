"""Read a video's frame rate, frame count and size exactly the way the harness does.

run_submission.py timestamps frame i as i / CAP_PROP_FPS and takes the duration as
CAP_PROP_FRAME_COUNT / CAP_PROP_FPS, both read with OpenCV. Reading them from the same source
keeps our timestamps in agreement with the harness's.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoInfo:
    name: str  # file name: the key used in predictions.json
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps


def probe_video(path: str | Path) -> VideoInfo:
    """Return the metadata run_submission.video_meta() would compute for this file."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    try:
        return VideoInfo(
            name=Path(path).name,
            fps=float(cap.get(cv2.CAP_PROP_FPS) or 25.0),  # same fallback as the harness
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()
