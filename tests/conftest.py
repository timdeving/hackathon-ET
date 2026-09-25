"""Shared fixtures. Tests use a tiny synthetic clip and never touch the real sample videos."""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class TinyVideo:
    path: Path
    fps: float
    n_frames: int
    size: tuple[int, int]  # (width, height)


@pytest.fixture(scope="session")
def tiny_video(tmp_path_factory) -> TinyVideo:
    """A 2-second 64x48 clip, alone in its folder.

    Every frame differs from its neighbours (the grey level changes and a white square moves),
    so frame-numbering checks can tell frames apart.
    """
    video = TinyVideo(
        path=tmp_path_factory.mktemp("videos") / "tiny.mp4", fps=25.0, n_frames=50, size=(64, 48)
    )
    width, height = video.size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video.path), fourcc, video.fps, video.size)
    assert writer.isOpened(), "OpenCV cannot write mp4v video"
    for i in range(video.n_frames):
        frame = np.full((height, width, 3), 20 + (37 * i) % 200, np.uint8)
        x = i % (width - 8)
        frame[20:28, x : x + 8] = 255
        writer.write(frame)
    writer.release()
    return video


@pytest.fixture(scope="session")
def run_python():
    """Run `python ARGS...` from the repository root; returns the finished process."""

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )

    return run
