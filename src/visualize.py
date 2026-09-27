"""Draw what the system found onto video frames: the live demo's annotated playback and the
website's annotated sample videos. The submission itself never draws.

Each frame shows the tracked road users (Part A's tracks, in the video's own pixels, scaled to
the image drawn on), a banner with the time and the events under way, the top arm's traffic
light as read, and Part B's risk as a bar that turns red at the alarm threshold. Frames are
encoded as H.264 through the system's ffmpeg, so browsers can play the result.
"""
from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from src.rules.common import AMBER, GREEN, RED

# The road users the tracker follows (COCO ids), with a colour each (BGR).
ROAD_USERS = {
    0: ("person", (0, 165, 255)),
    1: ("bicycle", (80, 200, 80)),
    2: ("car", (255, 160, 60)),
    3: ("motorcycle", (80, 200, 80)),
    5: ("bus", (220, 90, 200)),
    7: ("truck", (200, 120, 40)),
}
EVENT_COLOURS = {  # BGR; the classes we predict get distinct colours
    "jaywalking": (0, 165, 255),
    "failure_to_yield": (60, 60, 230),
    "red_light": (40, 40, 200),
    "stop_line": (0, 215, 255),
    "solid_line_crossing": (230, 120, 60),
}
OTHER_EVENT_COLOUR = (160, 160, 160)
LIGHT_COLOURS = {RED: (40, 40, 220), AMBER: (0, 190, 255), GREEN: (60, 200, 60)}
ALARM = 0.5  # the metric's alarm threshold (evaluate.py)


def events_under_way(events: Sequence[Sequence], t_sec: float) -> list[str]:
    """Labels of the events whose segment covers t_sec, in order of their start."""
    return [label for start, end, label, *_ in sorted(events) if start <= t_sec < end]


def draw_frame(
    image: np.ndarray,
    tracks: np.ndarray,
    scale: tuple[float, float],
    t_sec: float,
    events: Sequence[str],
    risk: float | None = None,
    light: int | None = None,
) -> np.ndarray:
    """A copy of image with the frame's tracked road users, events, light and risk drawn on it.

    tracks: rows of the perception result's tracks table for this frame (full-resolution
        pixels); scale: (x, y) image pixels per full-resolution pixel.
    events: the labels under way (events_under_way); light: the top arm's phase code, if read.
    """
    out = image.copy()
    unit = max(1.0, out.shape[1] / 1280)  # line widths and text grow with the image
    for row in tracks:
        name, colour = ROAD_USERS.get(int(row["class_id"]), (None, None))
        if name is None:
            continue
        x0, y0 = int(row["x0"] * scale[0]), int(row["y0"] * scale[1])
        x1, y1 = int(row["x1"] * scale[0]), int(row["y1"] * scale[1])
        cv2.rectangle(out, (x0, y0), (x1, y1), colour, max(1, round(2 * unit)))
        _label(out, f"{name} {int(row['track_id'])}", (x0, y0 - 3), colour, 0.4 * unit)

    # Banner: the time, then one chip per event under way.
    x, y = round(10 * unit), round(28 * unit)
    x = _chip(out, f"{t_sec:6.1f} s", (x, y), (40, 40, 40), 0.6 * unit)
    for label in events:
        x = _chip(out, label, (x, y), EVENT_COLOURS.get(label, OTHER_EVENT_COLOUR), 0.6 * unit)

    if light in LIGHT_COLOURS:  # top right: the light as the system read it
        centre = (out.shape[1] - round(24 * unit), round(22 * unit))
        cv2.circle(out, centre, round(12 * unit), LIGHT_COLOURS[light], -1)
        cv2.circle(out, centre, round(12 * unit), (255, 255, 255), max(1, round(unit)))

    if risk is not None:  # bottom left: Part B's risk
        width, height = round(200 * unit), round(14 * unit)
        x0, y0 = round(10 * unit), out.shape[0] - round(12 * unit) - height
        cv2.rectangle(out, (x0, y0), (x0 + width, y0 + height), (40, 40, 40), -1)
        colour = (40, 40, 220) if risk >= ALARM else (60, 200, 60)
        cv2.rectangle(out, (x0, y0), (x0 + round(width * min(risk, 1.0)), y0 + height), colour, -1)
        _label(out, f"accident risk {risk:.2f}", (x0 + width + round(8 * unit), y0 + height),
               (255, 255, 255), 0.5 * unit)
    return out


def _chip(image: np.ndarray, text: str, origin: tuple[int, int], colour, size: float) -> int:
    """Text on a filled box at origin (its bottom left); returns the x where the next one goes."""
    thickness = max(1, round(size * 2))
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, thickness)
    x, y = origin
    pad = max(3, round(h / 3))
    cv2.rectangle(image, (x, y - h - pad), (x + w + 2 * pad, y + base + pad // 2), colour, -1)
    cv2.putText(image, text, (x + pad, y), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255),
                thickness, cv2.LINE_AA)
    return x + w + 3 * pad


def _label(image: np.ndarray, text: str, origin: tuple[int, int], colour, size: float) -> None:
    """Small text with a dark outline, readable on any background."""
    thickness = max(1, round(size * 2))
    for ink, width in (((0, 0, 0), thickness + 2), (colour, thickness)):
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, size, ink, width, cv2.LINE_AA)


def by_frame(table: np.ndarray) -> dict[int, np.ndarray]:
    """A table with a "frame" column (tracks, detections) split by frame number, in any order."""
    if len(table) == 0:
        return {}
    ordered = table[np.argsort(table["frame"], kind="stable")]
    frames, starts = np.unique(ordered["frame"], return_index=True)
    return dict(zip(frames.tolist(), np.split(ordered, starts[1:]), strict=True))


class H264Writer:
    """Writes BGR frames of one size to an H.264 MP4 that browsers play, through ffmpeg."""

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is needed to write browser-playable video")
        width, height = size
        self.size = size
        self.process = subprocess.Popen(
            [ffmpeg, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{width}x{height}", "-r", f"{fps:.4f}", "-i", "-",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE,
        )

    def write(self, image: np.ndarray) -> None:
        if (image.shape[1], image.shape[0]) != self.size:
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        self.process.stdin.write(np.ascontiguousarray(image).tobytes())

    def close(self) -> None:
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError("ffmpeg failed to write the video")

    def __enter__(self) -> H264Writer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
