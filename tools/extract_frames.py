"""Reference images for the scene map: a median background per video, plus sample frames.

For each video, decodes it once and keeps --samples frames spread evenly across it, at full
resolution. Their per-pixel median shows the empty road: moving vehicles and people vanish,
leaving lane markings, stop lines and crossings clear to draw on. Also saves the first, middle
and last of those frames, to look at. Run on the GPU PC.

    python -m tools.extract_frames VIDEOS [--out outputs/frames] [--samples 60]

Writes <out>/<video>/background.png (full resolution; draw the scene map on it) and
frame_start.jpg, frame_middle.jpg, frame_end.jpg.
"""
from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

import cv2
import numpy as np

from src.video.probe import probe_video
from src.video.reader import VideoReader

ROWS_PER_BLOCK = 216  # the median is computed in strips of rows, so memory stays small


def median_image(frames: list[np.ndarray]) -> np.ndarray:
    """Per-pixel median of equally sized images."""
    median = np.empty_like(frames[0])
    for top in range(0, median.shape[0], ROWS_PER_BLOCK):
        strip = np.stack([frame[top : top + ROWS_PER_BLOCK] for frame in frames])
        median[top : top + ROWS_PER_BLOCK] = np.median(strip, axis=0).astype(np.uint8)
    return median


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("videos", type=Path, help="a folder of .mp4 files, or one .mp4")
    parser.add_argument("--out", type=Path, default=Path("outputs/frames"))
    parser.add_argument("--samples", type=int, default=60, help="frames per video (default 60)")
    args = parser.parse_args()

    if args.videos.is_file():
        videos = [args.videos]
    else:
        videos = sorted(p for p in args.videos.iterdir() if p.suffix.lower() == ".mp4")
    for video in videos:
        info = probe_video(video)
        stride = max(1, info.n_frames // args.samples)
        frames = [f.image for f in islice(VideoReader(video, stride=stride), args.samples)]
        folder = args.out / info.name
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / "background.png"), median_image(frames))
        middle = len(frames) // 2
        for name, frame in (("start", frames[0]), ("middle", frames[middle]), ("end", frames[-1])):
            cv2.imwrite(str(folder / f"frame_{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{info.name}: median of {len(frames)} frames -> {folder}/background.png")


if __name__ == "__main__":
    main()
