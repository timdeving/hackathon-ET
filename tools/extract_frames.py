"""Empty-road backgrounds for drawing the scene map.

Standalone: needs only OpenCV and numpy (pip install opencv-python-headless numpy), so it runs
anywhere the videos are. For each video it reads every frame once, keeps --samples frames
spread evenly across the video (full resolution), and saves in <out>/<video name>/:

  background.png          the per-pixel median of those frames: the road with moving traffic
                          removed, at full resolution. The scene map is drawn on this.
  background_preview.jpg  the same, 1920 px wide, for a quick look
  activity.jpg            where things moved (red = a lot, blue = never), 1920 px wide: shows
                          lanes, turning paths and crossings in use
  frame_start.jpg, frame_middle.jpg, frame_end.jpg   three of the sampled frames, full size

and <out>/all_backgrounds.jpg: every background preview side by side, for comparing them.

    python tools/extract_frames.py VIDEOS [--out scene_backgrounds] [--samples 80]

VIDEOS is a folder of .mp4 files or a single file.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

PREVIEW_WIDTH = 1920
ROWS_PER_STRIP = 216  # the median is computed strip by strip, so memory use stays small
JPEG = [cv2.IMWRITE_JPEG_QUALITY, 90]


def sample_frames(path: Path, samples: int) -> list[np.ndarray]:
    """`samples` frames spread evenly over the whole video, at full resolution.

    Reads every frame in order, since seeking is unreliable in these files, but converts only
    the ones it keeps, which is much faster than converting them all.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = set(np.linspace(0, total - 1, samples).round().astype(int).tolist())
    frames, index = [], 0
    while cap.grab():
        if index in wanted:
            ok, frame = cap.retrieve()
            if ok:
                frames.append(frame)
        index += 1
    cap.release()
    return frames


def median_image(frames: list[np.ndarray]) -> np.ndarray:
    """Per-pixel median of equally sized images."""
    median = np.empty_like(frames[0])
    for top in range(0, median.shape[0], ROWS_PER_STRIP):
        strip = np.stack([frame[top : top + ROWS_PER_STRIP] for frame in frames])
        median[top : top + ROWS_PER_STRIP] = np.median(strip, axis=0).astype(np.uint8)
    return median


def shrink(image: np.ndarray) -> np.ndarray:
    """The image at most PREVIEW_WIDTH pixels wide."""
    width = min(PREVIEW_WIDTH, image.shape[1])
    height = round(image.shape[0] * width / image.shape[1])
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def activity_image(frames: list[np.ndarray], background: np.ndarray) -> np.ndarray:
    """How far each pixel strays from the background on average, as a heat map over it."""
    base = shrink(background)
    total = np.zeros(base.shape[:2], np.float32)
    for frame in frames:
        total += cv2.absdiff(shrink(frame), base).max(axis=2)
    top = max(float(np.percentile(total, 99)), 1e-6)  # the busiest 1% saturate at red
    heat = np.clip(total / top * 255, 0, 255).astype(np.uint8)
    return cv2.addWeighted(base, 0.5, cv2.applyColorMap(heat, cv2.COLORMAP_JET), 0.5, 0)


def mosaic(previews: list[np.ndarray], names: list[str]) -> np.ndarray:
    """The previews two per row, each labelled with its video's name."""
    height, width = previews[0].shape[:2]
    tiles = []
    for preview, name in zip(previews, names, strict=True):
        tile = cv2.resize(preview, (width, height))
        cv2.putText(tile, name, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
        tiles.append(tile)
    if len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)]
    return np.vstack(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("videos", type=Path, help="a folder of .mp4 files, or one .mp4")
    parser.add_argument("--out", type=Path, default=Path("scene_backgrounds"))
    parser.add_argument("--samples", type=int, default=80, help="frames per video (default 80)")
    args = parser.parse_args()

    if args.videos.is_file():
        videos = [args.videos]
    else:
        videos = sorted(p for p in args.videos.iterdir() if p.suffix.lower() == ".mp4")
    previews, names = [], []
    for video in videos:
        start = time.time()
        frames = sample_frames(video, args.samples)
        background = median_image(frames)
        folder = args.out / video.name
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / "background.png"), background)
        cv2.imwrite(str(folder / "background_preview.jpg"), shrink(background), JPEG)
        cv2.imwrite(str(folder / "activity.jpg"), activity_image(frames, background), JPEG)
        middle = len(frames) // 2
        for name, frame in (("start", frames[0]), ("middle", frames[middle]), ("end", frames[-1])):
            cv2.imwrite(str(folder / f"frame_{name}.jpg"), frame, JPEG)
        previews.append(shrink(background))
        names.append(video.name)
        seconds = time.time() - start
        print(f"{video.name}: median of {len(frames)} frames, {seconds:.0f} s -> {folder}")
    if previews:
        cv2.imwrite(str(args.out / "all_backgrounds.jpg"), mosaic(previews, names), JPEG)
        print(f"comparison of all backgrounds -> {args.out / 'all_backgrounds.jpg'}")


if __name__ == "__main__":
    main()
