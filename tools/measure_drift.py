"""Does the view drift within a video? Measures the camera's movement second by second.

Camera alignment uses one homography per video, which is only right if the camera holds still
while it records. For each video in VIDEOS this aligns one frame every --every seconds (at the
matching width) with the video's own empty-road background (made by tools/extract_frames.py), by
the same method as src/scene/alignment.py; moving traffic only adds outlier matches, which RANSAC
ignores. A steady camera stays within a pixel or so of its background all along; shake or slow
drift shows up as a series that wanders.

Writes outputs/eda/camera_shift/<video>_drift.json (per sample: time, centre shift, largest corner
shift, rotation, zoom, inliers) for the website and drift.png (the centre shift over time, one
panel per video), and prints the median and largest shift per video. Runs on the GPU PC in the
export environment, where Ultralytics brings matplotlib.

    python -m tools.measure_drift VIDEOS --backgrounds DIR [--every 1.0]

VIDEOS is a folder of .mp4 files or one file; DIR is as for tools/align_videos.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt  # noqa: E402  (the backend must be chosen first)
import numpy as np  # noqa: E402

from src.config import REPO_ROOT, load_params  # noqa: E402
from src.scene.alignment import estimate_alignment, view_change  # noqa: E402
from src.video.probe import probe_video  # noqa: E402
from src.video.reader import VideoReader  # noqa: E402
from tools.align_videos import find_background  # noqa: E402

OUT_DIR = REPO_ROOT / "outputs" / "eda" / "camera_shift"


def measure(video: Path, background: np.ndarray, params: dict, every: float) -> list[dict]:
    """The change of view from the background to one frame every `every` seconds."""
    info = probe_video(video)
    size = (info.width, info.height)
    stride = max(1, round(info.fps * every))
    samples = []
    for frame in VideoReader(video, stride=stride, width=params["match_width"]):
        alignment = estimate_alignment(frame.image, background, params, size, size)
        change = view_change(alignment.homography, size)
        samples.append({
            "t_sec": round(frame.t_sec, 3),
            "ok": alignment.ok,
            "inliers": alignment.inliers,
            **{key: round(value, 3) for key, value in change.items()},
        })
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("videos", type=Path, help="a folder of .mp4 files, or one .mp4")
    parser.add_argument("--backgrounds", type=Path, required=True,
                        help="a folder with one background.png per video")
    parser.add_argument("--every", type=float, default=1.0, help="seconds between samples")
    args = parser.parse_args()

    params = load_params()["alignment"]
    if args.videos.is_file():
        videos = [args.videos]
    else:
        videos = sorted(p for p in args.videos.iterdir() if p.suffix.lower() == ".mp4")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(len(videos), 1, figsize=(10, 2.6 * len(videos)), squeeze=False)

    print("| video | samples | not aligned | median shift px | largest shift px "
          "| largest rotation deg |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for video, axis in zip(videos, axes[:, 0], strict=True):
        background = cv2.imread(str(find_background(args.backgrounds, video)))
        samples = measure(video, background, params, args.every)
        record = {"video": video.name, "every_sec": args.every, "samples": samples}
        (OUT_DIR / f"{video.stem}_drift.json").write_text(json.dumps(record, indent=1))

        good = [s for s in samples if s["ok"]]
        shifts = np.array([s["max_corner_shift_px"] for s in good])
        rotation = max((abs(s["rotation_deg"]) for s in good), default=0.0)
        print(f"| {video.name} | {len(samples)} | {len(samples) - len(good)} "
              f"| {np.median(shifts):.2f} | {shifts.max():.2f} | {rotation:.3f} |")

        times = [s["t_sec"] for s in good]
        axis.plot(times, [s["centre_dx_px"] for s in good], label="centre dx")
        axis.plot(times, [s["centre_dy_px"] for s in good], label="centre dy")
        axis.set_title(f"{video.name}: camera movement against its own background")
        axis.set_ylabel("pixels (4K)")
        axis.legend(loc="upper right")
    axes[-1, 0].set_xlabel("seconds")
    figure.tight_layout()
    figure.savefig(OUT_DIR / "drift.png", dpi=100)
    print(f"\nper-sample data and drift.png: {OUT_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
