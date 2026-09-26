"""Camera alignment for the sample videos: one homography per video onto the reference view.

For every video in VIDEOS it matches the video's empty-road background (made by
tools/extract_frames.py) with the reference video's, and writes cache/<video>/alignment.json with
src.scene.alignment.save_alignment(), next to the perception caches. It saves two check pictures
per video to outputs/eda/camera_shift/: the reference's edges in red over the video's background
as recorded, and over the same background after alignment, where road markings should sit on the
red edges. It prints each video's change of view (largest corner shift, centre shift, rotation,
zoom), inliers and matching error, and writes them to outputs/eda/camera_shift/alignment.json for
the website.

    python -m tools.align_videos VIDEOS --backgrounds DIR [--reference C3896.MP4] [--out cache]

VIDEOS is a folder of .mp4 files; only their names and frame sizes are read. DIR has one folder per
video, named after the file (C3896.MP4) or its stem (C3896), holding its background.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from src.config import REPO_ROOT, load_params
from src.scene.alignment import Alignment, estimate_alignment, save_alignment, view_change
from src.video.probe import probe_video

CHECK_DIR = REPO_ROOT / "outputs" / "eda" / "camera_shift"
JPEG = [cv2.IMWRITE_JPEG_QUALITY, 90]


def find_background(folder: Path, video: Path) -> Path:
    """A video's background.png, in a folder named after the video file or its stem."""
    for name in (video.name, video.stem):
        path = folder / name / "background.png"
        if path.exists():
            return path
    raise SystemExit(f"no {video.name}/background.png or {video.stem}/background.png in {folder}")


def check_picture(
    background: np.ndarray,
    reference: np.ndarray,
    alignment: Alignment,
    background_size: tuple[int, int],
    reference_size: tuple[int, int],
) -> np.ndarray:
    """The reference's edges in red over the background, moved into the reference's view."""
    height, width = reference.shape[:2]
    picture_to_full = np.diag(
        [background_size[0] / background.shape[1], background_size[1] / background.shape[0], 1.0]
    )
    full_to_picture = np.diag([width / reference_size[0], height / reference_size[1], 1.0])
    warp = full_to_picture @ alignment.homography @ picture_to_full
    moved = cv2.warpPerspective(background, warp, (width, height))
    canvas = cv2.cvtColor(cv2.cvtColor(moved, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    canvas[cv2.Canny(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), 60, 150) > 0] = (0, 0, 255)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("videos", type=Path, help="a folder of .mp4 files")
    parser.add_argument("--backgrounds", type=Path, required=True,
                        help="a folder with one background.png per video")
    parser.add_argument("--reference", default="C3896.MP4",
                        help="the video the scene map is drawn on (default C3896.MP4)")
    parser.add_argument("--out", type=Path, default=Path("cache"), help="cache folder")
    args = parser.parse_args()

    params = load_params()["alignment"]
    videos = sorted(p for p in args.videos.iterdir() if p.suffix.lower() == ".mp4")
    reference_video = next((v for v in videos if v.name == args.reference), None)
    if reference_video is None:
        raise SystemExit(f"{args.reference} is not in {args.videos}")
    reference = cv2.imread(str(find_background(args.backgrounds, reference_video)))
    reference_info = probe_video(reference_video)
    reference_size = (reference_info.width, reference_info.height)
    CHECK_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}
    print("| video | ok | inliers | error px | largest corner shift px | centre dx, dy px "
          "| rotation deg | zoom |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for video in videos:
        info = probe_video(video)
        size = (info.width, info.height)
        background = cv2.imread(str(find_background(args.backgrounds, video)))
        alignment = estimate_alignment(background, reference, params, size, reference_size)
        change = {k: round(v, 3) for k, v in view_change(alignment.homography, size).items()}
        extra = {"video": video.name, "reference": args.reference, "view_change": change}
        save_alignment(alignment, args.out / video.name / "alignment.json", extra)
        for name, shown in (("recorded", Alignment.identity()), ("aligned", alignment)):
            picture = check_picture(background, reference, shown, size, reference_size)
            cv2.imwrite(str(CHECK_DIR / f"{video.stem}_{name}.jpg"), picture, JPEG)
        summary[video.name] = {
            "ok": alignment.ok,
            "inliers": alignment.inliers,
            "error_px": round(alignment.error_px, 2),
            **change,
        }
        print(
            f"| {video.name} | {'yes' if alignment.ok else 'NO'} | {alignment.inliers} "
            f"| {alignment.error_px:.2f} | {change['max_corner_shift_px']:.1f} "
            f"| {change['centre_dx_px']:+.1f}, {change['centre_dy_px']:+.1f} "
            f"| {change['rotation_deg']:+.2f} | {change['zoom']:.4f} |"
        )
    (CHECK_DIR / "alignment.json").write_text(json.dumps(summary, indent=1))
    print(f"\nalignment files: {args.out}/<video>/alignment.json; check pictures and summary: "
          f"{CHECK_DIR.relative_to(REPO_ROOT)}/")
    return 0 if all(entry["ok"] for entry in summary.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
