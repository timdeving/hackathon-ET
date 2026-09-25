"""Run perception on videos once and cache the results, so rules can be developed on laptops.

Runs exactly the submission's perception (decode, detect, track) on each video and saves
cache/<video>/<settings version>/: perception.npz (the detections and tracks tables) and
meta.json (video, settings, weights hash, git commit, timing). Prints a timing table.
Run on the GPU PC; copy cache/ to laptops and load it with src.perception.cache.load_result.

    python -m tools.cache_tracks VIDEOS [--out cache]

VIDEOS is a folder of .mp4 files or a single file.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import WEIGHTS_DIR, load_params
from src.perception.cache import file_sha256, save_result
from src.perception.detector import load_detector
from src.perception.pipeline import Perception


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("videos", type=Path, help="a folder of .mp4 files, or one .mp4")
    parser.add_argument("--out", type=Path, default=Path("cache"), help="cache folder")
    args = parser.parse_args()

    params = load_params()
    perception = Perception(load_detector(params), params)
    settings = {section: params[section] for section in ("video", "detector", "tracker")}
    settings["weights_sha256"] = file_sha256(WEIGHTS_DIR / params["detector"]["weights"])
    if args.videos.is_file():
        videos = [args.videos]
    else:
        videos = sorted(p for p in args.videos.iterdir() if p.suffix.lower() == ".mp4")

    print("| video | duration s | seconds | x duration | detections/frame | tracks | saved to |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for video in videos:
        result = perception.run(video)
        folder = save_result(result, args.out, settings)
        n_tracks = len(np.unique(result.tracks["track_id"]))
        print(
            f"| {result.info.name} | {result.info.duration:.1f} | {result.seconds:.1f} "
            f"| {result.seconds / result.info.duration:.2f} "
            f"| {len(result.detections) / max(1, result.n_analysed):.1f} | {n_tracks} "
            f"| {folder} |"
        )


if __name__ == "__main__":
    main()
