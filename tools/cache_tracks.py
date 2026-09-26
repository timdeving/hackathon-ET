"""Run perception on videos once and cache the results, so rules can be developed on laptops.

Runs exactly the submission's perception (decode, detect, track, and the traffic lights' lit-pixel
counts) on each video and saves cache/<video>/<settings version>/: perception.npz (the detections
and tracks tables), lights.npz (the light counts) and meta.json (video, settings, light windows,
weights hash, git commit, timing). Prints a timing table. Run on the GPU PC; copy cache/ to
laptops and load it with src.perception.cache.load_result.

    python -m tools.cache_tracks VIDEOS [--out cache] [--light-snapshots]

VIDEOS is a folder of .mp4 files or a single file. --light-snapshots also saves each light's
window as a JPEG once per second of video, in the cache folder's lights/, to check the phases
read from it by eye (tools/check_signals.py).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from src.config import CONFIG_DIR, WEIGHTS_DIR, load_params
from src.perception.cache import file_sha256, save_result, settings_version
from src.perception.detector import load_detector
from src.perception.pipeline import Perception
from src.scene.scene_map import SceneMap
from src.scene.signals import CELL_PARAMS, light_windows
from src.video.probe import probe_video
from src.video.reader import SampledFrame

SCENE_MAP_PATH = CONFIG_DIR / "scene_map.json"


class LightSnapshots:
    """Saves every light window of a frame as a JPEG, about once per second of video."""

    def __init__(self, folder: Path, every_sec: float = 1.0) -> None:
        self.folder, self.every_sec, self.next_sec = folder, every_sec, 0.0

    def __call__(self, frame: SampledFrame) -> None:
        if frame.t_sec < self.next_sec:
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        for name, crop in frame.rois.items():
            path = self.folder / f"{name}_{frame.index:06d}.jpg"
            cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.next_sec = (math.floor(frame.t_sec / self.every_sec) + 1) * self.every_sec


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("videos", type=Path, help="a folder of .mp4 files, or one .mp4")
    parser.add_argument("--out", type=Path, default=Path("cache"), help="cache folder")
    parser.add_argument("--light-snapshots", action="store_true",
                        help="save each light's window once per second, to check by eye")
    args = parser.parse_args()

    params = load_params()
    perception = Perception(load_detector(params), params)
    scene = SceneMap.load(SCENE_MAP_PATH) if SCENE_MAP_PATH.exists() else None
    base = {section: params[section] for section in ("video", "detector", "tracker")}
    base["weights_sha256"] = file_sha256(WEIGHTS_DIR / params["detector"]["weights"])
    base["signal"] = {key: params["signal"][key] for key in CELL_PARAMS}
    if args.videos.is_file():
        videos = [args.videos]
    else:
        videos = sorted(p for p in args.videos.iterdir() if p.suffix.lower() == ".mp4")

    print("| video | duration s | seconds | x duration | detections/frame | tracks | lights "
          "| saved to |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for video in videos:
        info = probe_video(video)
        rois = {}
        if scene is not None:
            rois = light_windows(scene, params["signal"]["margin_px"], (info.width, info.height))
        settings = {**base, "light_windows": rois}
        snapshots = None
        if args.light_snapshots and rois:
            snapshots = LightSnapshots(args.out / info.name / settings_version(settings) / "lights")
        result = perception.run(video, rois=rois, on_frame=snapshots)
        folder = save_result(result, args.out, settings)
        n_tracks = len(np.unique(result.tracks["track_id"]))
        print(
            f"| {result.info.name} | {result.info.duration:.1f} | {result.seconds:.1f} "
            f"| {result.seconds / result.info.duration:.2f} "
            f"| {len(result.detections) / max(1, result.n_analysed):.1f} | {n_tracks} "
            f"| {', '.join(sorted(result.lights)) or 'none'} | {folder} |"
        )


if __name__ == "__main__":
    main()
