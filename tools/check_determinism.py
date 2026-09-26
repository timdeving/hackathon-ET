"""Do two runs give identical results? The task requires it. Run on the GPU PC.

    python -m tools.check_determinism perception VIDEO
        Runs perception twice in one process, cutting the traffic lights' windows as Part A
        does, and compares the detections and tracks tables and the lights' counts bit for bit.
    python -m tools.check_determinism predictions A.json B.json
        Compares two predictions files (events and risk curves), ignoring the harness's timing
        log, which always differs.

Exit code 0 = PASS.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def compare_predictions(a: dict, b: dict) -> list[str]:
    """Differences between the `videos` sections of two predictions files, one line each."""
    problems = []
    for name in sorted(set(a["videos"]) | set(b["videos"])):
        first, second = a["videos"].get(name), b["videos"].get(name)
        if first is None or second is None:
            problems.append(f"{name}: in only one of the files")
            continue
        if first["events"] != second["events"]:
            problems.append(f"{name}: events differ")
        if first.get("risk") != second.get("risk"):
            problems.append(f"{name}: risk curves differ")
    return problems


def compare_tables(first: np.ndarray, second: np.ndarray, name: str) -> list[str]:
    if len(first) != len(second):
        return [f"{name}: {len(first)} rows vs {len(second)}"]
    differing = np.flatnonzero(first != second)
    if len(differing) == 0:
        return []
    return [f"{name}: {len(differing)} rows differ, first at row {differing[0]}"]


def check_perception(video: Path) -> list[str]:
    # Imported here so the predictions check also works on laptops, without PyTorch.
    from src.config import CONFIG_DIR, load_params
    from src.perception.detector import load_detector
    from src.perception.pipeline import Perception
    from src.scene.scene_map import SceneMap
    from src.scene.signals import light_windows
    from src.video.probe import probe_video

    params = load_params()
    perception = Perception(load_detector(params), params)
    rois = {}
    if (CONFIG_DIR / "scene_map.json").exists():
        info = probe_video(video)
        scene = SceneMap.load(CONFIG_DIR / "scene_map.json")
        rois = light_windows(scene, params["signal"]["margin_px"], (info.width, info.height))
    first, second = perception.run(video, rois=rois), perception.run(video, rois=rois)
    print(f"{video.name}: {len(first.detections)} detections, {len(first.tracks)} track rows, "
          f"lights {', '.join(sorted(first.lights)) or 'none'}")
    problems = compare_tables(first.detections, second.detections, "detections")
    problems += compare_tables(first.tracks, second.tracks, "tracks")
    for name in sorted(set(first.lights) | set(second.lights)):
        if not np.array_equal(first.lights.get(name), second.lights.get(name)):
            problems.append(f"lights {name}: the counts differ")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("perception", help="run perception twice").add_argument("video", type=Path)
    predictions = modes.add_parser("predictions", help="compare two predictions files")
    predictions.add_argument("a", type=Path)
    predictions.add_argument("b", type=Path)
    args = parser.parse_args()

    if args.mode == "perception":
        problems = check_perception(args.video)
    else:
        first, second = (json.loads(path.read_text()) for path in (args.a, args.b))
        problems = compare_predictions(first, second)
    for problem in problems:
        print(f"  {problem}")
    print("PASS: identical" if not problems else "FAIL: the two runs differ")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
