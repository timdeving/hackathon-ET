"""Check the traffic-light phases read from cached videos: against the traffic, and by eye.

    python -m tools.check_signals [--videos C3896.MP4 ...] [--cache cache] [--sheets DIR]

For every video whose cache holds light data (tools/cache_tracks.py; the newest such version;
camera alignment from cache/<video>/alignment.json), it prints:

- each light's phase timeline, and each arm's (what the rules read);
- how many vehicles cross the stop line in each phase, per lane. If the phases are right, nearly
  all cross on green (docs/SIGNAL_DESIGN.md §7); a lane that often crosses on red may have an
  arrow of its own (§9).

--sheets DIR writes one picture per light and video: the snapshots saved by
tools/cache_tracks.py --light-snapshots on either side of each phase change, labelled with the
phase read there, to check the changes by eye.
"""
from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.config import CONFIG_DIR, REPO_ROOT, load_params
from src.features.tracks import VEHICLES, NoAlignment, PointMapper, compute_features
from src.perception.cache import load_result
from src.perception.pipeline import PerceptionResult
from src.perception.stitching import stitch_tracks
from src.rules.common import AMBER, GREEN, RED, UNKNOWN, stop_line_crossing
from src.scene.alignment import load_alignment
from src.scene.scene_map import SceneMap
from src.scene.signals import PHASE_NAMES, Phases, combine_heads, read_heads

SHEET_TILE_WIDTH = 240  # pixels per snapshot on a contact sheet
SHEET_MAX_CHANGES = 24  # phase changes shown per light
COLOURS_BGR = {UNKNOWN: (160, 160, 160), RED: (60, 60, 255), AMBER: (0, 190, 255),
               GREEN: (80, 220, 80)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--videos", nargs="+", help="default: every video in the cache")
    parser.add_argument("--cache", type=Path, default=REPO_ROOT / "cache")
    parser.add_argument("--scene", type=Path, default=CONFIG_DIR / "scene_map.json")
    parser.add_argument("--sheets", type=Path, metavar="DIR", help="write contact sheets here")
    args = parser.parse_args()
    # signals.py logs each light's reading, and how a pedestrian light relates to the vehicle's
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    params, scene = load_params(), SceneMap.load(args.scene)
    videos = args.videos or sorted(p.name for p in args.cache.iterdir() if p.is_dir())
    checked = 0
    for video in videos:
        folder = _newest_with_lights(args.cache / video)
        if folder is None:
            print(f"{video}: no cache with light data (tools/cache_tracks.py); skipped")
            continue
        result = load_result(folder)
        mapper = _alignment(args.cache / video)
        heads = read_heads(result, mapper, scene, params)
        arms = combine_heads(heads, scene, result)
        print(f"\n{video}: cache {folder.name}, {result.n_analysed} analysed frames")
        for name, phases in heads.items():
            print(_timeline(f"light {name}", phases, result.info.fps))
        for arm, phases in arms.items():
            print(_timeline(f"arm {arm}", phases, result.info.fps))
        _print_crossings(result, mapper, scene, params, arms)
        if args.sheets:
            for name, phases in heads.items():
                path = _write_sheet(args.sheets, folder, name, phases, result, mapper, scene)
                print(f"contact sheet: {path}" if path else f"{name}: no snapshots in {folder}")
        checked += 1
    return 0 if checked else 1


def _newest_with_lights(video_folder: Path) -> Path | None:
    folders = sorted(
        (meta.parent for meta in video_folder.glob("*/meta.json")
         if (meta.parent / "lights.npz").exists()),
        key=lambda folder: (folder / "meta.json").stat().st_mtime,
    )
    return folders[-1] if folders else None


def _alignment(video_folder: Path) -> PointMapper:
    path = video_folder / "alignment.json"
    return load_alignment(path) if path.exists() else NoAlignment()


def _timeline(label: str, phases: Phases, fps: float) -> str:
    """'light x: read 93%: red 0.0-31.2, green 31.2-58.0, ...' wrapped to the terminal."""
    read = 100 * float(np.mean(phases.codes != UNKNOWN)) if len(phases.codes) else 0.0
    spans = ", ".join(f"{PHASE_NAMES[code]} {start:.1f}-{end:.1f}"
                      for start, end, code in phases.spans(fps))
    return textwrap.fill(f"{label}: read on {read:.0f}% of the frames: {spans}", width=100,
                         subsequent_indent="    ")


def _print_crossings(
    result: PerceptionResult, mapper: PointMapper, scene: SceneMap, params: dict, arms: dict
) -> None:
    """Vehicles crossing the stop line, per lane and phase, as Part A sees them (stitched)."""
    features = compute_features(stitch_tracks(result, params), mapper, scene, params["features"])
    counts: Counter = Counter()
    for _, rows in features.tracks_of(VEHICLES):
        crossing = stop_line_crossing(rows, scene)
        if crossing is None or scene.lanes[crossing[1]].arm not in arms:
            continue
        index, lane = crossing
        phase = int(arms[scene.lanes[lane].arm].at(rows["frame"][index : index + 1])[0])
        counts[scene.lanes[lane].name, phase] += 1
    lanes = sorted({lane for lane, _ in counts})
    order = [GREEN, AMBER, RED, UNKNOWN]
    print("stop-line crossings by phase:  lane      " + "".join(f"{PHASE_NAMES[c]:>9}"
                                                                for c in order))
    for lane in lanes:
        print(f"{'':31}{lane:<10}" + "".join(f"{counts[lane, c]:>9}" for c in order))
    if not lanes:
        print(f"{'':31}(none)")


def _write_sheet(
    out: Path,
    folder: Path,
    name: str,
    phases: Phases,
    result: PerceptionResult,
    mapper: PointMapper,
    scene: SceneMap,
) -> Path | None:
    """The snapshots on either side of each phase change, one change per row."""
    snapshots = (folder / "lights").glob(f"{name}_*.jpg")  # <light>_<frame>.jpg
    shots = sorted((int(path.stem.rsplit("_", 1)[1]), path) for path in snapshots)
    if not shots:
        return None
    frames = np.array([frame for frame, _ in shots])
    fps, rows = result.info.fps, []
    for start, _, _ in phases.spans(fps)[1 : SHEET_MAX_CHANGES + 1]:
        change = round(start * fps)
        before, after = np.searchsorted(frames, change) - 1, np.searchsorted(frames, change)
        tiles = [_tile(shots[i][1], shots[i][0], phases, fps, result, mapper, scene, name)
                 for i in (before, after) if 0 <= i < len(shots)]
        if len(tiles) == 2:
            rows.append(np.hstack(tiles))
    if not rows:
        return None
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{result.info.name}_{name}.jpg"
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return path


def _tile(path: Path, frame: int, phases: Phases, fps: float, result: PerceptionResult,
          mapper: PointMapper, scene: SceneMap, name: str) -> np.ndarray:
    """One snapshot with the head's box (mapped into the video) and the phase read there."""
    image = cv2.imread(str(path))
    x0, y0, x1, y1 = scene.lights[name].box
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    to_video = np.linalg.inv(getattr(mapper, "homography", np.eye(3)))
    in_video = cv2.perspectiveTransform(corners[None], to_video)[0]
    window = result.light_windows[name]
    outline = (in_video - [window[0], window[1]]).round().astype(np.int32)
    cv2.polylines(image, [outline], True, (255, 255, 0), 2)
    scale = SHEET_TILE_WIDTH / image.shape[1]
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    code = int(phases.at(np.array([frame]))[0])
    cv2.rectangle(image, (0, 0), (image.shape[1], 26), COLOURS_BGR[code], -1)
    cv2.putText(image, f"{frame / fps:.1f} s {PHASE_NAMES[code]}", (6, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return cv2.copyMakeBorder(image, 0, 4, 0, 4, cv2.BORDER_CONSTANT)


if __name__ == "__main__":
    sys.exit(main())
