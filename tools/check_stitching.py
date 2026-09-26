"""Does track stitching join the right fragments? Counts before and after, and joins to look at.

For every perception cache under --cache it runs src.perception.stitching on the cached tracks
and prints the joins per class group, the track counts before and after, and how many tracks are
shorter than 2 s. With --videos it also saves outputs/stitching_check/<video>.jpg: for --samples
joins picked at random (fixed seed), the last box of the earlier fragment and the first box of the
later one, cropped from the video side by side. Each pair should show the same object.

    python -m tools.check_stitching [--cache cache] [--videos samples] [--samples 24]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.config import REPO_ROOT, load_params
from src.perception.cache import load_result
from src.perception.stitching import stitch_tracks

OUT_DIR = REPO_ROOT / "outputs" / "stitching_check"
TILE_HEIGHT = 180  # px of each crop in the check picture
PAIRS_PER_ROW = 3
GROUP_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "moto", 5: "bus", 7: "truck"}


def joins(table: np.ndarray) -> list[tuple[np.void, np.void]]:
    """(last row of the earlier fragment, first row of the later one) for every join.

    A joined object's fragments never overlap in time, so in frame order each change of the
    tracker's ID is one join.
    """
    table = np.sort(table, order=["track_id", "frame"])
    _, starts = np.unique(table["track_id"], return_index=True)
    pairs = []
    for rows in np.split(table, starts[1:]):
        for i in np.flatnonzero(np.diff(rows["tracker_id"]) != 0):
            pairs.append((rows[i], rows[i + 1]))
    return pairs


def short_share(table: np.ndarray, id_field: str, fps: float) -> float:
    """Share of tracks (counted by `id_field`) spanning less than 2 seconds."""
    table = np.sort(table, order=[id_field, "frame"])
    _, starts = np.unique(table[id_field], return_index=True)
    ends = np.append(starts[1:], len(table)) - 1
    spans = (table["frame"][ends] - table["frame"][starts]) / fps
    return float(np.mean(spans < 2.0)) if len(spans) else 0.0


def crop(frame: np.ndarray, row: np.void) -> np.ndarray:
    """The row's detection box with a margin of one box height, drawn in, TILE_HEIGHT tall."""
    x0, y0, x1, y1 = (float(row[k]) for k in ("det_x0", "det_y0", "det_x1", "det_y1"))
    margin = max(y1 - y0, 20.0)
    h, w = frame.shape[:2]
    left, top = int(max(0, x0 - margin)), int(max(0, y0 - margin))
    right, bottom = int(min(w, x1 + margin)), int(min(h, y1 + margin))
    tile = frame[top:bottom, left:right].copy()
    cv2.rectangle(tile, (int(x0) - left, int(y0) - top), (int(x1) - left, int(y1) - top),
                  (0, 255, 0), max(2, int(margin) // 30))
    scale = TILE_HEIGHT / tile.shape[0]
    return cv2.resize(tile, (max(1, round(tile.shape[1] * scale)), TILE_HEIGHT))


def frames_at(video: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    """The wanted frames, full size, numbered the way the harness numbers them."""
    cap = cv2.VideoCapture(str(video))
    found, index = {}, 0
    while index <= max(wanted) and cap.grab():
        if index in wanted:
            found[index] = cap.retrieve()[1]
        index += 1
    cap.release()
    return found


def check_picture(video: Path, pairs: list, fps: float) -> np.ndarray:
    frames = frames_at(video, {int(r["frame"]) for pair in pairs for r in pair})
    rows, current = [], []
    for earlier, later in pairs:
        gap = (int(later["frame"]) - int(earlier["frame"])) / fps
        before = crop(frames[int(earlier["frame"])], earlier)
        after = crop(frames[int(later["frame"])], later)
        pair = np.hstack([before, np.full((TILE_HEIGHT, 6, 3), 255, np.uint8), after])
        label = f"{GROUP_NAMES.get(int(earlier['class_id']), '?')} gap {gap:.1f}s"
        cv2.putText(pair, label, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        current.append(pair)
        if len(current) == PAIRS_PER_ROW:
            rows.append(current)
            current = []
    if current:
        rows.append(current)
    width = max(sum(p.shape[1] for p in row) + 20 * (len(row) - 1) for row in rows)
    lines = []
    for row in rows:
        line = np.zeros((TILE_HEIGHT + 20, width, 3), np.uint8)
        x = 0
        for pair in row:
            line[:TILE_HEIGHT, x : x + pair.shape[1]] = pair
            x += pair.shape[1] + 20
        lines.append(line)
    return np.vstack(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache", type=Path, default=Path("cache"), help="cache folder")
    parser.add_argument("--videos", type=Path, help="folder of the videos, for check pictures")
    parser.add_argument("--samples", type=int, default=24, help="joins per check picture")
    args = parser.parse_args()

    params = load_params()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("| video | tracks before | after | joins by kind | shorter than 2 s: before, after |")
    print("| --- | ---: | ---: | --- | --- |")
    for npz in sorted(args.cache.glob("*/*/perception.npz")):
        result = stitch_tracks(load_result(npz.parent), params)
        table, fps = result.tracks, result.info.fps
        pairs = joins(table)
        kinds = np.unique([GROUP_NAMES.get(int(a["class_id"]), "other") for a, _ in pairs],
                          return_counts=True)
        by_kind = ", ".join(f"{k} {n}" for k, n in zip(*kinds, strict=True))
        print(f"| {result.info.name} | {len(np.unique(table['tracker_id']))} "
              f"| {len(np.unique(table['track_id']))} | {by_kind} "
              f"| {short_share(table, 'tracker_id', fps):.0%}, "
              f"{short_share(table, 'track_id', fps):.0%} |")
        if args.videos and pairs:
            rng = np.random.default_rng(0)
            picked = [pairs[i] for i in sorted(rng.choice(len(pairs), min(args.samples, len(pairs)),
                                                          replace=False))]
            picture = check_picture(args.videos / result.info.name, picked, fps)
            cv2.imwrite(str(OUT_DIR / f"{Path(result.info.name).stem}.jpg"), picture,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
    if args.videos:
        print(f"\ncheck pictures: {OUT_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
