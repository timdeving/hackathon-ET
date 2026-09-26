"""Turn our label files into data/dev_labels.json, in the organizers' ground-truth format.

    python -m tools.labels_to_ground_truth LABELS.json [LABELS.json ...] --videos samples
        [--out data/dev_labels.json]

Label files may be in the ground-truth format ({video: {"events": [...]}}) or the predictions
format ({"videos": {video: {"events": [...]}}}). Video names are matched to the files in
--videos ignoring case, since the scorer compares names exactly; each video's duration and fps
are read from its file the way the harness reads them. Same-class segments that overlap are
merged, as the organizers' annotations do. Every problem is listed at once (exit code 1).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.events import CLASSES
from src.video.probe import probe_video


def events_by_video(labels: dict) -> dict[str, list]:
    videos = labels["videos"] if "videos" in labels else labels
    return {name: entry["events"] for name, entry in videos.items()}


def merge_overlaps(events: list[list]) -> tuple[list[list], int]:
    """Same-class segments that overlap become one; returns the events and how many merged."""
    merged: list[list] = []
    n_merged = 0
    for start, end, label in sorted(events, key=lambda e: (e[2], e[0], e[1])):
        last = next((m for m in reversed(merged) if m[2] == label), None)
        if last is not None and start < last[1]:
            last[1] = max(last[1], end)
            n_merged += 1
        else:
            merged.append([start, end, label])
    return sorted(merged, key=lambda e: (e[0], e[1], e[2])), n_merged


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("labels", type=Path, nargs="+", help="label files")
    parser.add_argument("--videos", type=Path, required=True, help="folder with the videos")
    parser.add_argument("--out", type=Path, default=Path("data/dev_labels.json"))
    args = parser.parse_args()

    files = {p.name.lower(): p for p in args.videos.iterdir() if p.suffix.lower() == ".mp4"}
    ground_truth: dict[str, dict] = {}
    problems: list[str] = []
    for label_file in args.labels:
        for name, events in events_by_video(json.loads(label_file.read_text())).items():
            video = files.get(name.lower())
            if video is None:
                problems.append(f"{label_file.name}: no video called {name!r} in {args.videos}")
                continue
            if video.name in ground_truth:
                problems.append(f"{video.name} is labelled in more than one file")
                continue
            info = probe_video(video)
            for start, end, label in events:
                if label not in CLASSES:
                    problems.append(f"{video.name}: {label!r} is not an official class")
                if not 0 <= start < end <= info.duration + 0.5:
                    problems.append(f"{video.name}: bad times [{start}, {end}] for {label}")
            clipped = [[s, min(e, info.duration), label] for s, e, label in events]
            merged, n_merged = merge_overlaps(clipped)
            if n_merged:
                print(f"{video.name}: merged {n_merged} overlapping same-class segment(s)")
            ground_truth[video.name] = {
                "duration": round(info.duration, 4),
                "fps": info.fps,
                "events": merged,
            }
    if problems:
        for problem in problems:
            print(f"  {problem}")
        print(f"{len(problems)} problem(s): nothing written")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(ground_truth, indent=1))
    for name, entry in ground_truth.items():
        print(f"{name}: {len(entry['events'])} events, {entry['duration']} s")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
