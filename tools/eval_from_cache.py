"""Score the rules on cached tracks against our dev labels: seconds, on a laptop.

    python -m tools.eval_from_cache [--rules jaywalking ...] [--videos C3905.MP4 ...]
        [--set rules.jaywalking.min_sec=1.5 ...] [--timeline CLASS]
        [--labels data/dev_labels.json] [--cache cache] [--scene configs/scene_map.json]
        [--log docs/results_log.md | --no-log]

For every labelled video with a track cache (cache/<video>/<version>/, made on the GPU PC by
tools/cache_tracks.py; the newest version if there are several), it runs what the submission
runs after perception: camera alignment, per-object features, the rules and finalize_events().
Then it scores the result with the organizers' evaluate.py: F1 per class at tIoU 0.3, 0.5 and
0.7, and Score A. By default every rule runs, enabled or not, so a class can be measured before
the team enables it.

- --set overrides any parameter for this run (the value is read as YAML), so thresholds can be
  tried without editing configs/params.yaml. Unknown parameters are refused.
- --timeline CLASS lists one class's labelled and predicted events side by side, with their
  overlaps: that's where boundary errors show.
- Each run appends a line (commit, rules, overrides, scores) to the results log, where the
  before/after numbers of every change come from. On the GPU PC, pass
  --log docs/gpu/results_log.md: that machine writes only in docs/gpu/.

Camera alignment comes from cache/<video>/alignment.json, read by src.scene.alignment (the GPU
PC's). Until that exists, videos are treated as framed exactly like the reference picture.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

import evaluate
from src.config import CONFIG_DIR, REPO_ROOT, load_params
from src.events import Segment
from src.features.tracks import NoAlignment, PointMapper
from src.perception.cache import load_result
from src.postprocess.segments import finalize_events
from src.rules import RULES, find_events
from src.scene.scene_map import SceneMap

LOG_HEADER = """# Results log

One line per run of `tools/eval_from_cache.py`: rules on cached tracks, scored against the dev
labels (mean F1 over tIoU 0.3/0.5/0.7).

| When | Commit | Videos | Rules | Overrides | Score A | Mean F1 per class |
| --- | --- | --- | --- | --- | --- | --- |
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rules", nargs="+", default=sorted(RULES), help="default: every rule")
    parser.add_argument("--videos", nargs="+", help="default: every labelled video with a cache")
    parser.add_argument("--set", nargs="+", default=[], dest="overrides", metavar="KEY=VALUE")
    parser.add_argument("--timeline", metavar="CLASS")
    parser.add_argument("--labels", type=Path, default=REPO_ROOT / "data" / "dev_labels.json")
    parser.add_argument("--cache", type=Path, default=REPO_ROOT / "cache")
    parser.add_argument("--scene", type=Path, default=CONFIG_DIR / "scene_map.json")
    parser.add_argument("--log", type=Path, default=REPO_ROOT / "docs" / "results_log.md")
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()

    params = with_overrides(load_params(), args.overrides)
    if not args.scene.exists():
        print(f"no scene map at {args.scene}: draw it first (docs/SCENE_MAP.md)", file=sys.stderr)
        return 1
    scene = SceneMap.load(args.scene)
    labels = json.loads(args.labels.read_text())
    unknown = sorted(set(args.videos or []) - set(labels))
    if unknown:
        print(f"not in {args.labels}: {', '.join(unknown)}", file=sys.stderr)
        return 1

    predictions, raw = {}, {}
    for video in args.videos or sorted(labels):
        folder = cache_folder(args.cache, video)
        if folder is None:
            print(f"{video}: no track cache in {args.cache / video}; skipped")
            continue
        mapper, alignment = alignment_for(args.cache / video)
        segments = find_events(load_result(folder), mapper, scene, params, args.rules)
        events = finalize_events(segments, labels[video]["duration"], params["postprocess"])
        predictions[video], raw[video] = {"events": events}, segments
        print(f"{video}: cache {folder.name}, {alignment}, {len(events)} events")
    if not predictions:
        print("nothing to score: no labelled video has a track cache", file=sys.stderr)
        return 1

    report = evaluate.evaluate_part_a({video: labels[video] for video in predictions}, predictions)
    print_scores(report, args.rules)
    if args.timeline:
        print_timeline(args.timeline, labels, predictions, raw)
    if not args.no_log:
        append_log(args.log, report, sorted(predictions), args.rules, args.overrides)
    return 0


def with_overrides(params: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """A copy of params with each KEY=VALUE applied; KEY is a dotted path that must exist."""
    params = copy.deepcopy(params)
    for item in overrides:
        key, equals, value = item.partition("=")
        *path, last = key.split(".")
        section = params
        for part in path:
            section = section.get(part) if isinstance(section, dict) else None
        if not equals or not isinstance(section, dict) or last not in section:
            raise SystemExit(f"--set {item!r}: no parameter {key!r} in configs/params.yaml")
        section[last] = yaml.safe_load(value)
    return params


def cache_folder(cache: Path, video: str) -> Path | None:
    """The video's newest cache version: the folder of its most recently written meta.json."""
    metas = sorted((cache / video).glob("*/meta.json"), key=lambda meta: meta.stat().st_mtime)
    return metas[-1].parent if metas else None


def alignment_for(video_folder: Path) -> tuple[PointMapper, str]:
    """The video's camera alignment, and a note saying which one is used."""
    path = video_folder / "alignment.json"
    if not path.exists():
        return NoAlignment(), "no alignment"
    try:
        from src.scene.alignment import load_alignment  # the GPU PC's module, once merged
    except ImportError:
        return NoAlignment(), "alignment.json found but src/scene/alignment.py missing"
    return load_alignment(path), "aligned"


def print_scores(report: dict, rules: list[str]) -> None:
    thresholds = evaluate.TIOU_THRESHOLDS
    print(f"\n{'class':<22}" + "".join(f"{'F1@' + str(t):>9}" for t in thresholds)
          + f"{'mean':>9}{'TP/FP/FN@0.5':>15}")
    for label in report["classes"]:
        scores = report["per_class"][label]
        counts = scores["0.5"]
        name = label if label in rules else f"{label} (no rule)"
        print(f"{name:<22}" + "".join(f"{scores[str(t)]['f1']:>9.3f}" for t in thresholds)
              + f"{scores['f1_mean']:>9.3f}"
              + f"{'{tp}/{fp}/{fn}'.format(**counts):>15}")
    print(f"\nScore A (labelled and predicted classes, as the organizers score): "
          f"{report['score_a']:.4f}")
    covered = [label for label in report["classes"] if label in rules]
    if covered:
        mean = sum(report["per_class"][label]["f1_mean"] for label in covered) / len(covered)
        print(f"Mean over the classes these rules cover ({len(covered)}): {mean:.4f}")


def print_timeline(
    label: str, labels: dict, predictions: dict, raw: dict[str, list[Segment]]
) -> None:
    """Each labelled event with its best-overlapping prediction, then predictions that overlap
    no label, with the tracks behind them."""
    print(f"\nTimeline: {label}")
    for video in predictions:
        truth = [(s, e) for s, e, lab in labels[video]["events"] if lab == label]
        found = [(s, e) for s, e, lab in predictions[video]["events"] if lab == label]
        print(f"  {video}: {len(truth)} labelled, {len(found)} predicted")
        for event in sorted(truth):
            best = max(found, key=lambda f: evaluate.tiou(event, f), default=None)
            overlap = evaluate.tiou(event, best) if best else 0.0
            match = f"predicted {best[0]:.2f}-{best[1]:.2f}, tIoU {overlap:.2f}"
            print(f"    labelled  {event[0]:8.2f}-{event[1]:<8.2f} "
                  + (match if overlap > 0 else "nothing predicted there"))
        for event in sorted(found):
            if all(evaluate.tiou(event, t) == 0 for t in truth):
                tracks = sorted({track for seg in raw[video] if seg.label == label
                                 and seg.start < event[1] and seg.end > event[0]
                                 for track in seg.tracks})
                print(f"    predicted {event[0]:8.2f}-{event[1]:<8.2f} no label there "
                      f"(tracks {', '.join(map(str, tracks))})")


def append_log(
    path: Path, report: dict, videos: list[str], rules: list[str], overrides: list[str]
) -> None:
    if not path.exists():
        path.write_text(LOG_HEADER)
    per_class = ", ".join(f"{label} {report['per_class'][label]['f1_mean']:.3f}"
                          for label in report["classes"])
    cells = [
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        _commit(),
        ", ".join(videos),
        "all" if sorted(rules) == sorted(RULES) else ", ".join(rules),
        " ".join(f"`{item}`" for item in overrides) or "—",
        f"{report['score_a']:.4f}",
        per_class,
    ]
    with path.open("a", encoding="utf-8") as log:
        log.write("| " + " | ".join(cells) + " |\n")


def _commit() -> str:
    """The current commit, marked "-dirty" when tracked files have uncommitted changes."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=False).stdout.strip()

    dirty = git("status", "--porcelain", "--untracked-files=no")
    return git("rev-parse", "--short", "HEAD") + ("-dirty" if dirty else "")


if __name__ == "__main__":
    sys.exit(main())
