"""Annotated versions of whole videos, for the website: tracked road users, the events under
way, the light as read and the accident risk, drawn on every analysed frame (10 per second).

    python -m tools.render_samples samples [--out outputs/website/media]

Runs the live demo's processing (demo/pipeline.py) without its length limit; on the GPU server
the four samples take about 10 minutes. Each video gets <name>.mp4 (H.264, 1280 x 720) and
<name>.json (its events and risk curve, as the demo returns them).
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from demo.pipeline import process
from src import part_a


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("videos", type=Path, help="a folder of .mp4 files, or one .mp4")
    parser.add_argument("--out", type=Path, default=Path("outputs/website/media"))
    args = parser.parse_args()
    videos = [args.videos] if args.videos.is_file() else sorted(
        p for p in args.videos.iterdir() if p.suffix.lower() == ".mp4")
    args.out.mkdir(parents=True, exist_ok=True)
    part_a.load_models()
    for video in videos:
        with tempfile.TemporaryDirectory() as tmp:
            result = process(video, tmp, max_seconds=None)
            shutil.move(result.video, args.out / f"{video.stem}.mp4")
        (args.out / f"{video.stem}.json").write_text(json.dumps(
            {"video": video.name, "duration": result.info.duration, "events": result.events,
             "risk": result.risk}))
        print(f"{video.name}: {len(result.events)} events, {result.seconds:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
