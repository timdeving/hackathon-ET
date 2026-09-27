"""Ready-made results for the live demo: one-minute clips of the sample videos, run once through
the demo's processing, so that visitors see results without waiting.

    python -m demo.make_examples samples [--out demo/examples]

Run it on the GPU server (the clips take about a minute each there, far longer on the demo's
CPU), then upload the folder to the media dataset the app reads them from:

    hf upload akmaloio/wiut-traffic-media demo/examples demo_examples --repo-type dataset
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from demo.pipeline import process
from src import part_a

# (folder, sample video, start second, title shown in the demo)
EXAMPLES = [
    ("C3896_morning", "C3896.MP4", 75, "C3896, morning, 1:15-2:15: all five event types"),
    ("C3902_afternoon", "C3902.MP4", 90,
     "C3902, afternoon, 1:30-2:30: a red light run, and a stop past the line"),
    ("C3905_evening", "C3905.MP4", 0, "C3905, evening, 0:00-1:00: pedestrians at dusk"),
    ("C3897_close_call", "C3897.MP4", 210,
     "C3897, 3:30-4:30: a close call, where the accident risk passes 0.5"),
]
SECONDS = 60


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("videos", type=Path, help="the folder holding the sample videos")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "examples")
    args = parser.parse_args()
    part_a.load_models()
    for folder, video, start, title in EXAMPLES:
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / f"{folder}.mp4"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(start),
                            "-i", str(args.videos / video), "-t", str(SECONDS), "-c", "copy",
                            str(clip)], check=True)
            result = process(clip, Path(tmp) / "work")
            out = args.out / folder
            out.mkdir(parents=True, exist_ok=True)
            shutil.move(result.video, out / "annotated.mp4")
            shutil.move(result.predictions, out / "predictions.json")
            meta = {"title": title, "video": video, "start_sec": start,
                    "duration": round(result.info.duration, 2), "events": len(result.events),
                    "seconds": round(result.seconds, 1)}
            (out / "meta.json").write_text(json.dumps(meta, indent=1))
            print(f"{folder}: {len(result.events)} events, {result.seconds:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
