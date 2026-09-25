"""How fast is the detector on this GPU? Run it on the GPU PC.

Decodes --frames working images from VIDEO first (so decoding isn't timed), then runs every
--weights file over them: letterbox, network and post-processing, in milliseconds per frame.
"x duration" is the detector's share of the time budget at the current video.stride
(processing seconds per second of video).

    python -m tools.bench_detector VIDEO --weights weights/A.torchscript [weights/B.torchscript ...]
"""
from __future__ import annotations

import argparse
import time
from itertools import islice
from pathlib import Path

from src.config import load_params
from src.perception.detector import Detector
from src.video.probe import probe_video
from src.video.reader import VideoReader

WARM_UP = 5  # frames run before timing starts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("video")
    parser.add_argument("--weights", nargs="+", required=True, help="TorchScript files to time")
    parser.add_argument("--frames", type=int, default=100, help="frames per file (default 100)")
    args = parser.parse_args()

    params = load_params()
    detector_params, stride = params["detector"], params["video"]["stride"]
    video_fps = probe_video(args.video).fps
    frames_by_width: dict[int, list] = {}  # decoded once per input width

    print(f"| weights | input (h x w) | ms/frame | frames/s | x duration at stride {stride} "
          "| detections/frame |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for weights in args.weights:
        detector = Detector(
            weights,
            sorted(detector_params["classes"]),
            detector_params["conf"],
            detector_params["iou"],
            detector_params["max_det"],
        )
        height, width = detector.input_size
        if width not in frames_by_width:
            reader = VideoReader(args.video, stride=stride, width=width)
            frames_by_width[width] = [f.image for f in islice(reader, args.frames + WARM_UP)]
        images = frames_by_width[width]
        for image in images[:WARM_UP]:
            detector(image)

        start = time.perf_counter()
        n_detections = sum(len(detector(image)) for image in images[WARM_UP:])
        elapsed = time.perf_counter() - start
        n = len(images) - WARM_UP
        per_second = n / elapsed
        share = (video_fps / stride) / per_second
        print(f"| {Path(weights).name} | {height} x {width} | {1000 / per_second:.1f} "
              f"| {per_second:.1f} | {share:.2f} | {n_detections / n:.1f} |")


if __name__ == "__main__":
    main()
