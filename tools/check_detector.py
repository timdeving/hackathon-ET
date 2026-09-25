"""Do our detections match Ultralytics' own? Also saves annotated frames to look at.

For --frames frames spread across VIDEO, runs our Detector on the exported TorchScript file and
Ultralytics' predict() on the original .pt model with the same settings. It pairs boxes of the
same class by overlap and reports how many of Ultralytics' confident boxes (score >= 0.25) we
reproduce with IoU >= 0.9. Small differences near the confidence threshold are expected.
Annotated frames (ours in green, Ultralytics' in red) go to outputs/detector_check/: check that
distant cars and pedestrians get boxes.

Runs on the GPU PC in the export environment, because it needs Ultralytics.

    python -m tools.check_detector VIDEO --weights weights/X.torchscript --pt weights/X.pt
"""
from __future__ import annotations

import argparse
import sys
from itertools import islice

import cv2
import numpy as np
from ultralytics import YOLO

from src.config import REPO_ROOT, load_params
from src.perception.boxes import Detections, box_iou
from src.perception.detector import Detector
from src.video.probe import probe_video
from src.video.reader import VideoReader

CONFIDENT = 0.25  # Ultralytics' default predict() threshold: the boxes that clearly matter
MATCH_IOU = 0.9
PASS_RATE = 0.95
OUT_DIR = REPO_ROOT / "outputs" / "detector_check"


def best_overlaps(theirs: Detections, ours: Detections) -> list[float]:
    """For each of their confident boxes, the best IoU with one of our boxes of the same class."""
    overlaps = []
    confident = theirs.scores >= CONFIDENT
    for box, class_id in zip(theirs.boxes[confident], theirs.class_ids[confident], strict=True):
        same_class = ours.boxes[ours.class_ids == class_id]
        overlaps.append(float(box_iou(box[None], same_class).max()) if len(same_class) else 0.0)
    return overlaps


def draw(image: np.ndarray, theirs: Detections, ours: Detections) -> np.ndarray:
    canvas = image.copy()
    for boxes, colour, thickness in ((theirs.boxes, (0, 0, 255), 3), (ours.boxes, (0, 255, 0), 1)):
        for x0, y0, x1, y1 in boxes.astype(int):
            cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, thickness)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("video")
    parser.add_argument("--weights", required=True, help="exported TorchScript file")
    parser.add_argument("--pt", required=True, help="the original Ultralytics .pt file")
    parser.add_argument("--frames", type=int, default=10, help="frames to compare (default 10)")
    args = parser.parse_args()

    params = load_params()["detector"]
    class_ids = sorted(params["classes"])
    ours = Detector(args.weights, class_ids, params["conf"], params["iou"], params["max_det"])
    theirs = YOLO(args.pt)
    height, width = ours.input_size
    info = probe_video(args.video)
    reader = VideoReader(args.video, stride=max(1, info.n_frames // args.frames), width=width)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    overlaps, n_ours, n_theirs = [], 0, 0
    for frame in islice(reader, args.frames):
        mine = ours(frame.image)
        result = theirs.predict(
            frame.image,
            imgsz=[height, width],
            conf=params["conf"],
            iou=params["iou"],
            classes=class_ids,
            max_det=params["max_det"],
            half=True,
            device=0,
            verbose=False,
        )[0].boxes
        reference = Detections(
            result.xyxy.cpu().numpy(),
            result.conf.cpu().numpy(),
            result.cls.cpu().numpy().astype(np.int64),
        )
        overlaps += best_overlaps(reference, mine)
        n_ours, n_theirs = n_ours + len(mine), n_theirs + len(reference)
        out = OUT_DIR / f"{info.name}_{frame.index:06d}.jpg"
        cv2.imwrite(str(out), draw(frame.image, reference, mine))

    matched = np.mean([o >= MATCH_IOU for o in overlaps]) if overlaps else 0.0
    print(f"{info.name}: {args.frames} frames, {n_ours} boxes ours, {n_theirs} Ultralytics'")
    print(f"  Ultralytics' confident boxes (score >= {CONFIDENT}): {len(overlaps)}")
    print(f"  reproduced by ours with IoU >= {MATCH_IOU}: {matched:.1%}"
          f" (median IoU {np.median(overlaps) if overlaps else 0:.3f})")
    print(f"  annotated frames: {OUT_DIR.relative_to(REPO_ROOT)}/")
    passed = matched >= PASS_RATE
    print("PASS" if passed else f"FAIL: under {PASS_RATE:.0%} of confident boxes reproduced")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
