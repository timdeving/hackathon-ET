"""Box arithmetic for detection and tracking, in numpy so it is testable without PyTorch."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PAD_VALUE = 114  # the grey YOLO models are trained with for letterbox padding


@dataclass(frozen=True)
class Detections:
    """Objects found in one image. Boxes are [x0, y0, x1, y1] in that image's pixels."""

    boxes: np.ndarray  # (N, 4) float32
    scores: np.ndarray  # (N,) float32
    class_ids: np.ndarray  # (N,) int64, COCO class ids

    def __len__(self) -> int:
        return len(self.scores)


@dataclass(frozen=True)
class Letterboxed:
    image: np.ndarray  # exactly the model's input size
    scale_x: float  # model-input pixels per original-image pixel, horizontally
    scale_y: float  # the same, vertically


def letterbox(image: np.ndarray, size: tuple[int, int]) -> Letterboxed:
    """Fit an image into the model's input `size` (height, width).

    The image keeps its aspect ratio and is only ever shrunk, never enlarged. The rest is padded
    with grey on the right and bottom, so boxes need rescaling afterwards but no shifting.
    """
    height, width = image.shape[:2]
    target_h, target_w = size
    scale = min(target_h / height, target_w / width, 1.0)
    if scale < 1.0:
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    padded = np.full((target_h, target_w, 3), PAD_VALUE, dtype=np.uint8)
    padded[: image.shape[0], : image.shape[1]] = image
    return Letterboxed(padded, image.shape[1] / width, image.shape[0] / height)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over union of every box in a (N, 4) with every box in b (M, 4): (N, M)."""
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.prod(np.clip(bottom_right - top_left, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / np.maximum(union, 1e-9)


def detections_from_rows(
    rows: np.ndarray,
    keep_classes: np.ndarray,
    conf: float,
    max_det: int,
    scale_x: float,
    scale_y: float,
) -> Detections:
    """Turn model output rows [x0, y0, x1, y1, score, COCO class] into Detections.

    Keeps rows of our classes scoring at least `conf`, best first, at most `max_det`, and maps
    boxes from model-input pixels back to the original image by undoing the letterbox scale.
    """
    rows = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
    classes = np.rint(rows[:, 5]).astype(np.int64)
    keep = (rows[:, 4] >= conf) & np.isin(classes, keep_classes)
    rows, classes = rows[keep], classes[keep]
    order = np.argsort(-rows[:, 4], kind="stable")[:max_det]
    scale = np.array([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)
    return Detections(rows[order, :4] / scale, rows[order, 4], classes[order])
