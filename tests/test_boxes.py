"""Detector box arithmetic: numpy only, so these run on laptops."""
from __future__ import annotations

import numpy as np
import pytest

from src.perception.boxes import PAD_VALUE, box_iou, detections_from_rows, letterbox


def test_letterbox_pads_an_image_that_fits_without_resizing_it():
    image = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    boxed = letterbox(image, (736, 1280))
    assert boxed.image.shape == (736, 1280, 3)
    np.testing.assert_array_equal(boxed.image[:720], image)
    assert (boxed.image[720:] == PAD_VALUE).all()
    assert (boxed.scale_x, boxed.scale_y) == (1.0, 1.0)


def test_letterbox_shrinks_a_larger_image_and_keeps_its_aspect_ratio():
    boxed = letterbox(np.zeros((1080, 1920, 3), np.uint8), (736, 1280))
    assert boxed.scale_x == pytest.approx(2 / 3)
    assert boxed.scale_y == pytest.approx(2 / 3)
    assert (boxed.image[:720] == 0).all()  # 1080 rows shrink to 720
    assert (boxed.image[720:] == PAD_VALUE).all()


def test_letterbox_never_enlarges():
    boxed = letterbox(np.zeros((360, 640, 3), np.uint8), (736, 1280))
    assert (boxed.scale_x, boxed.scale_y) == (1.0, 1.0)
    assert (boxed.image[:360, :640] == 0).all()
    assert (boxed.image[360:] == PAD_VALUE).all()


def test_box_iou():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    np.testing.assert_allclose(box_iou(a, b), [[1.0, 1 / 3, 0.0]])


def test_rows_become_detections_in_original_image_pixels():
    rows = np.array(
        [
            [10, 10, 50, 50, 0.9, 2],  # car: kept
            [0, 0, 20, 20, 0.05, 2],  # car below the confidence threshold: dropped
            [30, 30, 60, 60, 0.8, 56],  # chair: not one of our classes
            [100, 20, 140, 90, 0.95, 0],  # person: kept, and first (best score)
        ]
    )
    detections = detections_from_rows(
        rows, keep_classes=np.array([0, 2]), conf=0.1, max_det=300, scale_x=0.5, scale_y=0.5
    )
    np.testing.assert_allclose(detections.boxes, [[200, 40, 280, 180], [20, 20, 100, 100]])
    np.testing.assert_allclose(detections.scores, [0.95, 0.9], rtol=1e-6)
    assert detections.class_ids.tolist() == [0, 2]


def test_rows_are_capped_at_max_det_keeping_the_best():
    rows = np.array([[0, 0, 1, 1, score, 0] for score in (0.3, 0.9, 0.5)])
    detections = detections_from_rows(rows, np.array([0]), 0.1, max_det=2, scale_x=1, scale_y=1)
    np.testing.assert_allclose(detections.scores, [0.9, 0.5], rtol=1e-6)


def test_no_rows_give_no_detections():
    detections = detections_from_rows(np.zeros((0, 6)), np.array([0]), 0.1, 300, 1.0, 1.0)
    assert len(detections) == 0
    assert detections.boxes.shape == (0, 4)
