"""The PyTorch side of the detector, with a stand-in model. Needs PyTorch: runs on the GPU PC
and is skipped on laptops and CI."""
from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.perception.detector import Detector  # noqa: E402  (only importable with PyTorch)

INPUT_SIZE = (32, 64)  # (height, width) the stand-in model pretends to be exported with


class FixedOutput(torch.nn.Module):
    """Stands in for an exported YOLO model: returns the same output whatever the image."""

    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("output", output)
        self.unused = torch.nn.Parameter(torch.zeros(1))  # a parameter gives the model a dtype

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.output.unsqueeze(0).repeat(images.shape[0], 1, 1)


def stand_in_detector(path, output: torch.Tensor) -> Detector:
    """Save the stand-in like an Ultralytics export (with its metadata) and load it."""
    metadata = {"config.txt": json.dumps({"imgsz": list(INPUT_SIZE)})}
    torch.jit.script(FixedOutput(output)).save(str(path), _extra_files=metadata)
    return Detector(path, class_ids=[0, 2], conf=0.25, iou=0.7, max_det=300, device="cpu")


def test_nms_free_rows_are_filtered_and_mapped_back_to_the_image(tmp_path):
    rows = torch.tensor(
        [
            [4.0, 4.0, 12.0, 20.0, 0.9, 2.0],  # car
            [20.0, 2.0, 30.0, 30.0, 0.8, 0.0],  # person
            [0.0, 0.0, 5.0, 5.0, 0.1, 2.0],  # car below the confidence threshold
            [40.0, 0.0, 60.0, 30.0, 0.9, 56.0],  # chair: not one of our classes
        ]
    )
    detector = stand_in_detector(tmp_path / "nms_free.torchscript", rows)
    assert detector.end2end
    detections = detector(np.zeros((64, 128, 3), np.uint8))  # letterboxed at half size
    np.testing.assert_allclose(detections.boxes, [[8, 8, 24, 40], [40, 4, 60, 60]])
    assert detections.class_ids.tolist() == [2, 0]


def test_raw_output_goes_through_nms(tmp_path):
    raw = torch.zeros(4 + 80, 3)  # (4 box values + 80 class scores, 3 anchors)
    raw[:4, 0] = torch.tensor([10.0, 10.0, 8.0, 8.0])  # a car: centre x, centre y, w, h
    raw[:4, 1] = torch.tensor([10.5, 10.0, 8.0, 8.0])  # the same car, found twice
    raw[:4, 2] = torch.tensor([40.0, 16.0, 10.0, 20.0])  # a person
    raw[4 + 2, 0], raw[4 + 2, 1], raw[4 + 0, 2] = 0.9, 0.6, 0.7
    detector = stand_in_detector(tmp_path / "raw.torchscript", raw)
    assert not detector.end2end
    detections = detector(np.zeros((*INPUT_SIZE, 3), np.uint8))
    assert detections.class_ids.tolist() == [2, 0]
    np.testing.assert_allclose(detections.boxes, [[6, 6, 14, 14], [35, 6, 45, 26]])
    np.testing.assert_allclose(detections.scores, [0.9, 0.7], rtol=1e-6)
