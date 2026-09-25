"""Perception pipeline end to end on the tiny clip, with a stand-in detector (no PyTorch)."""
from __future__ import annotations

import copy
import time

import numpy as np
import pytest

from src.config import load_params
from src.perception.boxes import Detections
from src.perception.pipeline import Perception

CAR = 2
WORKING_WIDTH = 32  # half the tiny clip's 64 px, so boxes must be mapped back to full size


class SquareDetector:
    """Stands in for YOLO on the tiny clip: reports its white square as a car.

    blip_call: on this call (0-based) it also reports a one-frame false detection.
    """

    input_size = (24, WORKING_WIDTH)

    def __init__(self, blip_call: int | None = None) -> None:
        self.blip_call = blip_call
        self.calls = 0

    def __call__(self, image: np.ndarray) -> Detections:
        rows = []
        ys, xs = np.nonzero(image.min(axis=2) >= 240)  # the square is white, the rest greyer
        if len(xs):
            rows.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, 0.9, CAR])
        if self.calls == self.blip_call:
            rows.append([0, 0, 4, 4, 0.9, CAR])
        self.calls += 1
        array = np.array(rows, dtype=np.float32).reshape(-1, 6)
        return Detections(array[:, :4], array[:, 4], array[:, 5].astype(np.int64))


def params_with(**video) -> dict:
    params = copy.deepcopy(load_params())
    params["video"].update(video)
    return params


def square_at(frame: int) -> list[int]:
    """Where the tiny clip draws its square, in full-resolution pixels (see conftest.py)."""
    x = frame % (64 - 8)
    return [x, 20, x + 8, 28]


def test_one_object_becomes_one_track_in_full_resolution_pixels(tiny_video):
    perception = Perception(SquareDetector(), params_with(stride=1, width=WORKING_WIDTH))
    result = perception.run(tiny_video.path)

    assert result.info.name == tiny_video.path.name
    assert result.n_analysed == tiny_video.n_frames
    assert result.complete
    assert result.detections["frame"].tolist() == list(range(tiny_video.n_frames))
    tracks = result.tracks
    assert set(tracks["track_id"].tolist()) == {1}
    assert tracks["frame"].tolist() == list(range(tiny_video.n_frames))
    for row in tracks:
        detected = [row["det_x0"], row["det_y0"], row["det_x1"], row["det_y1"]]
        np.testing.assert_allclose(detected, square_at(row["frame"]), atol=2.5)


def test_stride_skips_frames(tiny_video):
    perception = Perception(SquareDetector(), params_with(stride=3, width=WORKING_WIDTH))
    result = perception.run(tiny_video.path)
    assert result.detections["frame"].tolist() == list(range(0, tiny_video.n_frames, 3))


def test_tracks_that_are_never_confirmed_are_dropped(tiny_video):
    detector = SquareDetector(blip_call=10)
    result = Perception(detector, params_with(stride=1, width=WORKING_WIDTH)).run(tiny_video.path)
    assert len(result.detections) == tiny_video.n_frames + 1  # the blip was detected...
    assert set(result.tracks["track_id"].tolist()) == {1}  # ...but never became a track


def test_perception_stops_at_the_deadline(tiny_video):
    perception = Perception(SquareDetector(), params_with(stride=1, width=WORKING_WIDTH))
    result = perception.run(tiny_video.path, deadline=time.perf_counter() - 1.0)  # already past
    assert result.n_analysed == 1
    assert not result.complete
    assert result.detections["frame"].tolist() == [0]


def test_detector_and_working_width_must_agree():
    with pytest.raises(ValueError, match="must match"):
        Perception(SquareDetector(), params_with(width=64))
