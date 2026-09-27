"""Drawing results onto frames, and writing browser-playable video."""
from __future__ import annotations

import shutil

import cv2
import numpy as np
import pytest

from src.perception.pipeline import TRACK_DTYPE
from src.rules.common import RED
from src.visualize import H264Writer, draw_frame, events_under_way, tracks_by_frame


def a_car_and_a_person() -> np.ndarray:
    tracks = np.zeros(3, dtype=TRACK_DTYPE)
    tracks["frame"] = [3, 0, 3]
    tracks["track_id"] = [1, 1, 2]
    tracks["class_id"] = [2, 2, 0]
    tracks["x0"], tracks["y0"], tracks["x1"], tracks["y1"] = [400, 400, 1000], 200, 800, 600
    return tracks


def test_the_events_under_way_are_those_covering_the_moment():
    events = [[5.0, 9.0, "jaywalking"], [0.0, 4.0, "red_light"], [2.0, 6.0, "stop_line"]]
    assert events_under_way(events, 5.0) == ["stop_line", "jaywalking"]
    assert events_under_way(events, 9.0) == []  # the end is outside the segment


def test_tracks_are_split_by_frame():
    by_frame = tracks_by_frame(a_car_and_a_person())
    assert sorted(by_frame) == [0, 3]
    assert by_frame[3]["track_id"].tolist() == [1, 2]


def test_drawing_leaves_the_original_alone_and_marks_the_objects():
    image = np.zeros((240, 320, 3), np.uint8)
    tracks = tracks_by_frame(a_car_and_a_person())[3]
    drawn = draw_frame(image, tracks, (0.25, 0.25), 1.5, ["jaywalking"], risk=0.7, light=RED)
    assert image.sum() == 0 and drawn.shape == image.shape
    assert drawn[50:150, 100:200].any()  # the car's box, at a quarter of its full-res position
    assert drawn[-30:, :100].any()  # the risk bar


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_the_writer_makes_a_video_with_every_frame(tmp_path):
    path = tmp_path / "out.mp4"
    with H264Writer(path, 10.0, (64, 48)) as writer:
        for i in range(5):
            writer.write(np.full((96, 128, 3), 40 * i, np.uint8))  # resized to 64 x 48
    capture = cv2.VideoCapture(str(path))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    size = capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    assert size == (64, 48)
