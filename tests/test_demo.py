"""The live demo's processing (demo/pipeline.py): which clips it takes, and Part B's replay of
Part A's detections."""
from __future__ import annotations

import numpy as np
import pytest

from demo.pipeline import ReplayDetector, check
from src.perception.pipeline import DETECTION_DTYPE
from src.video.probe import VideoInfo


def test_a_clip_within_the_limit_is_taken(tiny_video):
    assert check(tiny_video.path).n_frames == tiny_video.n_frames


def test_a_clip_over_the_limit_is_refused_with_a_reason(tiny_video):
    with pytest.raises(ValueError, match="up to 1 s"):
        check(tiny_video.path, max_seconds=0.9)  # the tiny clip lasts 2 s; 1 s of leeway
    assert check(tiny_video.path, max_seconds=None).n_frames == tiny_video.n_frames


def test_a_file_that_is_no_video_is_refused(tmp_path):
    fake = tmp_path / "notes.mp4"
    fake.write_text("not a video")
    with pytest.raises(ValueError):
        check(fake)


def test_the_replay_gives_part_b_part_as_detections_in_working_image_pixels():
    detections = np.zeros(3, dtype=DETECTION_DTYPE)
    detections["frame"] = [3, 0, 3]
    detections["x0"], detections["x1"] = [100, 200, 300], [400, 500, 600]
    detections["class_id"] = [2, 0, 0]
    info = VideoInfo(name="v.mp4", fps=30.0, n_frames=9, width=3840, height=2160)
    frame = {"now": 3}
    replay = ReplayDetector(detections, 1280, info, lambda: frame["now"])
    assert replay.input_size == (720, 1280)
    found = replay(np.zeros((720, 1280, 3), np.uint8))
    assert found.class_ids.tolist() == [2, 0]
    np.testing.assert_allclose(found.boxes[:, [0, 2]], [[100 / 3, 400 / 3], [300 / 3, 600 / 3]])
    frame["now"] = 6  # a frame Part A found nothing on
    assert len(replay(np.zeros((720, 1280, 3), np.uint8))) == 0


def test_the_replay_of_a_clip_where_nothing_was_found_gives_nothing():
    info = VideoInfo(name="v.mp4", fps=30.0, n_frames=9, width=3840, height=2160)
    replay = ReplayDetector(np.zeros(0, dtype=DETECTION_DTYPE), 1280, info, lambda: 0)
    assert len(replay(np.zeros((720, 1280, 3), np.uint8))) == 0
