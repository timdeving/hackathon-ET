"""The time budget: how long Part A may run so the harness still has time for Part B."""
from __future__ import annotations

import pytest

from src.budget import harness_read_seconds, part_a_seconds, part_b_seconds

PARAMS = {
    "time_factor": 3.0,
    "calibration_frames": 60,
    "part_b_safety": 1.2,
    "part_b_extra_ratio": 0.0,
    "margin_ratio": 0.05,
}


def test_part_a_gets_the_budget_minus_part_b_and_the_margin():
    # A 100 s video of 3000 frames that OpenCV reads at 20 ms per frame.
    part_b = part_b_seconds(100.0, 3000, 0.02, PARAMS)
    assert part_b == pytest.approx(72.0)  # 3000 x 0.02 s x 1.2
    assert part_a_seconds(100.0, part_b, PARAMS) == pytest.approx(213.0)  # 300 x 0.95 - 72


def test_part_b_work_beyond_decoding_is_counted():
    params = {**PARAMS, "part_b_extra_ratio": 0.5}
    assert part_b_seconds(100.0, 3000, 0.02, params) == pytest.approx(122.0)


def test_part_a_gets_nothing_when_part_b_alone_does_not_fit():
    part_b = part_b_seconds(100.0, 3000, 0.1, PARAMS)  # 360 s of decoding in a 300 s budget
    assert part_a_seconds(100.0, part_b, PARAMS) < 0


def test_reading_speed_is_measured_on_the_video(tiny_video):
    seconds = harness_read_seconds(tiny_video.path, frames=60)  # the clip has only 50 frames
    assert 0.0 < seconds < 0.1
