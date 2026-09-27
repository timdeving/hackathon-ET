"""Part B's risk: time to collision between road users (src/risk/conflicts.py), and the causal
estimator the harness steps through a video (src/risk/estimator.py), with a stand-in detector."""
from __future__ import annotations

import numpy as np
import pytest

from src.config import load_params
from src.perception.boxes import Detections
from src.risk import estimator
from src.risk.conflicts import OTHER, VEHICLE, VULNERABLE, closest_approach, conflict_risk

RISK = load_params()["risk"]
HEIGHT = 100.0  # every box 100 px tall: distances and speeds below are in box heights x 100


def risk_of(points, velocities, kinds) -> float:
    n = len(points)
    return conflict_risk(np.array(points, float), np.array(velocities, float),
                         np.full(n, HEIGHT), np.array(kinds), RISK)


# time to collision --------------------------------------------------------------------------

def test_closest_approach_head_on_and_moving_apart():
    when, gap = closest_approach(np.array([[4.0, 0.0], [4.0, 0.0]]),
                                 np.array([[-2.0, 0.0], [2.0, 0.0]]))
    assert when.tolist() == [2.0, 0.0] and gap.tolist() == [0.0, 4.0]


def test_a_car_about_to_hit_a_pedestrian_raises_an_alarm():
    # 1 box height apart, closing at 4 box heights/s: they meet in 0.25 s.
    risk = risk_of([[0, 0], [100, 0]], [[400, 0], [0, 0]], [VEHICLE, VULNERABLE])
    assert risk > 0.8


def test_a_collision_two_seconds_away_is_only_a_low_score():
    risk = risk_of([[0, 0], [800, 0]], [[400, 0], [0, 0]], [VEHICLE, VEHICLE])  # 2 s
    assert RISK["base"] < risk < 0.1


def test_passing_wide_or_moving_apart_or_in_step_is_ordinary_traffic():
    base = RISK["base"]
    passing = risk_of([[0, 0], [300, 150]], [[400, 0], [0, 0]], [VEHICLE, VEHICLE])  # 1.5 apart
    parting = risk_of([[0, 0], [100, 0]], [[-400, 0], [0, 0]], [VEHICLE, VEHICLE])
    queue = risk_of([[0, 0], [100, 0]], [[100, 0], [100, 0]], [VEHICLE, VEHICLE])
    assert passing == parting == queue == base


def test_slow_approaches_are_ordinary_traffic():
    """A queue shuffling up, or someone walking into a standing car: no accident coming."""
    creeping = risk_of([[0, 0], [100, 0]], [[150, 0], [0, 0]], [VEHICLE, VEHICLE])  # 1.5 /s
    walking = risk_of([[0, 0], [100, 0]], [[0, 0], [-300, 0]], [VEHICLE, VULNERABLE])
    assert creeping == walking == RISK["base"]


def test_only_pairs_with_a_vehicle_count():
    walkers = risk_of([[0, 0], [100, 0]], [[400, 0], [0, 0]], [VULNERABLE, VULNERABLE])
    light = risk_of([[0, 0], [100, 0]], [[400, 0], [0, 0]], [VEHICLE, OTHER])
    assert walkers == light == RISK["base"]


# the estimator --------------------------------------------------------------------------------

class Collision:
    """Stands in for YOLO on a 1000 x 500 frame: a car driving at a standing pedestrian.

    In the working image (half size) the car, 75 x 50 px, moves 21.6 px per call, 10 calls per
    second: 4.3 of its box heights per second in the frame. `top` sets where both stand in the
    picture (5: up on the far road; 200: bottoms on the frame's edge)."""

    input_size = (250, 500)  # the working image: half the frame's size

    def __init__(self, top: float = 100) -> None:
        self.calls, self.top = 0, top

    def __call__(self, image: np.ndarray) -> Detections:
        x = 20 + 21.6 * self.calls
        self.calls += 1
        y0, y1 = self.top, self.top + 50
        boxes = np.array([[x, y0, x + 75, y1], [440, y0, 465, y1]], np.float32)
        return Detections(boxes, np.array([0.9, 0.9], np.float32), np.array([2, 0]))


def run(detector, n_frames=180, fps=30.0) -> list[float]:
    """Step an estimator through n_frames blank frames with this detector standing in."""
    estimator._detector = detector
    risk = estimator.RiskEstimator()
    risk.reset({"video_id": "clip", "fps": fps, "width": 1000, "height": 500,
                "n_frames": n_frames})
    frame = np.zeros((500, 1000, 3), np.uint8)
    return [risk.step(frame, index / fps) for index in range(n_frames)]


@pytest.fixture(autouse=True)
def no_detector_left_behind():
    yield
    estimator._detector = None


def test_a_car_driving_into_a_pedestrian_raises_the_risk_before_they_meet():
    scores = run(Collision())
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores[12] < 0.5  # 0.4 s in, the car still far off
    alarm = next(i for i, s in enumerate(scores) if s >= 0.5)
    meet = (440 - 95) / 21.6 * 3  # frame at which the car's front reaches the pedestrian
    assert 0 < meet - alarm < 3 * 30  # the alarm comes before they meet, within 3 s


def test_the_far_road_and_boxes_cut_off_by_the_frame_edge_take_no_part():
    assert max(run(Collision(top=5))) == RISK["base"]  # ground points in the top 30%
    assert max(run(Collision(top=200))) == RISK["base"]  # bottoms on the frame's edge


def test_the_frames_in_between_repeat_the_last_score():
    scores = run(Collision())
    stride = load_params()["risk"]["stride"]
    assert all(scores[i] == scores[i - i % stride] for i in range(len(scores)))


def test_two_estimators_on_the_same_frames_agree():
    assert run(Collision()) == run(Collision())


def test_a_failing_detector_keeps_the_last_score_and_never_raises():
    def broken(image):
        raise RuntimeError("boom")

    broken.input_size = (250, 500)
    assert set(run(broken)) == {RISK["base"]}


def test_without_a_detector_every_frame_scores_the_base():
    assert set(run(None)) == {RISK["base"]}


def test_part_b_stops_analysing_once_its_time_is_up(monkeypatch):
    params = load_params()
    monkeypatch.setitem(params["budget"], "part_b_extra_ratio", 0.0)
    detector = Collision()
    run(detector)
    assert detector.calls == 1  # the first analysed frame used up the (zero) budget


def test_the_working_image_is_the_detector_width():
    seen = []

    class Recorder(Collision):
        def __call__(self, image):
            seen.append(image.shape)
            return super().__call__(image)

    run(Recorder(), n_frames=3)
    assert seen[0] == (250, 500, 3)
