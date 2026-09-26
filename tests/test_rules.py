"""Batch A rules on synthetic tracks in a small street (tests/synthetic_tracks.py): each rule finds
its event with the right start and end, and ignores the look-alikes it must ignore."""
from __future__ import annotations

import numpy as np
import pytest

from src.features.tracks import NoAlignment
from src.rules import find_events
from src.rules.solid_line_crossing import _pair_corners
from src.scene.scene_map import SceneMap
from tests.synthetic_tracks import (
    CAR,
    MOTORCYCLE,
    PERSON,
    params_with,
    result_of,
    street_scene,
    times,
    track,
)

SCENE = SceneMap(street_scene())
PARAMS = params_with(rules={"jaywalking": {"edge_margin_px": 0}})
TOLERANCE = 0.15  # seconds: a little over one update


def events(label: str, *tracks: np.ndarray, params: dict = PARAMS) -> list[tuple[float, float]]:
    segments = find_events(result_of(*tracks), NoAlignment(), SCENE, params, [label])
    return sorted((round(s.start, 2), round(s.end, 2)) for s in segments)


def assert_one_event(found: list[tuple[float, float]], start: float, end: float) -> None:
    assert len(found) == 1, found
    assert found[0] == pytest.approx((start, end), abs=TOLERANCE)


# jaywalking ----------------------------------------------------------------------------------

def walker_across(x: float, track_id: int = 1) -> np.ndarray:
    """A pedestrian crossing the whole road at x, from the lower pavement (y = 550) to the upper
    one (y = 150), at 50 px/s: on the road (y 500 to 200) from t = 1 s to t = 7 s."""
    t = times(0, 8)
    return track(track_id, PERSON, t, x, 550 - 50 * t, width=20, height=60)


def test_a_pedestrian_crossing_the_road_away_from_the_zebra_is_jaywalking():
    assert_one_event(events("jaywalking", walker_across(900)), 1.0, 7.0)


def test_crossing_on_the_zebra_is_not_jaywalking():
    assert events("jaywalking", walker_across(620)) == []


def test_standing_on_the_median_is_not_jaywalking():
    t = times(0, 5)
    assert events("jaywalking", track(1, PERSON, t, 300, 350, width=20, height=60)) == []


def test_a_rider_is_not_a_pedestrian():
    t = times(0, 8)
    y = 550 - 50 * t
    motorcycle = track(1, MOTORCYCLE, t, 900, y + 10, width=40, height=40)
    rider = track(2, PERSON, t, 900, y, width=20, height=50)
    assert events("jaywalking", motorcycle, rider) == []


def test_the_edge_margin_ignores_feet_at_the_kerb():
    t = times(0, 5)
    at_kerb = track(1, PERSON, t, 900, 495 - t, width=20, height=60)  # 5-10 px onto the road
    assert events("jaywalking", at_kerb) != []
    margin = params_with(rules={"jaywalking": {"edge_margin_px": 20}})
    assert events("jaywalking", at_kerb, params=margin) == []


# failure_to_yield ----------------------------------------------------------------------------

def car_through_zebra(track_id: int = 1) -> np.ndarray:
    """A car 80 px long driving east at 100 px/s along lane west_in_2: its front reaches the
    zebra (x = 600) at t = 1.2 s and its rear leaves it (x = 640) at t = 2.4 s."""
    t = times(0, 4)
    return track(track_id, CAR, t, 440 + 100 * t, 480, width=80, height=50)


def pedestrian_on_zebra(track_id: int = 2) -> np.ndarray:
    t = times(0, 4)
    return track(track_id, PERSON, t, 620, 300, width=20, height=60)


def test_driving_through_the_zebra_while_someone_is_on_it_is_failure_to_yield():
    found = events("failure_to_yield", car_through_zebra(), pedestrian_on_zebra())
    assert_one_event(found, 1.2, 2.4)


def test_driving_through_an_empty_zebra_is_fine():
    assert events("failure_to_yield", car_through_zebra()) == []


def test_a_car_queued_on_the_zebra_does_not_drive_through_it():
    t = times(0, 4)
    queued = track(1, CAR, t, 620, 480, width=80, height=50)
    assert events("failure_to_yield", queued, pedestrian_on_zebra()) == []


# wrong_way -----------------------------------------------------------------------------------

def test_driving_against_the_lane_is_wrong_way():
    t = times(0, 4)
    against = track(1, CAR, t, 500 - 100 * t, 400, width=80, height=50)  # west, in west_in_1
    assert_one_event(events("wrong_way", against), 0.0, 4.0)


def test_driving_with_the_lane_is_fine():
    t = times(0, 4)
    assert events("wrong_way", track(1, CAR, t, 100 + 100 * t, 400, width=80, height=50)) == []


def test_reversing_slowly_is_not_wrong_way():
    t = times(0, 4)
    reversing = track(1, CAR, t, 300 - 10 * t, 400, width=80, height=50)  # 0.2 box heights/s
    assert events("wrong_way", reversing) == []


# solid_line_crossing -------------------------------------------------------------------------

def test_crossing_the_solid_line_runs_from_the_first_corner_to_the_second():
    # The line rises from y = 400 at x = 0 to y = 460 at x = 600, so at y = 435 it's at x = 350.
    # The car's right corner (x + 40) gets there at t = 1.7 s, its left corner (x - 40) at 2.5 s.
    t = times(0, 4)
    car = track(1, CAR, t, 140 + 100 * t, 435, width=80, height=50)
    assert_one_event(events("solid_line_crossing", car), 1.7, 2.5)


def test_staying_on_one_side_of_the_solid_line_is_fine():
    t = times(0, 4)
    assert events("solid_line_crossing", track(1, CAR, t, 100 + 100 * t, 480, 80, 50)) == []


def test_a_corner_that_crosses_and_comes_back_is_only_a_touch():
    touch = [(1.0, "left", 1.0), (1.5, "left", -1.0), (2.0, "right", 1.0)]
    assert _pair_corners(touch, max_sec=4.0, dt=0.1) == []
    assert _pair_corners([(1.0, "right", 1.0), (1.8, "left", 1.0)], 4.0, 0.1) == [(1.0, 1.8)]
    assert _pair_corners([(1.0, "right", 1.0), (6.0, "left", 1.0)], 4.0, 0.1) == []


# the framework -------------------------------------------------------------------------------

def test_only_the_requested_rules_run_and_unknown_ones_are_refused():
    walker = walker_across(900)
    assert find_events(result_of(walker), NoAlignment(), SCENE, PARAMS, []) == []
    segments = find_events(result_of(walker), NoAlignment(), SCENE, PARAMS, ["jaywalking"])
    assert [(s.label, s.tracks) for s in segments] == [("jaywalking", (1,))]
    with pytest.raises(ValueError, match="no rule"):
        find_events(result_of(walker), NoAlignment(), SCENE, PARAMS, ["fire_smoke"])


def test_nothing_is_predicted_until_the_team_enables_classes():
    assert PARAMS["rules"]["enabled"] == []
    assert find_events(result_of(walker_across(900)), NoAlignment(), SCENE, PARAMS) == []
