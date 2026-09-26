"""Batch B rules on synthetic tracks in the small street (tests/synthetic_tracks.py): stopped
vehicles, congestion, stopping past the stop line, and U-turns."""
from __future__ import annotations

import numpy as np
import pytest

from src.features.tracks import NoAlignment
from src.rules import find_events
from src.scene.scene_map import SceneMap
from tests.synthetic_tracks import CAR, params_with, result_of, street_scene, times, track

TOLERANCE = 0.25  # seconds: smoothing blurs starts and stops by about a tenth of a second


def events(label, *tracks, scene=None, params=None, seconds=20.0) -> list[tuple[float, float]]:
    scene = SceneMap(scene or street_scene())
    segments = find_events(result_of(*tracks, seconds=seconds), NoAlignment(), scene,
                           params or params_with(), [label])
    return sorted((round(s.start, 2), round(s.end, 2)) for s in segments)


def assert_one_event(found, start: float, end: float) -> None:
    assert len(found) == 1, found
    assert found[0] == pytest.approx((start, end), abs=TOLERANCE)


def drive_stop_drive(stop_x: float, y: float, stop_from: float, stop_to: float, seconds: float,
                     track_id: int = 1, speed: float = 100.0) -> np.ndarray:
    """A car driving east at `speed` px/s that stands at stop_x from stop_from to stop_to."""
    t = times(0, seconds)
    x = np.select(
        [t < stop_from, t < stop_to],
        [stop_x - speed * (stop_from - t), stop_x],
        stop_x + speed * (t - stop_to),
    )
    return track(track_id, CAR, t, x, y, width=80, height=50)


def standing(x: float, y: float, seconds: float, track_id: int) -> np.ndarray:
    return track(track_id, CAR, times(0, seconds), x, y, width=80, height=50)


# stopped_vehicle -----------------------------------------------------------------------------

def test_a_car_standing_in_its_lane_for_15_s_is_a_stopped_vehicle():
    car = drive_stop_drive(300, 400, stop_from=2, stop_to=17, seconds=20)
    assert_one_event(events("stopped_vehicle", car), 2.0, 17.0)


def test_a_short_stop_is_not_a_stopped_vehicle():
    car = drive_stop_drive(300, 400, stop_from=2, stop_to=7, seconds=10)
    assert events("stopped_vehicle", car) == []


def test_a_queue_at_the_stop_line_is_not_a_stopped_vehicle():
    first = drive_stop_drive(560, 400, stop_from=2, stop_to=17, seconds=20, track_id=1)
    second = drive_stop_drive(460, 400, stop_from=3, stop_to=17.5, seconds=20, track_id=2)
    assert events("stopped_vehicle", first, second) == []


def test_stopping_at_the_bus_stop_is_not_a_stopped_vehicle():
    car = drive_stop_drive(900, 480, stop_from=2, stop_to=17, seconds=20)
    assert events("stopped_vehicle", car) == []


# congestion ----------------------------------------------------------------------------------

def queue(lanes_y: list[float], crawl_until: float, seconds: float) -> list[np.ndarray]:
    """Three cars per lane, crawling at 5 px/s (0.1 box heights/s) until crawl_until, then
    driving off at 100 px/s."""
    t = times(0, seconds)
    moved = np.where(t < crawl_until, 5 * t, 5 * crawl_until + 100 * (t - crawl_until))
    cars = []
    for lane, y in enumerate(lanes_y):
        for k in range(3):
            cars.append(track(10 * lane + k, CAR, t, 100 + 100 * k + 50 * lane + moved, y, 80, 50))
    return cars


SHORT_JAMS = params_with(rules={"congestion": {"min_sec": 5.0}})


def test_a_standstill_across_all_incoming_lanes_is_congestion():
    found = events("congestion", *queue([400, 470], crawl_until=12, seconds=15), params=SHORT_JAMS)
    assert_one_event(found, 0.0, 12.0)


def test_a_standstill_in_one_lane_only_is_not_congestion():
    found = events("congestion", *queue([400], crawl_until=12, seconds=15), params=SHORT_JAMS)
    assert found == []


def test_moving_traffic_is_not_congestion():
    found = events("congestion", *queue([400, 470], crawl_until=0, seconds=15), params=SHORT_JAMS)
    assert found == []


# stop_line -----------------------------------------------------------------------------------

def test_stopping_past_the_line_while_the_queue_waits_is_stop_line():
    # The car's centre stops at x = 580, so its front (x + 40 = 620) is past the stop line at
    # x = 600, on the zebra but not in the junction; another car waits in the next lane.
    car = drive_stop_drive(580, 400, stop_from=3, stop_to=8, seconds=10, track_id=1)
    waiting = standing(500, 470, seconds=10, track_id=2)
    assert_one_event(events("stop_line", car, waiting), 3.0, 8.0)


def test_stopping_before_the_line_is_fine():
    car = drive_stop_drive(550, 400, stop_from=3, stop_to=8, seconds=10, track_id=1)  # front 590
    waiting = standing(500, 470, seconds=10, track_id=2)
    assert events("stop_line", car, waiting) == []


def test_without_anyone_waiting_there_is_no_call():
    car = drive_stop_drive(580, 400, stop_from=3, stop_to=8, seconds=10)
    assert events("stop_line", car) == []


# illegal_u_turn ------------------------------------------------------------------------------

def u_turn(seconds: float = 8.0) -> np.ndarray:
    """A car driving east in lane west_in_1 (y = 400) at 100 px/s to x = 650 (t = 3.5 s), turning
    back on a half circle of radius 45 into lane west_out_1 (y = 310), then driving west.
    Its heading leaves east by 20 deg at t = 3.66 s and has turned 150 deg at t = 4.68 s."""
    t = times(0, seconds)
    arc = 3.5 + np.pi * 45 / 100  # when the half circle ends
    angle = np.pi / 2 - (np.clip(t, 3.5, arc) - 3.5) * 100 / 45
    x = np.select([t < 3.5, t < arc], [300 + 100 * t, 650 + 45 * np.cos(angle)],
                  650 - 100 * (t - arc))
    y = np.select([t < 3.5, t < arc], [400, 355 + 45 * np.sin(angle)], 310)
    return track(1, CAR, t, x, y, width=80, height=50)


def test_a_u_turn_from_a_lane_that_may_not_turn_back_is_illegal():
    scene = street_scene(exits={"west_in_1": ["east"]})
    assert_one_event(events("illegal_u_turn", u_turn(), scene=scene), 3.66, 4.68)


def test_a_u_turn_where_it_is_allowed_is_fine():
    scene = street_scene(exits={"west_in_1": ["east", "west"]})
    assert events("illegal_u_turn", u_turn(), scene=scene) == []


def test_unknown_exits_mean_no_call():
    assert events("illegal_u_turn", u_turn()) == []


def test_driving_straight_on_is_not_a_u_turn():
    scene = street_scene(exits={"west_in_1": ["east"]})
    t = times(0, 8)
    straight = track(1, CAR, t, 100 + 100 * t, 400, width=80, height=50)
    assert events("illegal_u_turn", straight, scene=scene) == []
