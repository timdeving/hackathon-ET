"""illegal_turn on synthetic tracks: the lane a vehicle came from against the road it left by,
the exit zones that tell where it left, and the map pieces behind them."""
from __future__ import annotations

import numpy as np
import pytest

from src.features.tracks import NoAlignment
from src.rules import find_events
from src.scene.scene_map import SceneMap
from tests.synthetic_tracks import CAR, params_with, result_of, street_scene, times, track
from tests.test_rules_batch_b import u_turn

TOLERANCE = 0.25


def events(*tracks, exits: dict | None = None, seconds: float = 8.0) -> list[tuple[float, float]]:
    scene = SceneMap(street_scene(exits=exits, side_road=True))
    segments = find_events(result_of(*tracks, seconds=seconds), NoAlignment(), scene,
                           params_with(), ["illegal_turn"])
    return sorted((round(s.start, 2), round(s.end, 2)) for s in segments)


def assert_one_event(found, start: float, end: float) -> None:
    assert len(found) == 1, found
    assert found[0] == pytest.approx((start, end), abs=TOLERANCE)


def right_turn() -> np.ndarray:
    """A car driving east in lane west_in_2 (y = 465) at 100 px/s to x = 680 (t = 3.8 s), turning
    right on a quarter circle of radius 80 into the side road (x = 760), then driving south into
    its exit zone. Its heading leaves east by 20 deg at t = 4.08 s and is within 15 deg of south
    from t = 4.85 s."""
    t = times(0, 6)
    arc = 3.8 + (np.pi / 2) * 80 / 100  # when the quarter circle ends
    angle = -np.pi / 2 + (np.clip(t, 3.8, arc) - 3.8) * 100 / 80
    x = np.select([t < 3.8, t < arc], [300 + 100 * t, 680 + 80 * np.cos(angle)], 760)
    y = np.select([t < 3.8, t < arc], [465, 545 + 80 * np.sin(angle)], 545 + 100 * (t - arc))
    return track(1, CAR, t, x, y, width=80, height=50)


def straight_on() -> np.ndarray:
    """A car driving east in lane west_in_2 at 100 px/s: its front crosses the stop line (x = 600)
    at t = 2.6 s, and it reaches the east exit zone (x = 960) at t = 6.6 s."""
    t = times(0, 7)
    return track(1, CAR, t, 300 + 100 * t, 465, width=80, height=50)


def test_turning_right_from_a_straight_on_lane_is_illegal():
    found = events(right_turn(), exits={"west_in_2": ["east"]})
    assert_one_event(found, 4.08, 4.85)


def test_turning_right_where_allowed_is_fine():
    assert events(right_turn(), exits={"west_in_2": ["east", "south"]}) == []


def test_going_straight_from_a_turn_only_lane_is_illegal():
    found = events(straight_on(), exits={"west_in_2": ["south"]}, seconds=7.0)
    assert_one_event(found, 2.6, 6.6)


def test_unknown_exits_mean_no_call():
    assert events(right_turn()) == []


def test_u_turns_are_left_to_illegal_u_turn():
    assert events(u_turn(), exits={"west_in_1": ["east"]}) == []


def test_exit_zones_are_looked_up_by_arm():
    scene = SceneMap(street_scene(side_road=True))
    zones = scene.exit_index(np.array([[980, 300], [760, 590], [500, 300]]))
    assert [scene.exit_arms[i] if i >= 0 else None for i in zones] == ["east", "south", None]
    assert SceneMap(street_scene()).exit_arms == []  # maps without exit zones still load
