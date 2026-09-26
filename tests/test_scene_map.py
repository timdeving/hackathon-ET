"""The scene map loader and the questions the rules ask it."""
from __future__ import annotations

import numpy as np
import pytest

from src.scene.scene_map import SceneMap


def small_scene() -> dict:
    """A 200x100 picture: a road between y = 20 and 80, split by a median at y = 48-52. Below
    the median, an incoming lane moves right towards a junction at x = 90-110, behind a
    crossing at x = 80-88; above it, an outgoing lane moves left."""
    return {
        "format": 1,
        "reference": {"video": "tiny.mp4", "width": 200, "height": 100},
        "roads": {"main": [[0, 20], [200, 20], [200, 80], [0, 80]]},
        "islands": {"median": [[0, 48], [200, 48], [200, 52], [0, 52]]},
        "crossings": {"cw": [[80, 20], [88, 20], [88, 80], [80, 80]]},
        "junction": [[90, 20], [110, 20], [110, 80], [90, 80]],
        "lanes": {
            "west_in_1": {
                "arm": "west", "role": "in", "signal": True, "exits": ["west"],
                "polygon": [[0, 52], [80, 52], [80, 80], [0, 80]], "flow": [[5, 66], [75, 66]],
            },
            "west_out_1": {
                "arm": "west", "role": "out", "signal": False, "exits": None,
                "polygon": [[0, 20], [80, 20], [80, 48], [0, 48]], "flow": [[75, 34], [5, 34]],
            },
        },
        "stop_lines": {"west": [[80, 52], [80, 80]]},
        "solid_lines": {"centre": [[0, 50], [80, 50]]},
        "lights": {"west_main": {"box": [150, 5, 156, 18], "controls": ["west"],
                                 "layout": "vertical"}},
        "parking": {},
        "bus_stops": {"stop1": [[150, 70], [180, 70], [180, 80], [150, 80]]},
        "no_uturn": {},
        "ground_points": {},
        "image_to_ground": None,
    }


def test_the_road_excludes_islands_and_everything_outside():
    scene = SceneMap(small_scene())
    points = np.array([[10, 30], [10, 50], [10, 10], [-5, 30], [500, 30]])
    assert scene.on_road(points).tolist() == [True, False, False, False, False]


def test_points_are_placed_in_lanes_crossings_junction_and_bus_stops():
    scene = SceneMap(small_scene())
    names = [lane.name for lane in scene.lanes]
    lanes = scene.lane_index(np.array([[10, 66], [10, 34], [100, 50]]))
    assert [names[i] if i >= 0 else None for i in lanes] == ["west_in_1", "west_out_1", None]
    assert scene.crossing_index(np.array([[84, 30], [70, 30]])).tolist() == [0, -1]
    assert scene.in_junction(np.array([[100, 30], [70, 30]])).tolist() == [True, False]
    assert scene.at_bus_stop(np.array([[160, 75], [160, 30]])).tolist() == [True, False]


def test_flow_direction_follows_each_lane():
    scene = SceneMap(small_scene())
    points = np.array([[10, 66], [10, 34], [100, 50]])
    directions = scene.flow_direction(scene.lane_index(points), points)
    np.testing.assert_allclose(directions, [[1, 0], [-1, 0], [0, 0]])


def test_lane_and_light_attributes_are_loaded():
    scene = SceneMap(small_scene())
    lane = scene.lanes[0]
    assert (lane.arm, lane.role, lane.signal, lane.exits) == ("west", "in", True, ("west",))
    assert scene.lanes[1].exits is None  # unknown: rules make no call
    assert scene.lights["west_main"].controls == ("west",)
    assert set(scene.stop_lines) == {"west"}


def test_an_unknown_format_is_rejected():
    with pytest.raises(ValueError, match="format"):
        SceneMap({**small_scene(), "format": 2})
