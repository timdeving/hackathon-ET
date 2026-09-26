"""Plane geometry shared by the scene map and the rules."""
from __future__ import annotations

import numpy as np

from src.scene.geometry import (
    apply_homography,
    crosses_polyline,
    direction_along,
    distance_to_polyline,
    side_of_line,
)

LINE = np.array([[0.0, 0.0], [10.0, 0.0]])  # a horizontal line from x = 0 to x = 10
BEND = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])  # right, then down


def test_points_on_opposite_sides_get_opposite_signs():
    sides = side_of_line(np.array([[5.0, 1.0], [5.0, -1.0], [5.0, 0.0]]), LINE[0], LINE[1])
    assert sides[0] * sides[1] < 0
    assert sides[2] == 0


def test_crossing_a_line():
    start = np.array([[5.0, -1.0], [5.0, -1.0], [15.0, -1.0], [5.0, -1.0]])
    end = np.array([[5.0, 1.0], [6.0, -0.5], [15.0, 1.0], [5.0, 0.0]])
    # across it; staying on one side; passing beyond its end; stopping exactly on it
    assert crosses_polyline(start, end, LINE).tolist() == [True, False, False, True]


def test_landing_on_a_line_and_leaving_it_counts_once():
    path = np.array([[5.0, -1.0], [5.0, 0.0], [5.0, 1.0]])
    assert crosses_polyline(path[:-1], path[1:], LINE).sum() == 1


def test_crossing_any_segment_of_a_polyline():
    start, end = np.array([[9.0, 5.0], [5.0, 5.0]]), np.array([[11.0, 5.0], [6.0, 5.0]])
    assert crosses_polyline(start, end, BEND).tolist() == [True, False]


def test_direction_and_distance_follow_the_nearest_segment():
    points = np.array([[5.0, 3.0], [13.0, 8.0]])
    np.testing.assert_allclose(direction_along(BEND, points), [[1, 0], [0, 1]])
    np.testing.assert_allclose(distance_to_polyline(points, BEND), [3, 3])


def test_homography_maps_points():
    scale_and_shift = np.array([[2.0, 0.0, 1.0], [0.0, 2.0, -1.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(apply_homography(scale_and_shift, np.array([[1.0, 1.0]])), [[3, 1]])
