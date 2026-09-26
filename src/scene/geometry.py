"""Plane geometry the scene map and the rules share: sides of lines, crossings, directions.

Points are (N, 2) arrays of x, y pixels; lines and polylines are (K, 2) arrays of vertices.
Every function is vectorised over the points, because the rules ask these questions for every
track position in a video at once.
"""
from __future__ import annotations

import numpy as np


def _cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """z-component of the 2-D cross product u x v, over the last axis."""
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def side_of_line(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Which side of the line through a and b each point is on: positive on one side, negative
    on the other, zero on the line."""
    return _cross(b - a, points - a)


def crosses_polyline(start: np.ndarray, end: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """For each movement start[i] -> end[i], whether it crosses the polyline.

    A point exactly on a segment counts as being on its positive side, so a movement that ends on
    the line and the next one that leaves it are not both counted as crossings.
    """
    hits = np.zeros(len(start), dtype=bool)
    for a, b in zip(polyline[:-1], polyline[1:], strict=True):
        changes_side = (side_of_line(start, a, b) >= 0) != (side_of_line(end, a, b) >= 0)
        # ...and the crossing lies within this segment: a and b are on opposite sides of the move
        within = _cross(end - start, a - start) * _cross(end - start, b - start) <= 0
        hits |= changes_side & within
    return hits


def _nearest_on_polyline(points: np.ndarray, polyline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each point, the index of the nearest polyline segment and the distance to it."""
    a, b = polyline[:-1], polyline[1:]  # (S, 2)
    ab = b - a
    lengths = np.maximum((ab**2).sum(axis=1), 1e-12)
    t = np.clip(((points[:, None, :] - a) * ab).sum(axis=2) / lengths, 0.0, 1.0)  # (N, S)
    nearest = a + t[..., None] * ab
    distances = np.linalg.norm(points[:, None, :] - nearest, axis=2)
    segment = distances.argmin(axis=1)
    return segment, distances[np.arange(len(points)), segment]


def distance_to_polyline(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest point of the polyline."""
    return _nearest_on_polyline(points, polyline)[1]


def direction_along(polyline: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Unit direction of the polyline at the segment nearest to each point: (N, 2)."""
    segment, _ = _nearest_on_polyline(points, polyline)
    steps = polyline[1:] - polyline[:-1]
    directions = steps / np.maximum(np.linalg.norm(steps, axis=1, keepdims=True), 1e-12)
    return directions[segment]


def apply_homography(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map points through a 3x3 homography (e.g. picture pixels -> metres on the ground)."""
    mapped = np.column_stack([points, np.ones(len(points))]) @ np.asarray(homography).T
    return mapped[:, :2] / mapped[:, 2:3]
