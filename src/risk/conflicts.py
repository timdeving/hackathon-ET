"""Time to collision between road users, from their recent motion: the core of Part B's risk.

Positions are ground points (the bottom centres of the boxes) in the video's own pixels.
Distances and speeds are measured in box heights, as the rules' features are: for a pair, the
mean of their two box heights, which perspective scales like distances on the road around them.
If both keep their velocity, the pair is closest at time t*, at distance d*: a collision course
when d* is under contact_heights, imminent when t* is short. Numpy only.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

VEHICLE, VULNERABLE, OTHER = 0, 1, 2  # what kind of road user a track is, for pairing


def closest_approach(offset: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For relative positions (N, 2) and relative velocities (N, 2): when the pairs are closest
    if both keep their velocity (0 if they're already moving apart), and how close."""
    speed2 = (velocity * velocity).sum(axis=1)
    toward = -(offset * velocity).sum(axis=1)
    when = np.where(speed2 > 1e-12, toward / np.maximum(speed2, 1e-12), 0.0)
    when = np.maximum(when, 0.0)
    return when, np.linalg.norm(offset + velocity * when[:, None], axis=1)


def conflict_risk(
    points: np.ndarray,
    velocities: np.ndarray,
    heights: np.ndarray,
    kinds: np.ndarray,
    params: Mapping[str, Any],
) -> float:
    """The riskiest pair's score in [base, 1].

    points, velocities: (N, 2), pixels and pixels per second; heights: (N,) box heights in
    pixels; kinds: (N,) VEHICLE, VULNERABLE or OTHER. Only pairs with a vehicle in them count
    (vehicle with vehicle, pedestrian or cyclist), and only while that vehicle moves (the faster
    one, for two): at least min_vehicle_speed box heights/s, since someone walking into a
    standing car, or a queue shuffling, isn't an accident about to happen. A pair on a collision
    course, closing at min_closing or faster, scores by its time to closest approach, 0.5 at
    ttc_alarm_sec (an alarm), rising towards 1 as it nears, and lower the wider it would pass
    (d* against contact_heights). params: the `risk` section.
    """
    base = float(params["base"])
    n = len(points)
    if n < 2:
        return base
    first, second = np.triu_indices(n, k=1)
    with_vehicle = (kinds[first] == VEHICLE) | (kinds[second] == VEHICLE)
    counted = with_vehicle & (kinds[first] != OTHER) & (kinds[second] != OTHER)
    first, second = first[counted], second[counted]
    if len(first) == 0:
        return base
    speed = np.linalg.norm(velocities, axis=1) / heights
    vehicle_speed = np.maximum(np.where(kinds[first] == VEHICLE, speed[first], 0.0),
                               np.where(kinds[second] == VEHICLE, speed[second], 0.0))
    scale = (heights[first] + heights[second]) / 2
    offset = (points[second] - points[first]) / scale[:, None]
    velocity = (velocities[second] - velocities[first]) / scale[:, None]
    when, gap = closest_approach(offset, velocity)
    closing = (velocity * -offset).sum(axis=1) / np.maximum(np.linalg.norm(offset, axis=1), 1e-9)
    on_course = (gap < params["contact_heights"]) & (when > 0) & (when <= params["horizon_sec"])
    on_course &= (closing >= params["min_closing"]) & (vehicle_speed >= params["min_vehicle_speed"])
    if not on_course.any():
        return base
    lateness = (when - params["ttc_alarm_sec"]) / params["ttc_softness_sec"]
    soon = 1.0 / (1.0 + np.exp(np.clip(lateness, -50.0, 50.0)))
    near = np.sqrt(np.clip(1.0 - gap / params["contact_heights"], 0.0, 1.0))
    return float(max(base, np.max((soon * near)[on_course])))
