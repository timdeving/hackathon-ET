"""stop_line: a vehicle stops past the stop line on red, without entering the junction.

Starts when the vehicle stops and ends when the signal turns green (task conventions):

- past the line: the front of the vehicle (the bottom point of its box furthest along its
  lane's flow) is beyond its arm's stop line, while its ground point isn't in the junction and
  is within max_line_heights of its own box heights of the line. A vehicle stopped far across
  the junction is somewhere else, whatever lane it drove in long before;
- on red, and the end: where the arm's traffic lights are read (docs/SIGNAL_DESIGN.md), the
  stop must mostly be on red before the light turns green, and the vehicle must wait for the
  plain red to end: driving off on red is running the light, not stopping past the line. The
  event ends at the green, even when the vehicle moves off a moment before it, on red and amber
  (looking ahead green_within_sec). A stop in an unknown stretch while the lights are read
  around it (lights_window_sec) gets no call. Only where they aren't read at all is red guessed
  from another vehicle of the arm waiting too (no call with nobody waiting), and the stop ends
  when the vehicle moves off.

The arm is the one of the last incoming lane the vehicle drove in before stopping, or, for a
vehicle never seen in a drawn lane (a moped along the kerb), of the incoming lane nearest to
where it stops. Only lanes under the arm's signal count.
"""
from __future__ import annotations

import numpy as np

from src.events import Segment
from src.features.tracks import VEHICLES
from src.rules.common import (
    GREEN,
    RED,
    UNKNOWN,
    FrameIndex,
    PhaseTimeline,
    RuleContext,
    front_points,
    lane_direction,
    last_incoming_lane,
    mostly,
    nearest_incoming_lane,
    past_stop_line,
    span,
    spread,
    stops,
)
from src.scene.geometry import distance_to_polyline
from src.scene.scene_map import SceneMap

LABEL = "stop_line"


def find(context: RuleContext) -> list[Segment]:
    p, scene, features = context.params, context.scene, context.features
    others = FrameIndex(features.rows_of(VEHICLES))
    segments = []
    for info, rows in features.tracks_of(VEHICLES):
        for first, last in stops(rows, features.dt, p["min_sec"], p["gap_sec"]):
            stop = rows[first : last + 1]
            ground = np.column_stack([stop["x"], stop["y"]])
            lane = last_incoming_lane(rows[: last + 1], scene)
            if lane is None:
                lane = nearest_incoming_lane(ground, scene)
            if lane is None or not scene.lanes[lane].signal:
                continue
            arm = scene.lanes[lane].arm
            if arm not in scene.stop_lines:
                continue
            line_gap = distance_to_polyline(ground, scene.stop_lines[arm])
            if np.median(line_gap / np.maximum(stop["height"], 1.0)) > p["max_line_heights"]:
                continue
            front = front_points(stop, lane_direction(lane, scene))
            if not mostly(past_stop_line(front, lane, scene) & ~stop["junction"]):
                continue
            timed = _on_red(rows, first, last, arm, context, others)
            if timed is not None:
                segments.append(Segment(*timed, LABEL, (int(info["track_id"]),)))
    return segments


def _on_red(
    rows: np.ndarray, first: int, last: int, arm: str, context: RuleContext, others: FrameIndex
) -> tuple[float, float] | None:
    """(start, end) if this stop is on red, else None: from the lights where they're read,
    otherwise guessed from the queue."""
    dt = context.features.dt
    start, end = span(rows["t"], first, last, dt)
    timeline = context.phases.get(arm)
    if timeline is not None:
        phase = timeline.at(rows["frame"][first : last + 1])
        if mostly(phase != UNKNOWN):
            green = np.flatnonzero(phase == GREEN)
            before_green = phase[: green[0]] if len(green) else phase
            known = before_green[before_green != UNKNOWN]
            if not mostly(known == RED):
                return None
            if len(green):  # the event ends when the signal turns green
                return start, float(rows["t"][first + green[0]] - dt / 2)
            if phase[-1] == RED:  # moved off on plain red, or lost from sight: no call
                return None
            turns_green = _next_green(timeline, int(rows["frame"][last]), context)
            return start, turns_green if turns_green is not None else end
        if _lights_read_nearby(timeline, rows["frame"][first], rows["frame"][last], context):
            return None  # the flashing green, or a bus hiding the head: unknown, no guess
    stop = rows[first : last + 1]
    waiting = _others_waiting(stop[spread(len(stop))], arm, others, context.scene)
    return (start, end) if mostly(waiting) else None


def _next_green(timeline: PhaseTimeline, frame: int, context: RuleContext) -> float | None:
    """When the signal turns green, within green_within_sec after this frame; None if it
    doesn't."""
    fps = context.features.fps
    frames = frame + np.arange(round(context.params["green_within_sec"] * fps) + 1)
    green = np.flatnonzero(timeline.at(frames) == GREEN)
    return float(frames[green[0]] / fps) if len(green) else None


def _lights_read_nearby(
    timeline: PhaseTimeline, first: int, last: int, context: RuleContext
) -> bool:
    """Are the lights read on most frames within lights_window_sec of this stop? Then a stop in
    an unknown stretch is judged unknown, not guessed from the queue."""
    reach = round(context.params["lights_window_sec"] * context.features.fps)
    frames = np.arange(max(0, first - reach), last + reach + 1, context.features.stride)
    return mostly(timeline.at(frames) != UNKNOWN)


def _others_waiting(rows: np.ndarray, arm: str, others: FrameIndex, scene: SceneMap) -> np.ndarray:
    """For each row: does another vehicle wait, stationary, in an incoming lane of `arm`?"""
    arm_lanes = [i for i, lane in enumerate(scene.lanes) if lane.arm == arm and lane.role == "in"]
    waiting = np.zeros(len(rows), dtype=bool)
    for i, row in enumerate(rows):
        near = others.at(int(row["frame"]))
        waiting[i] = bool(
            (near["stationary"] & np.isin(near["lane"], arm_lanes)
             & (near["track_id"] != row["track_id"])).any()
        )
    return waiting
