"""Rules that read the traffic lights: red_light, and stop_line once the lights are known. The
phases come from a stand-in for the GPU PC's src/scene/signals.py."""
from __future__ import annotations

import numpy as np
import pytest

from src.features.tracks import NoAlignment
from src.rules import find_events
from src.rules.common import AMBER, GREEN, RED, UNKNOWN
from src.scene.scene_map import SceneMap
from tests.synthetic_tracks import CAR, FPS, params_with, result_of, street_scene, times, track
from tests.test_rules_batch_b import drive_stop_drive, standing

TOLERANCE = 0.25


class Phases:
    """A phase timeline that changes at the given times: [(from_second, code), ...]."""

    def __init__(self, changes: list[tuple[float, int]]) -> None:
        self.changes = changes

    def at(self, frames: np.ndarray) -> np.ndarray:
        t = np.asarray(frames) / FPS
        codes = np.full(len(t), UNKNOWN)
        for start, code in self.changes:
            codes[t >= start] = code
        return codes


def events(label, *tracks, phases=None, seconds=10.0) -> list[tuple[float, float]]:
    segments = find_events(result_of(*tracks, seconds=seconds), NoAlignment(),
                           SceneMap(street_scene()), params_with(), [label], phases=phases)
    return sorted((round(s.start, 2), round(s.end, 2)) for s in segments)


def assert_one_event(found, start: float, end: float) -> None:
    assert len(found) == 1, found
    assert found[0] == pytest.approx((start, end), abs=TOLERANCE)


def through_the_line(seconds: float = 6.0) -> np.ndarray:
    """A car driving east in lane west_in_1 at 100 px/s, never stopping: its centre is at
    x = 340 + 100 t, so its front (centre + 40) reaches the stop line (x = 600) at t = 2.2 s and
    is past it from the next frame. It stays in view until the end."""
    return drive_stop_drive(560, 400, stop_from=2.2, stop_to=2.2, seconds=seconds)


# red_light -----------------------------------------------------------------------------------

def test_crossing_the_line_on_red_is_red_light():
    found = events("red_light", through_the_line(), phases={"west": Phases([(0.0, RED)])})
    assert_one_event(found, 2.2, 6.05)  # ends when it leaves the picture (its track ends)


def test_crossing_on_green_is_fine():
    assert events("red_light", through_the_line(), phases={"west": Phases([(0.0, GREEN)])}) == []


def test_the_first_moment_of_red_is_not_called():
    phases = {"west": Phases([(0.0, GREEN), (1.75, RED)])}  # red only 0.5 s before crossing
    assert events("red_light", through_the_line(), phases=phases) == []


def test_a_vehicle_first_seen_past_the_line_on_red_is_red_light_from_then():
    """Like a moped that comes along the kerb: it's never seen crossing the line."""
    t = times(2, 8)
    moped = track(1, CAR, t, 615 + 100 * (t - 2), 400, width=80, height=50)
    assert_one_event(events("red_light", moped, phases={"west": Phases([(0.0, RED)])}), 2.0, 8.05)


def test_a_vehicle_seen_past_the_line_only_briefly_is_not_called():
    t = times(2, 4)  # 2 s: a blip, or a car the tracker found for a moment
    blip = track(1, CAR, t, 615 + 100 * (t - 2), 400, width=80, height=50)
    assert events("red_light", blip, phases={"west": Phases([(0.0, RED)])}) == []


def test_a_red_light_run_ends_when_the_vehicle_leaves_even_after_waiting_past_the_line():
    car = drive_stop_drive(580, 400, stop_from=3, stop_to=13, seconds=16)  # waits 10 s past it
    found = events("red_light", car, phases={"west": Phases([(0.0, RED)])}, seconds=16.0)
    assert_one_event(found, 2.85, 16.05)  # leaves the picture at 16 s, not 8 s after crossing


def test_without_read_lights_there_is_no_call():
    assert events("red_light", through_the_line()) == []
    assert events("red_light", through_the_line(), phases={"west": Phases([])}) == []


# stop_line with the lights read --------------------------------------------------------------

def test_with_lights_a_stop_past_the_line_ends_when_the_light_turns_green():
    car = drive_stop_drive(580, 400, stop_from=3, stop_to=8, seconds=10)  # front at 620
    phases = {"west": Phases([(0.0, RED), (6.0, GREEN)])}
    assert_one_event(events("stop_line", car, phases=phases), 3.0, 6.0)  # nobody else waiting


def test_with_lights_driving_off_on_red_is_not_stop_line():
    car = drive_stop_drive(580, 400, stop_from=3, stop_to=6, seconds=10)  # stops, then runs it
    assert events("stop_line", car, phases={"west": Phases([(0.0, RED)])}) == []


def test_with_lights_moving_off_on_red_and_amber_still_ends_at_the_green():
    car = drive_stop_drive(580, 400, stop_from=3, stop_to=6, seconds=10)
    phases = {"west": Phases([(0.0, RED), (5.0, AMBER), (7.0, GREEN)])}  # red and amber from 5 s
    assert_one_event(events("stop_line", car, phases=phases), 3.0, 7.0)


def test_with_lights_read_around_it_a_stop_while_they_are_unknown_is_not_guessed():
    car = drive_stop_drive(580, 400, stop_from=3, stop_to=7, seconds=10)
    waiting = standing(500, 470, seconds=10, track_id=2)  # would imply red without the lights
    phases = {"west": Phases([(0.0, RED), (3.0, UNKNOWN), (7.0, RED)])}  # a bus hides the head
    assert events("stop_line", car, waiting, phases=phases) == []


def test_with_lights_a_stop_past_the_line_on_green_is_not_stop_line():
    car = drive_stop_drive(580, 400, stop_from=3, stop_to=8, seconds=10)
    waiting = standing(500, 470, seconds=10, track_id=2)  # would have implied red, lights say not
    assert events("stop_line", car, waiting, phases={"west": Phases([(0.0, GREEN)])}) == []
