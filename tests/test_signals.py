"""Reading the traffic lights (src/scene/signals.py), on heads drawn into synthetic frames."""
from __future__ import annotations

import dataclasses
import json
import logging

import cv2
import numpy as np
import pytest

from src import part_a
from src.features.tracks import NoAlignment
from src.perception.cache import save_result
from src.perception.pipeline import PerceptionResult
from src.rules import find_events
from src.rules.common import AMBER, GREEN, RED, UNKNOWN
from src.scene import signals
from src.scene.scene_map import SceneMap
from tests.synthetic_tracks import params_with, result_of, street_scene
from tests.test_rules_batch_c import through_the_line

PARAMS = params_with(signal={"margin_px": 40})
SIGNAL = PARAMS["signal"]
HEAD = (700, 100, 740, 170)  # a vehicle head on the street picture, above the road: 40 x 70 px
WALK = (820, 100, 850, 150)  # a pedestrian head beside it
# Lamp centres as fractions of the head's height, as measured on the samples' heads.
TOP, MIDDLE, BOTTOM = 0.28, 0.49, 0.71
# Lit lamps (BGR): bright in the shade, and dim but just as saturated in the sun.
RED_LAMP, AMBER_LAMP, GREEN_LAMP = (40, 40, 230), (30, 160, 250), (150, 220, 40)
SUNLIT_RED_LAMP = (20, 20, 110)


def street_with_heads() -> SceneMap:
    """The test street, with a vehicle head and a pedestrian head for its `west` arm."""
    data = street_scene()
    data["lights"] = {
        "head": {"box": list(HEAD), "controls": ["west"], "layout": "vertical"},
        "walk": {"box": list(WALK), "controls": ["west"], "layout": "vertical", "pedestrian": True},
    }
    return SceneMap(data)


SCENE = street_with_heads()
WINDOWS = signals.light_windows(SCENE, SIGNAL["margin_px"])


def frame_with(lamps: dict[str, tuple[float, tuple]] | None = None, shift=(0, 0), blob=None):
    """A frame of the street: dark heads, lamps lit at {head: (height fraction, colour)}, the
    camera moved by `shift` pixels, and optionally a red blob at `blob` (x, y)."""
    frame = np.full((600, 1000, 3), 70, np.uint8)
    dx, dy = shift
    for name, (x0, y0, x1, y1) in (("head", HEAD), ("walk", WALK)):
        cv2.rectangle(frame, (x0 + dx, y0 + dy), (x1 + dx - 1, y1 + dy - 1), (25, 25, 25), -1)
        if lamps and name in lamps:
            fraction, colour = lamps[name]
            centre = ((x0 + x1) // 2 + dx, round(y0 + fraction * (y1 - y0)) + dy)
            cv2.circle(frame, centre, 6, colour, -1)
    if blob is not None:
        cv2.circle(frame, blob, 6, RED_LAMP, -1)
    return frame


def result_with_lights(frames: list[np.ndarray], tracks=()) -> PerceptionResult:
    """A perception result (10 frames per second, stride 1) whose light data comes from these
    frames, cut and counted the way perception does it."""
    grids = {name: [] for name in WINDOWS}
    for frame in frames:
        for name, (x0, y0, x1, y1) in WINDOWS.items():
            grids[name].append(signals.lamp_cells(frame[y0:y1, x0:x1], SIGNAL))
    result = result_of(*tracks, seconds=len(frames) / 10)
    lights = {name: np.stack(cells) for name, cells in grids.items()}
    return dataclasses.replace(result, lights=lights, light_windows=WINDOWS)


def head_codes(frames: list[np.ndarray], mapper=None, name="head") -> np.ndarray:
    heads = signals.read_heads(result_with_lights(frames), mapper or NoAlignment(), SCENE, PARAMS)
    return heads[name].codes


class Shifted:
    """Camera alignment for a view moved by `shift` pixels."""

    def __init__(self, shift) -> None:
        self.shift = np.asarray(shift, dtype=float)

    def to_reference(self, points, frames=None):
        return np.asarray(points, dtype=float).reshape(-1, 2) - self.shift


# windows and counts ---------------------------------------------------------------------------

def test_a_window_is_the_box_grown_by_the_margin_and_stays_in_the_frame():
    data = street_scene()
    data["lights"] = {"edge": {"box": [10, 20, 50, 90], "controls": ["west"], "layout": "vertical"}}
    scene = SceneMap(data)
    assert signals.light_windows(scene, 40) == {"edge": (0, 0, 90, 130)}
    # a video at half the reference picture's size gets the window at half the size
    assert signals.light_windows(scene, 40, (500, 300)) == {"edge": (0, 0, 45, 65)}


def test_lamp_cells_count_lit_pixels_by_colour_and_block():
    crop = np.full((48, 64, 3), 70, np.uint8)
    cv2.circle(crop, (20, 20), 6, RED_LAMP, -1)
    cells = signals.lamp_cells(crop, SIGNAL)
    assert cells.shape == (3, 6, 8) and cells.dtype == np.uint8
    red, amber, green = cells
    assert red.sum() == np.count_nonzero(cv2.inRange(crop, RED_LAMP, RED_LAMP))
    assert red[1:4, 1:4].sum() == red.sum()  # all in the blocks around (20, 20)
    assert amber.sum() == 0 and green.sum() == 0


def test_only_strongly_coloured_pixels_count_as_lit():
    crop = np.full((32, 32, 3), 70, np.uint8)
    crop[:8] = (240, 240, 240)  # glare, a white car: bright but not coloured
    crop[8:16] = (20, 20, 45)  # an unlit red lens: coloured but dark
    crop[16:24] = SUNLIT_RED_LAMP  # a lit lamp in the sun: dim, but as saturated as any
    red = signals.lamp_cells(crop, SIGNAL)[0]
    assert red[:2].sum() == 0 and red[2].sum() == 8 * 32


# one head -------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("fraction", "colour", "phase"),
    [(TOP, RED_LAMP, RED), (MIDDLE, AMBER_LAMP, AMBER), (BOTTOM, GREEN_LAMP, GREEN),
     (TOP, SUNLIT_RED_LAMP, RED)],
)
def test_the_lit_lamp_gives_the_phase(fraction, colour, phase):
    codes = head_codes([frame_with({"head": (fraction, colour)})] * 30)
    assert (codes == phase).all()


def test_red_and_amber_together_read_amber_not_red():
    """The 3 s before green: both lamps lit. Moving off then isn't running a red light."""
    frame = frame_with({"head": (TOP, RED_LAMP)})
    x, y = (HEAD[0] + HEAD[2]) // 2, round(HEAD[1] + MIDDLE * (HEAD[3] - HEAD[1]))
    cv2.circle(frame, (x, y), 6, AMBER_LAMP, -1)
    assert (head_codes([frame] * 30) == AMBER).all()


def test_an_amber_lamp_that_looks_red_still_reads_amber():
    codes = head_codes([frame_with({"head": (MIDDLE, (40, 60, 240))})] * 30)  # orange-red
    assert (codes == AMBER).all()


def test_nothing_lit_is_unknown():
    assert (head_codes([frame_with()] * 30) == UNKNOWN).all()


def test_a_red_light_outside_the_head_does_not_count():
    frames = [frame_with(blob=(HEAD[2] + 20, HEAD[1] + 20))] * 30  # beside the head, in its window
    assert (head_codes(frames) == UNKNOWN).all()


def test_the_head_is_read_where_the_camera_moved_it():
    frames = [frame_with({"head": (TOP, RED_LAMP)}, shift=(25, -15))] * 30
    assert (head_codes(frames, Shifted((25, -15))) == RED).all()


def test_a_head_moved_out_of_its_window_is_not_read(caplog):
    frames = [frame_with({"head": (TOP, RED_LAMP)})] * 30
    with caplog.at_level(logging.WARNING):
        heads = signals.read_heads(result_with_lights(frames), Shifted((200, 0)), SCENE, PARAMS)
    assert "head" not in heads
    assert "isn't inside the window" in caplog.text


# over time ------------------------------------------------------------------------------------

def test_flicker_and_a_short_occlusion_are_smoothed_away():
    red, green = frame_with({"head": (TOP, RED_LAMP)}), frame_with({"head": (BOTTOM, GREEN_LAMP)})
    frames = [red] * 20 + [green] + [red] * 14 + [frame_with()] * 8 + [red] * 25  # 0.8 s hidden
    assert (head_codes(frames) == RED).all()


def test_a_long_occlusion_stays_unknown():
    red = frame_with({"head": (TOP, RED_LAMP)})
    codes = head_codes([red] * 40 + [frame_with()] * 30 + [red] * 40)
    assert (codes[45:65] == UNKNOWN).all() and (codes[:35] == RED).all()


def test_a_gap_between_two_different_phases_stays_unknown():
    red, green = frame_with({"head": (TOP, RED_LAMP)}), frame_with({"head": (BOTTOM, GREEN_LAMP)})
    codes = head_codes([green] * 40 + [frame_with()] * 10 + [red] * 40)
    assert list(signals._runs(codes)) == [(0, 40, GREEN), (40, 50, UNKNOWN), (50, 90, RED)]


def test_a_whole_cycle_keeps_its_short_amber():
    frames = ([frame_with({"head": (BOTTOM, GREEN_LAMP)})] * 40
              + [frame_with({"head": (MIDDLE, AMBER_LAMP)})] * 30
              + [frame_with({"head": (TOP, RED_LAMP)})] * 40)
    assert list(signals._runs(head_codes(frames))) == [(0, 40, GREEN), (40, 70, AMBER),
                                                       (70, 110, RED)]


def test_phases_at_any_frame_number():
    phases = signals.Phases(np.array([RED, RED, GREEN]), stride=3)  # frames 0, 3 and 6
    frames = np.array([0, 2, 3, 6, 8, 9, -1])
    assert phases.at(frames).tolist() == [RED, RED, RED, GREEN, GREEN, UNKNOWN, UNKNOWN]
    assert phases.spans(fps=3.0) == [(0.0, 2.0, RED), (2.0, 3.0, GREEN)]


# arms -----------------------------------------------------------------------------------------

def test_an_arm_takes_its_phase_from_the_vehicle_head_only():
    both = frame_with({"head": (TOP, RED_LAMP), "walk": (0.7, GREEN_LAMP)})
    walk_only = frame_with({"walk": (0.7, GREEN_LAMP)})
    phases = signals.arm_phases(result_with_lights([both] * 30 + [walk_only] * 30),
                                NoAlignment(), SCENE, PARAMS)
    assert list(phases) == ["west"]
    codes = phases["west"].codes
    assert (codes[:25] == RED).all() and (codes[35:] == UNKNOWN).all()


def test_a_result_without_light_data_has_no_phases():
    assert signals.arm_phases(result_of(), NoAlignment(), SCENE, PARAMS) == {}


def test_a_car_crossing_on_the_red_read_from_the_lights_is_red_light():
    """End to end: the lights read red, and a car crosses the stop line (SIGNAL_DESIGN.md §11)."""
    car = through_the_line()  # its front crosses the line at 2.2 s; it stays in view to 6 s
    for lamp, events in [((TOP, RED_LAMP), 1), ((BOTTOM, GREEN_LAMP), 0)]:
        result = result_with_lights([frame_with({"head": lamp})] * 61, tracks=[car])
        phases = signals.arm_phases(result, NoAlignment(), SCENE, PARAMS)
        found = find_events(result, NoAlignment(), SCENE, PARAMS, ["red_light"], phases=phases)
        assert len(found) == events


def test_the_check_tool_prints_timelines_and_stop_line_crossings(tmp_path, run_python):
    red = frame_with({"head": (TOP, RED_LAMP), "walk": (0.3, RED_LAMP)})
    green = frame_with({"head": (BOTTOM, GREEN_LAMP), "walk": (0.7, GREEN_LAMP)})
    frames = [red] * 61 + [green] * 30  # red, then green from 6.1 s
    folder = save_result(result_with_lights(frames, tracks=[through_the_line()]),
                         tmp_path / "cache", {"test": True})
    (folder / "lights").mkdir()
    x0, y0, x1, y1 = WINDOWS["head"]
    for index in range(0, len(frames), 10):  # a snapshot a second, as cache_tracks saves them
        cv2.imwrite(str(folder / "lights" / f"head_{index:06d}.jpg"), frames[index][y0:y1, x0:x1])
    data = street_scene()
    data["lights"] = {name: {"box": list(light.box), "controls": list(light.controls),
                             "layout": light.layout, "pedestrian": light.pedestrian}
                      for name, light in SCENE.lights.items()}
    (tmp_path / "scene.json").write_text(json.dumps(data))
    run = run_python("-m", "tools.check_signals", "--cache", str(tmp_path / "cache"),
                     "--scene", str(tmp_path / "scene.json"), "--sheets", str(tmp_path / "sheets"))
    assert run.returncode == 0, run.stdout + run.stderr
    assert "light head: read on 100% of the frames: red 0.0-6.1, green 6.1-9.1" in run.stdout
    assert "arm west" in run.stdout
    crossings = run.stdout.split("stop-line crossings by phase:")[1]
    assert crossings.split()[:6] == ["lane", "green", "amber", "red", "unknown", "west_in_1"]
    assert crossings.split()[6:10] == ["0", "0", "1", "0"]  # the car crossed on red
    sheet = cv2.imread(str(tmp_path / "sheets" / "street.mp4_head.jpg"))
    assert sheet is not None and sheet.shape[0] > sheet.shape[1] / 3  # one change: two tiles


def test_part_a_loses_only_the_lights_when_reading_them_fails(monkeypatch, caplog):
    def broken(*args):
        raise RuntimeError("boom")

    monkeypatch.setattr(part_a, "arm_phases", broken)
    result = result_with_lights([frame_with({"head": (TOP, RED_LAMP)})] * 10)
    with caplog.at_level(logging.ERROR):
        assert part_a._read_lights(result, NoAlignment(), SCENE, PARAMS) == {}
    assert "reading the traffic lights failed" in caplog.text
