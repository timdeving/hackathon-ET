"""Traffic-light phases, read from the signal heads the camera can see (docs/SIGNAL_DESIGN.md).

Perception cuts a window around each head out of every analysed frame, at full resolution, and
keeps only a small grid per window: how many lit red, amber and green pixels each block holds
(lamp_cells). The windows are fixed before the pass (light_windows), wide enough for any camera
shift, since alignment is only known after it. Then arm_phases uses the video's alignment to pick
the blocks inside each head's box, reads which lamp is lit from its position in the head, cleans
that up over time, and gives each arm its phase on every analysed frame. The rules read it
through PhaseTimeline.at(frames) (src/rules/common.py).

Only vehicle heads set an arm's phase. Pedestrian heads are read too, and how they relate to the
vehicle heads is logged: in the samples' frames the pedestrian head shows green together with the
vehicle head, not the other way round as SIGNAL_DESIGN.md §4 assumed, so it stays out of the
phases until the team settles what it means (docs/gpu/LOG.md, 2026-09-27).

numpy and OpenCV only, so laptops can run it on cached results.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np

from src.features.tracks import PointMapper
from src.perception.pipeline import PerceptionResult
from src.rules.common import AMBER, GREEN, RED, UNKNOWN
from src.scene.scene_map import Light, SceneMap
from src.video.reader import Box

log = logging.getLogger(__name__)

COLOURS = ("red", "amber", "green")  # the channels of a lamp_cells grid, in this order
PHASE_NAMES = {UNKNOWN: "unknown", RED: "red", AMBER: "amber", GREEN: "green"}

# The lamps of a head, top to bottom: the phase each one shows, and the colour channels that
# count for it. Red and amber lamps both glow warm on camera (an amber lamp reads half red), so
# only their position tells them apart; green is a colour of its own.
_WARM, _GREEN = (0, 1), (2,)
LAMPS = {
    "vehicle": ((RED, _WARM), (AMBER, _WARM), (GREEN, _GREEN)),
    "pedestrian": ((RED, _WARM), (GREEN, _GREEN)),
}

# The `signal` parameters that change the grids perception caches. The others only change how
# the grids are read, so they can be tuned on a cache without making a new one.
CELL_PARAMS = ("cell_px", "lit_saturation", "lit_value", "hues")


def light_windows(
    scene: SceneMap, margin_px: int, frame_size: tuple[int, int] | None = None
) -> dict[str, Box]:
    """Where to cut each head out of a video's frames: its box on the reference picture grown by
    margin_px on every side, in the video's own pixels, clipped to the frame.

    The margin has to cover any camera shift, since alignment is only known after the pass.
    frame_size (width, height) scales the windows to a video whose size differs from the
    reference picture's. A window that the clipping leaves empty is dropped.
    """
    width, height = frame_size or (scene.width, scene.height)
    scale_x, scale_y = width / scene.width, height / scene.height
    windows = {}
    for name, light in scene.lights.items():
        x0, y0, x1, y1 = light.box
        window = (
            max(0, math.floor((x0 - margin_px) * scale_x)),
            max(0, math.floor((y0 - margin_px) * scale_y)),
            min(width, math.ceil((x1 + margin_px) * scale_x)),
            min(height, math.ceil((y1 + margin_px) * scale_y)),
        )
        if window[0] < window[2] and window[1] < window[3]:
            windows[name] = window
    return windows


def lamp_cells(crop: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    """uint8 [3, rows, cols]: the lit red, amber and green pixels in each cell_px x cell_px block
    of a BGR crop (the blocks along the right and bottom edges may be partial).

    A pixel is lit when it is strongly coloured and not dark (lit_saturation, lit_value): a lit
    lamp is saturated in sun and in shade, but bright only in the shade. Its colour comes from
    its hue. params: the `signal` section of params.yaml. It runs on every analysed frame, so it
    sticks to whole-image OpenCV calls (about 1.5 ms for a 570 x 630 window).
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lit = cv2.inRange(hsv, (0, params["lit_saturation"], params["lit_value"]), (179, 255, 255))
    hue_table = np.zeros(256, np.uint8)  # hue -> 1 + its colour's channel; 0: none of them
    for channel, colour in enumerate(COLOURS):
        for low, high in params["hues"][colour]:
            hue_table[low : high + 1] = channel + 1
    colour_of = cv2.bitwise_and(cv2.LUT(cv2.extractChannel(hsv, 0), hue_table), lit)
    grids = [
        _block_sums(cv2.compare(colour_of, channel + 1, cv2.CMP_EQ), params["cell_px"]) // 255
        for channel in range(len(COLOURS))
    ]
    return np.minimum(np.stack(grids), 255).astype(np.uint8)


class Phases:
    """A signal's phase on every analysed frame of one video; a PhaseTimeline for the rules."""

    def __init__(self, codes: np.ndarray, stride: int) -> None:
        self.codes = np.asarray(codes, dtype=np.int8)  # frames 0, stride, 2 * stride, ...
        self.stride = stride

    def at(self, frames: np.ndarray) -> np.ndarray:
        """The phase at each of these frame numbers (that of the analysed frame at or before it);
        UNKNOWN outside the analysed range."""
        index = np.asarray(frames, dtype=np.int64) // self.stride
        codes = np.full(index.shape, UNKNOWN, dtype=np.int8)
        known = (index >= 0) & (index < len(self.codes))
        codes[known] = self.codes[index[known]]
        return codes

    def spans(self, fps: float) -> list[tuple[float, float, int]]:
        """(start_sec, end_sec, code) of each stretch of one phase, in order."""
        seconds = self.stride / fps
        return [(first * seconds, end * seconds, code) for first, end, code in _runs(self.codes)]


def arm_phases(
    result: PerceptionResult, mapper: PointMapper, scene: SceneMap, params: Mapping[str, Any]
) -> dict[str, Phases]:
    """The phase of every arm with a vehicle head that can be read, on every analysed frame; {}
    when the result has no light data (e.g. a cache made before the lights were read).

    mapper: the video's camera alignment. params: the whole of params.yaml.
    """
    return combine_heads(read_heads(result, mapper, scene, params), scene, result)


def read_heads(
    result: PerceptionResult, mapper: PointMapper, scene: SceneMap, params: Mapping[str, Any]
) -> dict[str, Phases]:
    """Each head's own phase on every analysed frame, by the name of its light in the scene map.

    A head is left out when the scene map doesn't describe it the way this reads it (a vertical
    head), or when its box isn't fully inside its window: the camera moved further than
    signal.margin_px. Unknown means don't judge.
    """
    signal = params["signal"]
    analysed_per_sec = result.info.fps / result.stride
    heads = {}
    for name, cells in sorted(result.lights.items()):
        light, window = scene.lights.get(name), result.light_windows.get(name)
        if light is None or window is None or light.layout != "vertical":
            continue
        kind = "pedestrian" if light.pedestrian else "vehicle"
        lamp_of = _lamp_of_block(window, cells.shape[2:], light, mapper, signal, kind)
        if not all((lamp_of == k).any() for k in range(len(LAMPS[kind]))):
            log.warning(
                "%s: light %s: its box isn't inside the window cut around it (a camera shift "
                "beyond signal.margin_px?); not read",
                result.info.name,
                name,
            )
            continue
        raw = _brightest_lamp(cells, lamp_of, LAMPS[kind], signal["min_lit_px"])
        heads[name] = Phases(_clean(raw, analysed_per_sec, signal), result.stride)
    return heads


def combine_heads(
    heads: Mapping[str, Phases], scene: SceneMap, result: PerceptionResult
) -> dict[str, Phases]:
    """Each arm's phase from the vehicle heads that control it: the first head (by name) that
    reads a frame decides it. Pedestrian heads are only compared with them, in the log."""
    arms: dict[str, np.ndarray] = {}
    for name, phases in sorted(heads.items()):
        light = scene.lights[name]
        if light.pedestrian:
            continue
        for arm in light.controls:
            codes = arms.setdefault(arm, np.full(len(phases.codes), UNKNOWN, dtype=np.int8))
            unread = codes == UNKNOWN
            codes[unread] = phases.codes[unread]
    _log_heads(heads, scene, result)
    return {arm: Phases(codes, result.stride) for arm, codes in arms.items()}


def _block_sums(mask: np.ndarray, cell: int) -> np.ndarray:
    """The sum of a uint8 mask over each cell x cell block (those along the right and bottom
    edges may be partial), read off its integral image at the blocks' corners."""
    height, width = mask.shape
    sums = cv2.integral(mask)  # (height + 1, width + 1): the sum of everything above and left
    ys = np.minimum(np.arange(0, height + cell, cell), height)
    xs = np.minimum(np.arange(0, width + cell, cell), width)
    corners = sums[np.ix_(ys, xs)].astype(np.int64)
    return corners[1:, 1:] - corners[:-1, 1:] - corners[1:, :-1] + corners[:-1, :-1]


def _lamp_of_block(
    window: Box,
    grid_shape: tuple[int, int],
    light: Light,
    mapper: PointMapper,
    signal: Mapping[str, Any],
    kind: str,
) -> np.ndarray:
    """int [rows, cols]: which lamp of the head (0 = the top one) each block of the window
    covers, from where the block's centre lands in the head's box on the reference picture;
    -1 outside the box."""
    rows, cols = grid_shape
    cell = signal["cell_px"]
    xs = window[0] + (np.arange(cols) + 0.5) * cell
    ys = window[1] + (np.arange(rows) + 0.5) * cell
    grid_x, grid_y = np.meshgrid(xs, ys)
    reference = mapper.to_reference(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
    x0, y0, x1, y1 = light.box
    across = (reference[:, 0] - x0) / (x1 - x0)
    down = (reference[:, 1] - y0) / (y1 - y0)
    lamp = np.searchsorted(signal["lamp_bands"][kind], down, side="right")
    inside = (across >= 0) & (across < 1) & (down >= 0) & (down < 1)
    return np.where(inside, lamp, -1).reshape(rows, cols)


def _brightest_lamp(
    cells: np.ndarray, lamp_of: np.ndarray, lamps: tuple, min_lit_px: int
) -> np.ndarray:
    """Each frame's phase: the lamp with the most lit pixels of its own colours, if it has at
    least min_lit_px; UNKNOWN otherwise."""
    flat = cells.reshape(len(cells), len(COLOURS), -1)
    blocks_of = lamp_of.ravel()
    scores = np.zeros((len(cells), len(lamps)), dtype=np.int64)
    for k, (_, colours) in enumerate(lamps):
        in_lamp = flat[:, :, blocks_of == k]  # [frames, colours, the lamp's blocks]
        scores[:, k] = in_lamp[:, list(colours)].sum(axis=(1, 2), dtype=np.int64)
    codes = np.array([code for code, _ in lamps], dtype=np.int8)[scores.argmax(axis=1)]
    return np.where(scores.max(axis=1) >= min_lit_px, codes, UNKNOWN).astype(np.int8)


def _clean(codes: np.ndarray, per_sec: float, signal: Mapping[str, Any]) -> np.ndarray:
    """Phases over time: a majority vote over smooth_sec; then phases too short to be real
    become unknown, and short unknown gaps inside one phase take that phase."""
    window = 2 * round(signal["smooth_sec"] * per_sec / 2) + 1  # the nearest odd count
    codes = _majority(codes, window)
    codes = _drop_short(codes, per_sec, signal)
    return _fill_gaps(codes, signal["min_phase_sec"] * per_sec)


def _majority(codes: np.ndarray, window: int) -> np.ndarray:
    """Each frame takes the phase most frequent within `window` frames centred on it, keeping
    its own unless another is strictly more frequent."""
    n = len(codes)
    if n == 0 or window <= 1:
        return codes.copy()
    counts = np.cumsum(np.eye(len(PHASE_NAMES), dtype=np.int32)[codes], axis=0)
    counts = np.vstack([np.zeros((1, len(PHASE_NAMES)), np.int32), counts])
    index, half = np.arange(n), window // 2
    votes = counts[np.minimum(index + half + 1, n)] - counts[np.maximum(index - half, 0)]
    own = votes[index, codes]
    return np.where(own >= votes.max(axis=1), codes, votes.argmax(axis=1)).astype(np.int8)


def _drop_short(codes: np.ndarray, per_sec: float, signal: Mapping[str, Any]) -> np.ndarray:
    """A phase shorter than its minimum becomes unknown (min_phase_sec; amber: amber_min_sec).
    No real phase is that short; a flicker, or a coloured car behind the head, is."""
    out = codes.copy()
    for first, end, code in _runs(codes):
        shortest = signal["amber_min_sec"] if code == AMBER else signal["min_phase_sec"]
        if code != UNKNOWN and (end - first) / per_sec < shortest:
            out[first:end] = UNKNOWN
    return out


def _fill_gaps(codes: np.ndarray, max_frames: float) -> np.ndarray:
    """An unknown gap shorter than max_frames between two stretches of the same phase takes that
    phase: the head was hidden for a moment. Gaps between different phases stay unknown."""
    out = codes.copy()
    runs = _runs(codes)
    for before, (first, end, code), after in zip(runs, runs[1:], runs[2:], strict=False):
        if code == UNKNOWN and before[2] == after[2] != UNKNOWN and end - first < max_frames:
            out[first:end] = before[2]
    return out


def _runs(codes: np.ndarray) -> list[tuple[int, int, int]]:
    """(first, end, code) of each stretch of one code; end is exclusive."""
    if len(codes) == 0:
        return []
    changes = np.flatnonzero(np.diff(codes)) + 1
    firsts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(codes)]])
    return [(int(a), int(b), int(codes[a])) for a, b in zip(firsts, ends, strict=True)]


def _log_heads(heads: Mapping[str, Phases], scene: SceneMap, result: PerceptionResult) -> None:
    """One line per head: how much of the video it reads, and its phases; and for each
    pedestrian head, how it relates to the vehicle heads of its arms."""
    for name, phases in heads.items():
        shares = {code: float(np.mean(phases.codes == code)) for code in PHASE_NAMES}
        log.info(
            "%s: light %s read on %.0f%% of the analysed frames (red %.0f%%, amber %.0f%%, "
            "green %.0f%%), %d changes of phase",
            result.info.name,
            name,
            100 * (1 - shares[UNKNOWN]),
            100 * shares[RED],
            100 * shares[AMBER],
            100 * shares[GREEN],
            len(_runs(phases.codes)) - 1,
        )
    for name, walk in heads.items():
        if not scene.lights[name].pedestrian:
            continue
        for other, drive in heads.items():
            shared = set(scene.lights[name].controls) & set(scene.lights[other].controls)
            if scene.lights[other].pedestrian or not shared:
                continue
            both = (walk.codes != UNKNOWN) & (drive.codes != UNKNOWN)
            walking, driving = walk.codes[both], drive.codes[both]
            shares = [
                100 * float(np.mean((walking == a) & (driving == b))) if both.any() else 0.0
                for a, b in [(GREEN, GREEN), (GREEN, RED), (RED, RED), (RED, GREEN)]
            ]
            log.info(
                "%s: pedestrian light %s against %s, on the %d frames both read: green with "
                "green %.0f%%, green with red %.0f%%, red with red %.0f%%, red with green %.0f%%",
                result.info.name,
                name,
                other,
                int(both.sum()),
                *shares,
            )
