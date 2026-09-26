"""Synthetic tracks and a small street, for testing features and rules without video."""
from __future__ import annotations

import copy

import numpy as np

from src.config import load_params
from src.perception.pipeline import DETECTION_DTYPE, TRACK_DTYPE, PerceptionResult
from src.video.probe import VideoInfo

PERSON, CAR, MOTORCYCLE, TRUCK = 0, 2, 3, 7
FPS = 10.0  # with stride 1: one update every 0.1 s, so times are easy to reason about


def street_scene(exits: dict[str, list[str]] | None = None, side_road: bool = False) -> dict:
    """A 1000x600 picture. The road runs from y = 200 to 500, with pavements above and below.

    West of x = 600 it has two carriageways split by a median (y = 340-360): below it, two
    incoming lanes flowing right (east) towards a stop line at x = 600; above it, two outgoing
    lanes flowing left. A zebra crossing covers x = 600-640; east of it is the junction
    (x = 640-800), then open road with a bus stop at the lower kerb (x = 850-950). A solid line
    runs diagonally from (0, 400) to (600, 460). exits: allowed exits per lane name (default:
    unknown).

    side_road adds a road leaving the junction southwards (x = 700-820, down to the bottom edge),
    and exit zones where the roads leave the picture: `east` (x >= 960) and `south` (y >= 570).
    """
    def band(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

    def lane(name: str, role: str, y0: float, y1: float) -> dict:
        middle = (y0 + y1) / 2
        flow = [[20, middle], [580, middle]] if role == "in" else [[580, middle], [20, middle]]
        return {"arm": "west", "role": role, "signal": role == "in",
                "exits": (exits or {}).get(name), "polygon": band(0, y0, 600, y1), "flow": flow}

    roads = {"main": band(0, 200, 1000, 500)}
    exit_zones = {}
    if side_road:
        roads["south"] = band(700, 500, 820, 600)
        exit_zones = {"east": band(960, 200, 1000, 500), "south": band(700, 570, 820, 600)}
    return {
        "format": 1,
        "reference": {"video": "street.mp4", "width": 1000, "height": 600},
        "roads": roads,
        "exit_zones": exit_zones,
        "islands": {"median": band(0, 340, 600, 360)},
        "crossings": {"zebra": band(600, 200, 640, 500)},
        "junction": band(640, 200, 800, 500),
        "lanes": {
            name: lane(name, role, y0, y1)
            for name, role, y0, y1 in [
                ("west_in_1", "in", 360, 430),
                ("west_in_2", "in", 430, 500),
                ("west_out_1", "out", 270, 340),
                ("west_out_2", "out", 200, 270),
            ]
        },
        "stop_lines": {"west": [[600, 360], [600, 500]]},
        "solid_lines": {"divider": [[0, 400], [600, 460]]},
        "lights": {},
        "parking": {},
        "bus_stops": {"east_kerb": band(850, 460, 950, 500)},
        "no_uturn": {},
        "ground_points": {},
        "image_to_ground": None,
    }


def track(
    track_id: int,
    class_id: int,
    t: np.ndarray,
    bottom_x: np.ndarray,
    bottom_y: np.ndarray,
    width: float,
    height: float,
    score: float = 0.9,
) -> np.ndarray:
    """Track rows (TRACK_DTYPE) of a box whose bottom-centre follows (bottom_x, bottom_y) at the
    times t (seconds, multiples of 1 / FPS). Scalars stand still."""
    t = np.asarray(t, dtype=np.float64)
    x = np.broadcast_to(np.asarray(bottom_x, dtype=np.float64), t.shape)
    y = np.broadcast_to(np.asarray(bottom_y, dtype=np.float64), t.shape)
    rows = np.zeros(len(t), dtype=TRACK_DTYPE)
    rows["frame"] = np.rint(t * FPS).astype(np.int32)
    rows["track_id"] = track_id
    for prefix in ("", "det_"):
        rows[f"{prefix}x0"], rows[f"{prefix}x1"] = x - width / 2, x + width / 2
        rows[f"{prefix}y0"], rows[f"{prefix}y1"] = y - height, y
    rows["score"] = score
    rows["class_id"] = class_id
    rows["confirmed"] = True
    rows["confirmed"][0] = False
    return rows


def result_of(*tracks: np.ndarray, seconds: float = 10.0) -> PerceptionResult:
    """A perception result holding these tracks, for a video `seconds` long at FPS, stride 1."""
    n_frames = int(round(seconds * FPS))
    info = VideoInfo(name="street.mp4", fps=FPS, n_frames=n_frames, width=1000, height=600)
    table = np.concatenate(tracks) if tracks else np.zeros(0, dtype=TRACK_DTYPE)
    return PerceptionResult(
        info, stride=1, detections=np.zeros(0, dtype=DETECTION_DTYPE), tracks=table,
        seconds=0.0, n_analysed=n_frames, complete=True,
    )


def params_with(**sections: dict) -> dict:
    """configs/params.yaml, with some values of some sections replaced."""
    params = copy.deepcopy(load_params())
    for section, values in sections.items():
        for key, value in values.items():
            if isinstance(value, dict):
                params[section][key].update(value)
            else:
                params[section][key] = value
    return params


def times(start: float, end: float) -> np.ndarray:
    """Update times from start to end inclusive, one every 1 / FPS seconds."""
    return np.arange(round(start * FPS), round(end * FPS) + 1) / FPS
