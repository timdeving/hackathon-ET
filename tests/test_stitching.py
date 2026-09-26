"""Track stitching on synthetic track tables: numpy + scipy only, so these run on laptops."""
from __future__ import annotations

import numpy as np

from src.perception.pipeline import DETECTION_DTYPE, TRACK_DTYPE, PerceptionResult
from src.perception.stitching import stitch_tracks
from src.video.probe import VideoInfo

# Written out, so tuning configs/params.yaml can't silently change the tests.
PARAMS = {
    "stitching": {
        "max_distance": 0.5,
        "max_gap_sec": 5.0,
        "max_size_change": 2.0,
        "edge_margin": 0.02,
        "velocity_sec": 1.0,
    },
    "tracker": {"class_groups": {"vehicle": [2, 5, 7], "person": [0]}},
}
FPS = 10.0  # one row per frame, so a second is 10 rows
WIDTH, HEIGHT = 1000, 600
CAR, PERSON = 2, 0


def track(track_id: int, frames: range, left: float, speed: float = 20.0,
          class_id: int = CAR, top: float = 300.0) -> np.ndarray:
    """Rows of a 40 x 20 px box moving right at `speed` px/s; `left` is its x at time 0."""
    rows = np.zeros(len(frames), dtype=TRACK_DTYPE)
    rows["frame"] = list(frames)
    rows["track_id"] = track_id
    x = left + speed * rows["frame"] / FPS
    for prefix in ("", "det_"):
        rows[prefix + "x0"], rows[prefix + "x1"] = x, x + 40.0
        rows[prefix + "y0"], rows[prefix + "y1"] = top, top + 20.0
    rows["score"], rows["class_id"], rows["confirmed"] = 0.9, class_id, True
    return rows


def stitched(*tracks: np.ndarray) -> np.ndarray:
    info = VideoInfo(name="v.mp4", fps=FPS, n_frames=300, width=WIDTH, height=HEIGHT)
    result = PerceptionResult(
        info, stride=1, detections=np.zeros(0, dtype=DETECTION_DTYPE),
        tracks=np.concatenate(tracks), seconds=0.0, n_analysed=300, complete=True,
    )
    return stitch_tracks(result, PARAMS).tracks


def ids_by_tracker_id(table: np.ndarray) -> dict[int, int]:
    """The stitched track_id of each of the tracker's IDs."""
    return {int(t): int(s) for t, s in zip(table["tracker_id"], table["track_id"], strict=True)}


def test_an_object_hidden_for_a_few_seconds_gets_one_id():
    # A car behind a bus from 3.0 s to 5.0 s, reappearing where its motion takes it.
    table = stitched(track(1, range(0, 30), left=100), track(2, range(50, 80), left=100))
    assert ids_by_tracker_id(table) == {1: 1, 2: 1}
    assert len(table) == 60  # rows are relabelled, never added or dropped
    assert np.all(np.diff(table["frame"]) > 0)  # one object, one row per frame


def test_an_object_appearing_elsewhere_is_not_joined():
    table = stitched(track(1, range(0, 30), left=100), track(2, range(50, 80), left=300))
    assert ids_by_tracker_id(table) == {1: 1, 2: 2}


def test_a_pedestrian_never_continues_a_car():
    table = stitched(track(1, range(0, 30), left=100),
                     track(2, range(50, 80), left=100, class_id=PERSON))
    assert ids_by_tracker_id(table) == {1: 1, 2: 2}


def test_objects_leaving_the_picture_are_not_joined():
    # The first box reaches the right border; the second appears at the same border.
    table = stitched(track(1, range(0, 30), left=900), track(2, range(35, 45), left=900))
    assert ids_by_tracker_id(table) == {1: 1, 2: 2}


def test_overlapping_tracks_are_never_joined():
    table = stitched(track(1, range(0, 30), left=100), track(2, range(25, 60), left=100))
    assert ids_by_tracker_id(table) == {1: 1, 2: 2}


def test_too_long_a_gap_is_not_joined():
    table = stitched(track(1, range(0, 30), left=100), track(2, range(90, 120), left=100))
    assert ids_by_tracker_id(table) == {1: 1, 2: 2}


def test_a_chain_of_fragments_becomes_one_object():
    fragments = [track(i + 1, range(40 * i, 40 * i + 20), left=100) for i in range(3)]
    table = stitched(*reversed(fragments))  # input order doesn't matter
    assert ids_by_tracker_id(table) == {1: 1, 2: 1, 3: 1}


def test_a_track_continues_at_most_one_other():
    # Two candidates reappear on the car's path; the sooner one is its continuation.
    table = stitched(track(1, range(0, 30), left=100), track(2, range(45, 60), left=100),
                     track(3, range(50, 70), left=100))
    assert ids_by_tracker_id(table) == {1: 1, 2: 1, 3: 3}


def test_the_input_is_left_unchanged_and_an_empty_table_passes_through():
    original = track(1, range(0, 30), left=100)
    before = original.copy()
    stitched(original, track(2, range(50, 80), left=100))
    np.testing.assert_array_equal(original, before)
    assert len(stitched(track(1, range(0, 0), left=100))) == 0
