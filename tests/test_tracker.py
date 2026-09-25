"""ByteTracker on synthetic detections: numpy + scipy only, so these run on laptops."""
from __future__ import annotations

import numpy as np

from src.perception.boxes import Detections
from src.perception.tracker import ByteTracker

# ByteTrack's defaults, written out so tuning configs/params.yaml can't silently change the tests.
PARAMS = {
    "high_score": 0.5,
    "low_score": 0.1,
    "new_track_score": 0.6,
    "match_cost_high": 0.8,
    "match_cost_low": 0.5,
    "match_cost_new": 0.7,
    "duplicate_iou": 0.85,
    "lost_seconds": 1.0,
    "class_groups": {"vehicle": [2, 5, 7], "person": [0]},
}
UPDATES_PER_SECOND = 10.0  # so a track survives 10 updates without a match
CAR, TRUCK, PERSON = 2, 7, 0


def detections(*rows: tuple) -> Detections:
    """Detections from rows (x0, y0, x1, y1, score, class id)."""
    array = np.array(rows, dtype=np.float32).reshape(-1, 6)
    return Detections(array[:, :4], array[:, 4], array[:, 5].astype(np.int64))


def car_at(step: int, score: float = 0.9, class_id: int = CAR, y: float = 100.0) -> tuple:
    """A 40x20 box moving 5 px right per update."""
    x = 100.0 + 5 * step
    return (x, y, x + 40, y + 20, score, class_id)


def new_tracker() -> ByteTracker:
    return ByteTracker(UPDATES_PER_SECOND, PARAMS)


def test_a_steadily_moving_object_keeps_one_id():
    tracker = new_tracker()
    for step in range(20):
        (tracked,) = tracker.update(detections(car_at(step)))
        assert tracked.track_id == 1
        assert tracked.confirmed
        # Within 2 px: the filtered box lags a little while the speed is still being learned.
        np.testing.assert_allclose(tracked.box, car_at(step)[:4], atol=2.0)


def test_two_objects_keep_separate_ids():
    tracker = new_tracker()
    for step in range(10):
        tracked = tracker.update(detections(car_at(step), car_at(step, y=300.0)))
        assert [t.track_id for t in tracked] == [1, 2]
        assert tracked[0].box[1] < tracked[1].box[1]  # track 1 stays the upper car


def test_detection_index_points_back_to_the_input():
    tracker = new_tracker()
    tracker.update(detections(car_at(0), car_at(0, y=300.0)))
    tracked = tracker.update(detections(car_at(1, y=300.0), car_at(1)))  # input order swapped
    assert [(t.track_id, t.detection) for t in tracked] == [(1, 1), (2, 0)]


def test_a_new_track_needs_a_second_match_and_one_frame_blips_vanish():
    tracker = new_tracker()
    tracker.update(detections(car_at(0)))
    blip = (400.0, 50.0, 420.0, 90.0, 0.9, PERSON)
    tracked = tracker.update(detections(car_at(1), blip))
    assert [(t.track_id, t.confirmed) for t in tracked] == [(1, True), (2, False)]
    tracked = tracker.update(detections(car_at(2)))  # the blip is gone
    assert [t.track_id for t in tracked] == [1]
    tracked = tracker.update(detections(car_at(3), blip))  # a real newcomer, seen twice
    tracked = tracker.update(detections(car_at(4), blip))
    assert [(t.track_id, t.confirmed) for t in tracked] == [(1, True), (3, True)]


def test_a_short_occlusion_keeps_the_id():
    tracker = new_tracker()
    for step in range(10):
        tracker.update(detections(car_at(step)))
    for _ in range(5):  # hidden for 5 updates: less than lost_seconds
        assert tracker.update(detections()) == []
    (tracked,) = tracker.update(detections(car_at(15)))  # back where its motion predicts
    assert tracked.track_id == 1


def test_a_long_occlusion_gives_a_new_id():
    tracker = new_tracker()
    for step in range(10):
        tracker.update(detections(car_at(step)))
    for _ in range(15):  # hidden for 15 updates: longer than lost_seconds
        tracker.update(detections())
    (tracked,) = tracker.update(detections(car_at(25)))
    assert tracked.track_id == 2


def test_weak_detections_extend_a_track_but_never_start_one():
    tracker = new_tracker()
    for step in range(5):
        tracker.update(detections(car_at(step)))
    for step in range(5, 8):  # partly hidden: the score drops below high_score
        (tracked,) = tracker.update(detections(car_at(step, score=0.3)))
        assert tracked.track_id == 1
    assert new_tracker().update(detections(car_at(0, score=0.3))) == []


def test_a_relabelled_vehicle_keeps_its_id():
    tracker = new_tracker()
    for step in range(5):
        tracker.update(detections(car_at(step)))
    (tracked,) = tracker.update(detections(car_at(5, class_id=TRUCK)))
    assert (tracked.track_id, tracked.class_id) == (1, TRUCK)


def test_a_pedestrian_never_takes_a_cars_id():
    tracker = new_tracker()
    for step in range(5):
        tracker.update(detections(car_at(step)))
    # The car is hidden and a pedestrian appears exactly where the car is expected.
    (tracked,) = tracker.update(detections(car_at(5, class_id=PERSON)))
    assert tracked.track_id != 1
    assert tracked.class_id == PERSON


def test_trackers_are_independent_and_deterministic():
    sequence = [detections(car_at(step), car_at(step, y=300.0)) for step in range(8)]
    first, second = new_tracker(), new_tracker()
    for frame in sequence:
        a, b = first.update(frame), second.update(frame)
        assert [(t.track_id, t.detection) for t in a] == [(t.track_id, t.detection) for t in b]
        for x, y in zip(a, b, strict=True):
            np.testing.assert_array_equal(x.box, y.box)
