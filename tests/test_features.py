"""Per-object features from synthetic tracks: speed, stops, gaps, zones, riders, alignment."""
from __future__ import annotations

import numpy as np

from src.features.tracks import NoAlignment, compute_features
from src.scene.scene_map import SceneMap
from tests.synthetic_tracks import (
    CAR,
    MOTORCYCLE,
    PERSON,
    TRUCK,
    params_with,
    result_of,
    street_scene,
    times,
    track,
)

FEATURES = params_with()["features"]


def features_of(*tracks: np.ndarray, mapper=None, scene=None):
    return compute_features(result_of(*tracks), mapper or NoAlignment(), scene, FEATURES)


def test_a_steady_walker_moves_at_the_right_speed_and_never_stops():
    t = times(0, 5)
    walker = track(1, PERSON, t, 100 + 30 * t, 550, width=20, height=60)  # 30 px/s, 60 px tall
    rows = features_of(walker).rows
    middle = rows[10:-10]  # away from the ends, where the smoothing window is cut short
    np.testing.assert_allclose(middle["speed"], 0.5, atol=1e-6)  # 30 / 60 box heights per s
    np.testing.assert_allclose(middle["vx"], 30, atol=1e-6)
    assert middle["moving"].all() and not rows["stationary"].any()
    np.testing.assert_allclose(rows["t"], t)


def test_a_vehicle_is_stationary_only_while_it_stands_still():
    t = times(0, 9)
    x = np.select([t < 3, t < 6], [100 + 100 * t, 400], 400 + 100 * (t - 6))  # moves, stops 3 s
    rows = features_of(track(1, CAR, t, x, 450, width=80, height=50)).rows
    at = {second: rows[np.isclose(rows["t"], second)][0] for second in (1, 4.5, 8)}
    assert not at[1]["stationary"] and at[4.5]["stationary"] and not at[8]["stationary"]


def test_frames_the_tracker_lost_are_filled_in_and_marked():
    t = times(0, 3)
    seen = np.ones(len(t), dtype=bool)
    seen[10:15] = False  # lost for half a second
    car = track(1, CAR, t, 100 + 100 * t, 450, width=80, height=50)[seen]
    rows = features_of(car).rows
    assert len(rows) == len(t)
    assert rows["filled"].tolist() == (~seen).tolist()
    np.testing.assert_allclose(rows["x"][10:15], 100 + 100 * t[10:15], atol=1e-6)


def test_zones_and_the_agreement_with_the_lane_direction():
    t = times(0, 3)
    with_flow = track(1, CAR, t, 100 + 100 * t, 400, width=80, height=50)  # lane west_in_1
    against = track(2, CAR, t, 500 - 100 * t, 465, width=80, height=50)  # lane west_in_2
    features = features_of(with_flow, against, scene=SceneMap(street_scene()))
    names = ["west_in_1", "west_in_2", "west_out_1", "west_out_2"]
    for info, rows in features.tracks_of([CAR]):
        lane = names[int(rows["lane"][15])]
        agreement = rows["agreement"][15]
        if info["track_id"] == 1:
            assert (lane, round(agreement, 3)) == ("west_in_1", 1.0)
        else:
            assert (lane, round(agreement, 3)) == ("west_in_2", -1.0)
        assert rows["on_road"].all() and (rows["crossing"] == -1).all()


def test_riders_and_people_inside_vehicles_are_not_pedestrians():
    t = times(0, 3)
    x = 100 + 100 * t
    motorcycle = track(1, MOTORCYCLE, t, x, 450, width=40, height=40)
    rider = track(2, PERSON, t, x, 440, width=20, height=50)  # sits on it
    car = track(3, CAR, t, x, 300, width=80, height=50)
    driver = track(4, PERSON, t, x, 290, width=20, height=30)  # seen through the windscreen
    walker = track(5, PERSON, t, 400, 500 - 50 * t, width=20, height=60)  # passes the car once
    features = features_of(motorcycle, rider, car, driver, walker)
    kinds = {
        int(info["track_id"]): (bool(info["rider"]), bool(info["occupant"]))
        for info, _ in features.tracks_of([PERSON])
    }
    assert kinds == {2: (True, False), 4: (False, True), 5: (False, False)}
    walkers = [int(info["track_id"]) for info, _ in features.tracks_of([PERSON], True)]
    assert walkers == [5]


def test_the_class_is_voted_by_detection_scores():
    t = times(0, 0.6)
    as_car = track(1, CAR, t[:3], 100, 450, width=80, height=50, score=0.9)
    as_truck = track(1, TRUCK, t[3:], 100, 450, width=80, height=50, score=0.3)
    assert features_of(np.concatenate([as_car, as_truck])).tracks["class_id"].tolist() == [CAR]


class Shift:
    """A camera alignment stand-in: this video is 10 px left of and 5 px above the reference."""

    def to_reference(self, points, frames=None):
        return np.asarray(points, dtype=np.float64) + [10, 5]


def test_every_point_goes_through_the_camera_alignment():
    t = times(0, 1)
    car = track(1, CAR, t, 100 + 100 * t, 450, width=80, height=50)
    plain, shifted = features_of(car).rows, features_of(car, mapper=Shift()).rows
    np.testing.assert_allclose(shifted["x"] - plain["x"], 10)
    np.testing.assert_allclose(shifted["left_y"] - plain["left_y"], 5)
    np.testing.assert_allclose(shifted["speed"], plain["speed"])


def test_causal_features_use_past_frames_only():
    t = times(0, 4)
    x = np.where(t < 2, 100.0, 100 + 100 * (t - 2))  # starts moving at 2 s
    car = track(1, CAR, t, x, 450, width=80, height=50)
    rows = compute_features(result_of(car), NoAlignment(), None, FEATURES, causal=True).rows
    assert (rows["speed"][rows["t"] < 2] == 0).all()  # nothing leaks back from the future
