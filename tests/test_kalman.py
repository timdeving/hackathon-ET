"""The tracker's Kalman filter."""
from __future__ import annotations

import numpy as np

from src.perception.kalman import KalmanFilter, xyah_to_xyxy, xyxy_to_xyah


def test_box_conversions_round_trip():
    box = np.array([10.0, 20.0, 50.0, 40.0])
    np.testing.assert_allclose(xyxy_to_xyah(box), [30, 30, 2, 20])
    np.testing.assert_allclose(xyah_to_xyxy(xyxy_to_xyah(box)), box)


def test_a_steady_motion_is_learned_and_predicted():
    kalman = KalmanFilter()
    mean, covariance = kalman.initiate(np.array([0.0, 0.0, 40.0, 20.0]))
    for step in range(1, 15):  # the box moves 5 px right per update
        box = np.array([5.0 * step, 0.0, 5.0 * step + 40, 20.0])
        mean, covariance = kalman.predict(mean, covariance)
        mean, covariance = kalman.update(mean, covariance, box)
    mean, _ = kalman.predict(mean, covariance)
    np.testing.assert_allclose(xyah_to_xyxy(mean[:4]), [75, 0, 115, 20], atol=0.5)
