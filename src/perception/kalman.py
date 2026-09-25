"""Kalman filter for tracking boxes: constant velocity on centre, aspect ratio and height.

The state is x, y (box centre), a (width / height), h (height) and their four velocities, per
tracker update. Noise scales with the box height, so near (big) and far (small) objects are
treated alike. The model and its noise levels are the ones SORT, DeepSORT and ByteTrack use.
"""
from __future__ import annotations

import numpy as np

# Standard deviations of the position and velocity noise, as a fraction of the box height.
STD_POSITION = 1 / 20
STD_VELOCITY = 1 / 160


def xyxy_to_xyah(box: np.ndarray) -> np.ndarray:
    """[x0, y0, x1, y1] -> [centre x, centre y, width / height, height]."""
    x0, y0, x1, y1 = box
    height = max(y1 - y0, 1e-6)
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / height, height])


def xyah_to_xyxy(xyah: np.ndarray) -> np.ndarray:
    """[centre x, centre y, width / height, height] -> [x0, y0, x1, y1]."""
    cx, cy, aspect, height = xyah
    width = aspect * height
    return np.array([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2])


class KalmanFilter:
    def __init__(self) -> None:
        self._motion = np.eye(8)
        self._motion[:4, 4:] = np.eye(4)  # every update: position += velocity
        self._observe = np.eye(4, 8)  # a detection measures the position part only

    def initiate(self, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean and covariance of a new track from its first box (velocity unknown, so zero)."""
        measurement = xyxy_to_xyah(box)
        h = measurement[3]
        std = np.array(
            [2 * STD_POSITION * h, 2 * STD_POSITION * h, 1e-2, 2 * STD_POSITION * h]
            + [10 * STD_VELOCITY * h, 10 * STD_VELOCITY * h, 1e-5, 10 * STD_VELOCITY * h]
        )
        return np.concatenate([measurement, np.zeros(4)]), np.diag(std**2)

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Move the state one update forward."""
        h = mean[3]
        std = np.array(
            [STD_POSITION * h, STD_POSITION * h, 1e-2, STD_POSITION * h]
            + [STD_VELOCITY * h, STD_VELOCITY * h, 1e-5, STD_VELOCITY * h]
        )
        mean = self._motion @ mean
        covariance = self._motion @ covariance @ self._motion.T + np.diag(std**2)
        return mean, covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, box: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Correct the state with a detected box [x0, y0, x1, y1]."""
        h = mean[3]
        std = np.array([STD_POSITION * h, STD_POSITION * h, 1e-1, STD_POSITION * h])
        projected = self._observe @ covariance @ self._observe.T + np.diag(std**2)
        gain = np.linalg.solve(projected, self._observe @ covariance).T  # P H^T S^-1
        innovation = xyxy_to_xyah(box) - self._observe @ mean
        return mean + gain @ innovation, covariance - gain @ projected @ gain.T
