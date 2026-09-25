"""Part B: a risk score per frame that an accident starts within the next 5 seconds.

The harness calls reset() once per video, then step() for every frame in order, and counts
that time in the same budget as Part A. Decoding every 4K frame for these calls already costs
the harness a large share of that budget, so step() must stay almost free on the frames it
does not analyse. For now it is the starter kit's stub; the causal risk pipeline replaces the
body of step().
"""
from __future__ import annotations

import numpy as np

RISK_HORIZON_SEC = 5.0  # the metric's anticipation horizon H (evaluate.py)


class RiskEstimator:
    """Causal: step() sees frames in order and nothing else; it never opens the video."""

    def reset(self, meta: dict) -> None:
        """Start a new video. meta = {"video_id", "fps", "width", "height", "n_frames"}."""
        self.meta = meta
        self.last_score = 0.0

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        """Return P(accident starts within RISK_HORIZON_SEC) in [0, 1] for this frame.

        frame is BGR uint8 with shape (H, W, 3); t_sec is its timestamp in seconds.
        """
        return self.last_score
