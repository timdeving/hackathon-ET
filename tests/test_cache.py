"""The perception cache: what is saved loads back unchanged; versions follow the settings."""
from __future__ import annotations

import numpy as np

from src.perception.cache import load_result, save_result, settings_version
from src.perception.pipeline import DETECTION_DTYPE, TRACK_DTYPE, PerceptionResult
from src.video.probe import VideoInfo

SETTINGS = {"video": {"stride": 3, "width": 1280}, "weights_sha256": "abc"}


def small_result() -> PerceptionResult:
    detections = np.zeros(2, dtype=DETECTION_DTYPE)
    detections["frame"] = [0, 3]
    detections["score"] = [0.9, 0.8]
    tracks = np.zeros(2, dtype=TRACK_DTYPE)
    tracks["frame"], tracks["track_id"], tracks["confirmed"] = [0, 3], [1, 1], [False, True]
    info = VideoInfo(name="v.mp4", fps=29.97, n_frames=6, width=3840, height=2160)
    return PerceptionResult(
        info, stride=3, detections=detections, tracks=tracks, seconds=1.25, n_analysed=2,
        complete=True,
    )


def test_a_saved_result_loads_back_unchanged(tmp_path):
    folder = save_result(small_result(), tmp_path, SETTINGS)
    assert folder == tmp_path / "v.mp4" / settings_version(SETTINGS)
    loaded = load_result(folder)
    assert (loaded.info, loaded.stride) == (small_result().info, 3)
    assert (loaded.n_analysed, loaded.complete) == (2, True)
    np.testing.assert_array_equal(loaded.detections, small_result().detections)
    np.testing.assert_array_equal(loaded.tracks, small_result().tracks)


def test_the_version_follows_the_settings():
    assert settings_version(SETTINGS) == settings_version(dict(reversed(list(SETTINGS.items()))))
    changed = {**SETTINGS, "video": {"stride": 2, "width": 1280}}
    assert settings_version(changed) != settings_version(SETTINGS)
