"""The perception cache: what is saved loads back unchanged; versions follow the settings."""
from __future__ import annotations

import dataclasses
import json
import subprocess

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


def test_light_data_is_saved_and_loaded_back(tmp_path):
    grid = np.arange(2 * 3 * 4 * 5, dtype=np.uint8).reshape(2, 3, 4, 5)
    result = dataclasses.replace(
        small_result(), lights={"head": grid}, light_windows={"head": (10, 20, 50, 52)}
    )
    loaded = load_result(save_result(result, tmp_path, SETTINGS))
    np.testing.assert_array_equal(loaded.lights["head"], grid)
    assert loaded.light_windows == {"head": (10, 20, 50, 52)}


def test_a_cache_made_before_the_lights_loads_without_them(tmp_path):
    folder = save_result(small_result(), tmp_path, SETTINGS)
    assert not (folder / "lights.npz").exists()
    meta = json.loads((folder / "meta.json").read_text())
    del meta["light_windows"]  # as in the caches made before the lights were read
    (folder / "meta.json").write_text(json.dumps(meta))
    loaded = load_result(folder)
    assert loaded.lights == {} and loaded.light_windows == {}


def test_the_version_follows_the_settings():
    assert settings_version(SETTINGS) == settings_version(dict(reversed(list(SETTINGS.items()))))
    changed = {**SETTINGS, "video": {"stride": 2, "width": 1280}}
    assert settings_version(changed) != settings_version(SETTINGS)


def test_a_result_is_saved_where_git_is_missing(tmp_path, monkeypatch):
    """The Docker image has no git: the cache records the commit as unknown instead of failing."""
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    folder = save_result(small_result(), tmp_path, SETTINGS)
    assert json.loads((folder / "meta.json").read_text())["git_commit"] == "unknown"
