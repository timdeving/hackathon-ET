"""probe_video() must report the same metadata as the harness, so our timestamps match."""
from __future__ import annotations

import pytest

from run_submission import video_meta
from src.video.probe import probe_video


def test_probe_reports_what_was_written(tiny_video):
    info = probe_video(tiny_video.path)
    assert info.name == tiny_video.path.name
    assert info.fps == pytest.approx(tiny_video.fps)
    assert info.n_frames == tiny_video.n_frames
    assert (info.width, info.height) == tiny_video.size


def test_probe_agrees_with_the_harness(tiny_video):
    info = probe_video(tiny_video.path)
    meta = video_meta(tiny_video.path)
    assert (info.fps, info.n_frames, info.width, info.height) == (
        meta["fps"],
        meta["n_frames"],
        meta["width"],
        meta["height"],
    )
    assert info.duration == meta["duration"]


def test_probe_rejects_a_missing_file(tmp_path):
    with pytest.raises(RuntimeError):
        probe_video(tmp_path / "missing.mp4")
