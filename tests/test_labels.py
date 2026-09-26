"""Our label files become ground truth the organizers' scorer accepts."""
from __future__ import annotations

import json

import pytest


def convert(tiny_video, tmp_path, events: list, run_python):
    """Label the tiny clip under an upper-case name, in the predictions format, and convert."""
    labels = {"team": "t", "videos": {tiny_video.path.name.upper(): {"events": events}}}
    (tmp_path / "labels.json").write_text(json.dumps(labels))
    result = run_python("-m", "tools.labels_to_ground_truth", tmp_path / "labels.json",
                        "--videos", tiny_video.path.parent, "--out", tmp_path / "dev_labels.json")
    return result, tmp_path / "dev_labels.json"


def test_labels_become_ground_truth_the_scorer_accepts(tiny_video, tmp_path, run_python):
    events = [[0.5, 1.0, "jaywalking"], [0.8, 1.5, "jaywalking"], [0.2, 0.4, "congestion"]]
    result, out = convert(tiny_video, tmp_path, events, run_python)
    assert result.returncode == 0, result.stdout + result.stderr

    truth = json.loads(out.read_text())
    entry = truth[tiny_video.path.name]  # the real file name, whatever case the labels used
    assert entry["duration"] == pytest.approx(tiny_video.n_frames / tiny_video.fps)
    assert entry["events"] == [[0.2, 0.4, "congestion"], [0.5, 1.5, "jaywalking"]]  # merged

    empty = {"team": "t", "videos": {tiny_video.path.name: {"events": [], "risk": []}}}
    (tmp_path / "empty.json").write_text(json.dumps(empty))
    scored = run_python("evaluate.py", "--pred", tmp_path / "empty.json", "--gt", out)
    assert scored.returncode == 0, scored.stdout + scored.stderr


def test_bad_labels_are_reported_and_nothing_is_written(tiny_video, tmp_path, run_python):
    result, out = convert(tiny_video, tmp_path, [[1.0, 0.5, "speeding"]], run_python)
    assert result.returncode == 1
    assert "'speeding' is not an official class" in result.stdout
    assert "bad times" in result.stdout
    assert not out.exists()
