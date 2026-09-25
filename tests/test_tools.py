"""The GPU PC tools run end to end on the tiny clip."""
from __future__ import annotations

import copy
import json

RUN = {
    "videos": {"v.mp4": {"events": [[1.0, 2.0, "congestion"]], "risk": [[0.0, 0.1]]}},
    "log": {"v.mp4": {"total_sec": 1.0}},
}


def test_determinism_check_compares_predictions_but_not_timings(tmp_path, run_python):
    same, changed = copy.deepcopy(RUN), copy.deepcopy(RUN)
    same["log"]["v.mp4"]["total_sec"] = 2.0  # only the harness's timing differs
    changed["videos"]["v.mp4"]["events"][0][1] = 2.5
    for name, run in (("a", RUN), ("b", same), ("c", changed)):
        (tmp_path / f"{name}.json").write_text(json.dumps(run))

    check = run_python("-m", "tools.check_determinism", "predictions", tmp_path / "a.json",
                       tmp_path / "b.json")
    assert check.returncode == 0, check.stdout + check.stderr
    check = run_python("-m", "tools.check_determinism", "predictions", tmp_path / "a.json",
                       tmp_path / "c.json")
    assert check.returncode == 1
    assert "events differ" in check.stdout


def test_frame_parity_check_passes_on_a_clean_clip(tiny_video, run_python):
    video = str(tiny_video.path)
    result = run_python("-m", "tools.check_frame_parity", video, "--frames", "30", "--full")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "shift +0" in result.stdout
    assert "PASS" in result.stdout


def test_decode_benchmark_measures_every_method(tiny_video, run_python):
    result = run_python("-m", "tools.bench_decode", str(tiny_video.path), "--seconds", "1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "failed" not in result.stdout
    assert "Decoding alone" in result.stdout
