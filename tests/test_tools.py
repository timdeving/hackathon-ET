"""The GPU PC tools run end to end on the tiny clip."""
from __future__ import annotations


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
