"""solution.py keeps the starter kit's interface, and the unchanged harness accepts its output."""
from __future__ import annotations

import importlib.util
import json

import numpy as np

import solution
from evaluate import OFFICIAL_CLASSES
from src.config import WEIGHTS_DIR, load_params


def perception_available() -> bool:
    """True where the detector can run: PyTorch, a GPU and the weights file (the GPU PC)."""
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    weights = WEIGHTS_DIR / load_params()["detector"]["weights"]
    return torch.cuda.is_available() and weights.exists()


def test_solution_exposes_the_starter_kit_interface():
    assert set(solution.CLASSES) <= set(OFFICIAL_CLASSES)
    assert len(set(solution.CLASSES)) == len(solution.CLASSES)
    estimator = solution.RiskEstimator()
    estimator.reset({"video_id": "v.mp4", "fps": 25.0, "width": 64, "height": 48, "n_frames": 1})
    assert 0.0 <= estimator.step(np.zeros((48, 64, 3), np.uint8), 0.0) <= 1.0


def test_harness_runs_the_solution_and_the_output_validates(tiny_video, tmp_path, run_python):
    out = tmp_path / "predictions.json"
    folder = str(tiny_video.path.parent)
    run = run_python("run_submission.py", "--videos", folder, "--out", str(out))
    assert run.returncode == 0, run.stdout + run.stderr

    pred = json.loads(out.read_text())
    name = tiny_video.path.name
    assert len(pred["videos"][name]["risk"]) == tiny_video.n_frames  # one score per frame
    errors = pred["log"][name]["errors"]
    if perception_available():
        assert errors == []
    else:  # laptops and CI: the harness still finishes, and its log says why Part A found nothing
        assert len(errors) == 1
        assert "perception models unavailable" in errors[0]

    check = run_python("evaluate.py", "--pred", str(out), "--validate-only")
    assert check.returncode == 0, check.stdout + check.stderr
