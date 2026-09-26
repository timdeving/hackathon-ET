"""The fast scoring loop: rules on a cached result, scored by the organizers' evaluate.py."""
from __future__ import annotations

import json

from src.perception.cache import save_result
from tests.synthetic_tracks import PERSON, result_of, street_scene, times, track


def make_inputs(tmp_path, labelled: list[list]) -> list[str]:
    """A cache with one pedestrian crossing the street away from the zebra (on the road from 1 s
    to 7 s), the street's scene map, and dev labels; returns the tool's path arguments."""
    t = times(0, 8)
    walker = track(1, PERSON, t, 900, 550 - 50 * t, width=20, height=60)
    save_result(result_of(walker), tmp_path / "cache", {"test": 1})
    (tmp_path / "scene_map.json").write_text(json.dumps(street_scene()))
    labels = {"street.mp4": {"duration": 10.0, "fps": 10.0, "events": labelled}}
    (tmp_path / "labels.json").write_text(json.dumps(labels))
    return ["--labels", tmp_path / "labels.json", "--cache", tmp_path / "cache",
            "--scene", tmp_path / "scene_map.json", "--log", tmp_path / "log.md"]


def test_a_rule_that_matches_the_labels_scores_one(tmp_path, run_python):
    paths = make_inputs(tmp_path, [[1.0, 7.0, "jaywalking"]])
    result = run_python("-m", "tools.eval_from_cache", *paths, "--rules", "jaywalking",
                        "--set", "rules.jaywalking.edge_margin_px=0", "--timeline", "jaywalking")
    assert result.returncode == 0, result.stderr
    row = next(line for line in result.stdout.splitlines() if line.startswith("jaywalking"))
    assert row.split()[1:5] == ["1.000"] * 4
    # Rows exactly on the road's edge: the half-update margin of runs() shows up as 0.05 s.
    assert "predicted 0.95-7.05, tIoU 0.98" in result.stdout
    log = (tmp_path / "log.md").read_text()
    assert "jaywalking 1.000" in log and "`rules.jaywalking.edge_margin_px=0`" in log


def test_misses_and_false_alarms_are_counted(tmp_path, run_python):
    paths = make_inputs(tmp_path, [[20.0, 25.0, "jaywalking"]])  # labelled elsewhere
    result = run_python("-m", "tools.eval_from_cache", *paths, "--rules", "jaywalking",
                        "--timeline", "jaywalking", "--no-log")
    assert result.returncode == 0, result.stderr
    row = next(line for line in result.stdout.splitlines() if line.startswith("jaywalking"))
    assert row.split()[-1] == "0/1/1"
    assert "no label there (tracks 1)" in result.stdout
    assert not (tmp_path / "log.md").exists()


def test_an_unknown_parameter_is_refused(tmp_path, run_python):
    paths = make_inputs(tmp_path, [])
    result = run_python("-m", "tools.eval_from_cache", *paths,
                        "--set", "rules.jaywalking.min_secs=2")
    assert result.returncode != 0
    assert "no parameter 'rules.jaywalking.min_secs'" in result.stderr
