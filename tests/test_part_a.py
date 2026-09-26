"""Part A's rules step: what runs, when the scene map is missing, and when a rule fails."""
from __future__ import annotations

import json
import logging

import pytest

import src.rules
from src import part_a
from tests.synthetic_tracks import CAR, PERSON, params_with, result_of, street_scene, times, track


@pytest.fixture()
def street_map(tmp_path, monkeypatch):
    """Point Part A at the synthetic street's scene map, loaded afresh."""
    path = tmp_path / "scene_map.json"
    path.write_text(json.dumps(street_scene()))
    monkeypatch.setattr(part_a, "SCENE_MAP_PATH", path)
    monkeypatch.setattr(part_a, "_scene_map", None)
    return path


def jaywalker_and_wrong_way_car():
    t = times(0, 8)
    walker = track(1, PERSON, t, 900, 550 - 50 * t, width=20, height=60)
    car = track(2, CAR, t, 900 - 100 * t, 400, width=80, height=50)  # west in an east lane
    return result_of(walker, car)


def labels_found(segments) -> list[str]:
    return sorted({segment.label for segment in segments})


def test_no_class_enabled_means_no_rules_run(street_map):
    assert part_a._find_events(jaywalker_and_wrong_way_car(), params_with()) == []
    assert part_a._scene_map is None  # not even loaded


def test_enabled_classes_are_found(street_map):
    params = params_with(rules={"enabled": ["jaywalking", "wrong_way"]})
    segments = part_a._find_events(jaywalker_and_wrong_way_car(), params)
    assert labels_found(segments) == ["jaywalking", "wrong_way"]


def test_without_a_scene_map_there_are_no_events_and_an_error_says_why(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(part_a, "SCENE_MAP_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(part_a, "_scene_map", None)
    params = params_with(rules={"enabled": ["jaywalking"]})
    with caplog.at_level(logging.ERROR):
        assert part_a._find_events(jaywalker_and_wrong_way_car(), params) == []
    assert "no scene map" in caplog.text


def test_a_failing_rule_costs_only_its_own_class(street_map, monkeypatch, caplog):
    def broken(context):
        raise RuntimeError("bug")

    monkeypatch.setitem(src.rules.RULES, "wrong_way", broken)
    params = params_with(rules={"enabled": ["jaywalking", "wrong_way"]})
    with caplog.at_level(logging.ERROR):
        segments = part_a._find_events(jaywalker_and_wrong_way_car(), params)
    assert labels_found(segments) == ["jaywalking"]
    assert "the wrong_way rule failed" in caplog.text
