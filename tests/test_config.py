"""Settings: params.yaml alone by default; a profile named by WIUT_PROFILE changes some values."""
from __future__ import annotations

import pytest
import yaml

from src import config


def params_yaml() -> dict:
    with open(config.CONFIG_DIR / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def fresh(monkeypatch):
    """load_params() reads the settings once per process: each test starts and ends afresh."""
    config.load_params.cache_clear()
    yield monkeypatch
    config.load_params.cache_clear()


def test_without_a_profile_the_settings_are_params_yaml(fresh):
    fresh.delenv(config.PROFILE_VARIABLE, raising=False)
    assert config.load_params() == params_yaml()
    assert config.load_params()["detector"]["device"] == "cuda"  # the judged run: the GPU


def test_the_demo_profile_runs_on_the_cpu_without_the_time_limit(fresh):
    fresh.setenv(config.PROFILE_VARIABLE, "demo")
    params, base = config.load_params(), params_yaml()
    assert params["detector"]["device"] == "cpu"
    budget = params["budget"]
    part_b_share = budget["part_b_extra_ratio"]
    part_a_share = budget["time_factor"] * (1 - budget["margin_ratio"]) - part_b_share
    assert part_a_share > 100 and part_b_share > 10  # neither limit binds on a CPU
    # everything else as in params.yaml
    for section in base:
        if section not in ("detector", "budget"):
            assert params[section] == base[section], section
    assert {k: v for k, v in params["detector"].items() if k != "device"} == {
        k: v for k, v in base["detector"].items() if k != "device"
    }


def test_a_profile_changes_only_what_it_names():
    base = {"detector": {"device": "cuda", "conf": 0.1}, "video": {"stride": 3}}
    merged = config.merge_profile(base, {"detector": {"device": "cpu"}})
    assert merged == {"detector": {"device": "cpu", "conf": 0.1}, "video": {"stride": 3}}
    assert base["detector"]["device"] == "cuda"  # the original is left alone


def test_a_misspelt_setting_in_a_profile_is_refused():
    with pytest.raises(ValueError, match="detector.devise"):
        config.merge_profile({"detector": {"device": "cuda"}}, {"detector": {"devise": "cpu"}})


@pytest.mark.parametrize("name", ["no-such-profile", "../params"])
def test_an_unknown_profile_is_refused(fresh, name):
    fresh.setenv(config.PROFILE_VARIABLE, name)
    with pytest.raises(ValueError, match="no such profile"):
        config.load_params()


def test_every_profile_names_only_existing_settings():
    profiles = sorted(config.PROFILES_DIR.glob("*.yaml"))
    assert profiles, "configs/profiles/ holds at least the demo profile"
    for path in profiles:
        config.merge_profile(params_yaml(), config.load_profile(path.stem))
