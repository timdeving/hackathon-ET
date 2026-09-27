"""Repository paths and tunable parameters.

Every threshold lives in configs/params.yaml with a comment saying where its value came from;
code reads it through load_params() instead of hard-coding numbers.

A profile changes some of them for another setting without touching params.yaml: the environment
variable WIUT_PROFILE=<name> merges configs/profiles/<name>.yaml over it. The live demo on a CPU
sets WIUT_PROFILE=demo. Without the variable params.yaml alone applies, as in the judged run.
"""
from __future__ import annotations

import copy
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
PROFILES_DIR = CONFIG_DIR / "profiles"
WEIGHTS_DIR = REPO_ROOT / "weights"
PROFILE_VARIABLE = "WIUT_PROFILE"


@lru_cache(maxsize=1)
def load_params() -> dict[str, Any]:
    """Return configs/params.yaml as a dict, with the profile named by WIUT_PROFILE merged over
    it when that is set. Read once per process; treat it as read-only."""
    with open(CONFIG_DIR / "params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    profile = os.environ.get(PROFILE_VARIABLE, "").strip()
    return merge_profile(params, load_profile(profile)) if profile else params


def load_profile(name: str) -> dict[str, Any]:
    """configs/profiles/<name>.yaml as a dict."""
    path = PROFILES_DIR / f"{name}.yaml"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or not path.is_file():
        available = sorted(p.stem for p in PROFILES_DIR.glob("*.yaml"))
        raise ValueError(f"{PROFILE_VARIABLE}={name!r}: no such profile; profiles: {available}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_profile(
    params: dict[str, Any], profile: dict[str, Any], where: str = ""
) -> dict[str, Any]:
    """A copy of params with the profile's values in place of theirs, section by section.

    Every key in the profile must already exist in params: a misspelt key would otherwise change
    nothing, silently.
    """
    merged = copy.deepcopy(params)
    for key, value in profile.items():
        path = f"{where}{key}"
        if key not in merged:
            raise ValueError(f"profile setting {path!r} is not in configs/params.yaml")
        if isinstance(value, dict) and isinstance(merged[key], dict):
            merged[key] = merge_profile(merged[key], value, f"{path}.")
        else:
            merged[key] = value
    return merged
