"""Repository paths and tunable parameters.

Every threshold lives in configs/params.yaml with a comment saying where its value came from;
code reads it through load_params() instead of hard-coding numbers.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


@lru_cache(maxsize=1)
def load_params() -> dict[str, Any]:
    """Return configs/params.yaml as a dict, read once per process. Treat it as read-only."""
    with open(CONFIG_DIR / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)
