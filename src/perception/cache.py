"""Save and load perception results, so rules can be developed without re-running perception.

Layout: cache/<video name>/<settings version>/perception.npz (the detections and tracks tables),
lights.npz (the traffic lights' lit-pixel counts, when perception read them) and meta.json (the
video, the perception settings, the light windows, the git commit and the processing time). The
settings version is a short hash of everything that changes perception's output, the weights
file included, so caches made with different settings sit side by side.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.config import REPO_ROOT
from src.perception.pipeline import PerceptionResult
from src.video.probe import VideoInfo


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def settings_version(settings: dict[str, Any]) -> str:
    """A short, stable name for a set of perception settings."""
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:8]


def save_result(result: PerceptionResult, cache_dir: str | Path, settings: dict[str, Any]) -> Path:
    """Write one video's result under cache_dir; returns the folder it went to."""
    folder = Path(cache_dir) / result.info.name / settings_version(settings)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        folder / "perception.npz", detections=result.detections, tracks=result.tracks
    )
    if result.lights:
        np.savez_compressed(folder / "lights.npz", **result.lights)
    meta = {
        "video": asdict(result.info),
        "stride": result.stride,
        "seconds": round(result.seconds, 1),
        "n_analysed": result.n_analysed,
        "complete": result.complete,
        "light_windows": {name: list(box) for name, box in result.light_windows.items()},
        "settings": settings,
        "git_commit": _git_commit(),
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    return folder


def load_result(folder: str | Path) -> PerceptionResult:
    """Read a result written by save_result() (numpy only: works on laptops). A cache made
    before the traffic lights were read loads without light data."""
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text())
    lights = {}
    if (folder / "lights.npz").exists():
        with np.load(folder / "lights.npz") as grids:
            lights = {name: grids[name] for name in grids.files}
    with np.load(folder / "perception.npz") as tables:
        return PerceptionResult(
            info=VideoInfo(**meta["video"]),
            stride=meta["stride"],
            detections=tables["detections"],
            tracks=tables["tracks"],
            seconds=meta["seconds"],
            n_analysed=meta["n_analysed"],
            complete=meta["complete"],
            lights=lights,
            light_windows={
                name: tuple(box) for name, box in meta.get("light_windows", {}).items()
            },
        )


def _git_commit() -> str:
    """The current commit, marked "-dirty" when tracked files have uncommitted changes; "unknown"
    outside a git checkout, or where git isn't installed (the Docker image)."""
    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
            ).stdout.strip()
        except OSError:
            return ""

    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    return commit + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")
