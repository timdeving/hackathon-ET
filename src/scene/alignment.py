"""Camera alignment: where a video's pixels sit in the reference picture the scene map is drawn on.

The camera was re-aimed slightly between recordings: the sample views differ by 1 to 139 px at 4K,
with up to 1.1° of rotation and 1.4% of zoom. The scene map is drawn once, on C3896, so every
video gets a homography from its own full-resolution pixels to the reference picture's. It is
estimated from the video's empty-road background: SIFT features matched to the reference
picture's, then a RANSAC homography. Backgrounds hold no moving traffic, so the matches land on
road markings, kerbs, poles and buildings. An implausible estimate falls back to "no shift" and
says so, so a failed alignment never throws the scene map far off.

Points are transformed, never frames: the rules map track positions with to_reference(), which
costs next to nothing.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.scene.geometry import apply_homography

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Alignment:
    """Maps one video's full-resolution pixels onto the reference picture's."""

    homography: np.ndarray  # 3x3: video pixels -> reference pixels
    ok: bool  # False: no trustworthy estimate, so this is the identity
    inliers: int  # feature matches that agree with the homography
    error_px: float  # median distance of those matches from it, in reference pixels

    def to_reference(self, points: np.ndarray, frames: np.ndarray | None = None) -> np.ndarray:
        """(N, 2) video pixels -> (N, 2) reference pixels.

        `frames` (N,) lets a per-frame correction come later without changing any caller; it is
        ignored while there is one homography per video.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return apply_homography(self.homography, points)

    @classmethod
    def identity(cls, inliers: int = 0, error_px: float = 0.0) -> Alignment:
        """No shift, marked not ok: what a video gets when there is no trustworthy estimate."""
        return cls(np.eye(3), ok=False, inliers=inliers, error_px=error_px)


def estimate_alignment(
    background: np.ndarray,
    reference: np.ndarray,
    params: Mapping[str, Any],
    background_size: tuple[int, int],
    reference_size: tuple[int, int],
) -> Alignment:
    """The homography from a video's pixels to the reference picture's, from their backgrounds.

    background, reference: the two empty-road pictures (BGR or grey), at any size; both are
        resized to params["match_width"] for matching.
    background_size, reference_size: (width, height) of the full-resolution frames the two
        pictures show, which are the units of the result.
    params: the `alignment` section of configs/params.yaml.
    """
    small_background, background_scale = _grey_at_width(background, params, background_size)
    small_reference, reference_scale = _grey_at_width(reference, params, reference_size)
    sift = cv2.SIFT_create(nfeatures=params["features"])
    points, descriptors = sift.detectAndCompute(small_background, None)
    reference_points, reference_descriptors = sift.detectAndCompute(small_reference, None)
    if descriptors is None or reference_descriptors is None:
        log.warning("camera alignment: no features found; treating the view as unshifted")
        return Alignment.identity()

    # Lowe's ratio test: keep a match only if it is clearly better than the runner-up, which
    # drops most matches between look-alike patterns such as zebra stripes.
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors, reference_descriptors, k=2)
    matches = [
        p[0] for p in pairs if len(p) == 2 and p[0].distance < params["ratio"] * p[1].distance
    ]
    if len(matches) < params["min_inliers"]:
        log.warning("camera alignment: only %d feature matches; treating the view as unshifted",
                    len(matches))
        return Alignment.identity()

    source = np.array([points[m.queryIdx].pt for m in matches], dtype=np.float64)
    target = np.array([reference_points[m.trainIdx].pt for m in matches], dtype=np.float64)
    # OpenCV's RANSAC draws its samples from a fixed seed, so the result is deterministic.
    small_homography, mask = cv2.findHomography(
        source,
        target,
        cv2.RANSAC,
        params["ransac_px"],
        maxIters=params["ransac_iterations"],
        confidence=0.999,
    )
    if small_homography is None:
        log.warning("camera alignment: no homography fits the matches; treating the view as "
                    "unshifted")
        return Alignment.identity()

    # From matching-size pixels back to full-resolution pixels on both sides.
    homography = (
        np.diag([reference_scale, reference_scale, 1.0])
        @ small_homography
        @ np.diag([1.0 / background_scale, 1.0 / background_scale, 1.0])
    )
    homography /= homography[2, 2]
    inlier = mask.ravel().astype(bool)
    errors = np.linalg.norm(
        apply_homography(homography, source[inlier] * background_scale)
        - target[inlier] * reference_scale,
        axis=1,
    )
    n_inliers, error_px = int(inlier.sum()), float(np.median(errors))

    problem = _implausible(homography, background_size, n_inliers, params)
    if problem:
        log.warning("camera alignment: %s; treating the view as unshifted", problem)
        return Alignment.identity(n_inliers, error_px)
    return Alignment(homography, ok=True, inliers=n_inliers, error_px=error_px)


def view_change(homography: np.ndarray, size: tuple[int, int]) -> dict[str, float]:
    """How far a homography moves a frame of `size` (width, height): the largest corner shift
    and the centre's shift in pixels, the rotation of the top edge in degrees, and the zoom."""
    width, height = size
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
    moved = apply_homography(homography, corners)
    centre = np.array([[width / 2, height / 2]])
    centre_shift = apply_homography(homography, centre)[0] - centre[0]
    top_edge = moved[1] - moved[0]
    diagonal = np.linalg.norm(moved[2] - moved[0]) / np.linalg.norm(corners[2] - corners[0])
    return {
        "max_corner_shift_px": float(np.linalg.norm(moved - corners, axis=1).max()),
        "centre_dx_px": float(centre_shift[0]),
        "centre_dy_px": float(centre_shift[1]),
        "rotation_deg": float(np.degrees(np.arctan2(top_edge[1], top_edge[0]))),
        "zoom": float(diagonal),
    }


def save_alignment(
    alignment: Alignment, path: str | Path, extra: Mapping[str, Any] | None = None
) -> None:
    """Write an alignment as JSON. `extra` (e.g. the video and reference names, the view change)
    is stored alongside for people to read; load_alignment() ignores it."""
    data = {
        "homography": alignment.homography.tolist(),
        "ok": alignment.ok,
        "inliers": alignment.inliers,
        "error_px": round(alignment.error_px, 3),
        **(extra or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))


def load_alignment(path: str | Path) -> Alignment:
    """Read an alignment written by save_alignment() (numpy only: works on laptops)."""
    data = json.loads(Path(path).read_text())
    return Alignment(
        homography=np.asarray(data["homography"], dtype=np.float64),
        ok=bool(data["ok"]),
        inliers=int(data["inliers"]),
        error_px=float(data["error_px"]),
    )


def _grey_at_width(
    image: np.ndarray, params: Mapping[str, Any], full_size: tuple[int, int]
) -> tuple[np.ndarray, float]:
    """The picture in grey at the matching width, and full-resolution pixels per matching pixel."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    width = params["match_width"]
    height = round(grey.shape[0] * width / grey.shape[1])
    shrinking = width < grey.shape[1]
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(grey, (width, height), interpolation=interpolation), full_size[0] / width


def _implausible(
    homography: np.ndarray, size: tuple[int, int], n_inliers: int, params: Mapping[str, Any]
) -> str | None:
    """Why an estimate can't be trusted, or None if it can."""
    if n_inliers < params["min_inliers"]:
        return f"only {n_inliers} matches agree with the homography"
    change = view_change(homography, size)
    if change["max_corner_shift_px"] > params["max_shift"] * size[0]:
        return f"a frame corner would move {change['max_corner_shift_px']:.0f} px"
    if abs(change["rotation_deg"]) > params["max_rotation_deg"]:
        return f"the view would rotate {change['rotation_deg']:.1f} degrees"
    if abs(change["zoom"] - 1.0) > params["max_zoom"]:
        return f"the view would zoom {change['zoom']:.3f}x"
    return None
