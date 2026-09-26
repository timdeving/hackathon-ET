"""Camera alignment on synthetic pictures with a known change of view."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.scene.alignment import (
    Alignment,
    estimate_alignment,
    load_alignment,
    save_alignment,
    view_change,
)
from src.scene.geometry import apply_homography

FULL = (1920, 1080)  # (width, height) of the frames the test pictures show
PARAMS = {
    "match_width": 960,
    "features": 4000,
    "ratio": 0.75,
    "ransac_px": 2.0,
    "ransac_iterations": 5000,
    "min_inliers": 50,
    "max_shift": 0.10,
    "max_rotation_deg": 5.0,
    "max_zoom": 0.10,
}
BACKGROUND_GREY = 90


def scene() -> np.ndarray:
    """A reference picture with enough texture for SIFT: random blocks, discs and lines."""
    rng = np.random.default_rng(0)
    width, height = FULL
    image = np.full((height, width, 3), BACKGROUND_GREY, np.uint8)
    for _ in range(400):
        colour = tuple(int(c) for c in rng.integers(0, 256, 3))
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        size = int(rng.integers(8, 60))
        kind = int(rng.integers(3))
        if kind == 0:
            cv2.rectangle(image, (x, y), (x + size, y + size // 2), colour, -1)
        elif kind == 1:
            cv2.circle(image, (x, y), size // 2, colour, -1)
        else:
            cv2.line(image, (x, y), (x + size * 2, y + size), colour, 3)
    return image


def view_moved(dx: float, dy: float, degrees: float, zoom: float) -> np.ndarray:
    """Homography from a moved camera's pixels to the reference's: rotated and zoomed about the
    centre, then shifted."""
    affine = cv2.getRotationMatrix2D((FULL[0] / 2, FULL[1] / 2), degrees, zoom)
    affine[:, 2] += (dx, dy)
    return np.vstack([affine, [0.0, 0.0, 1.0]])


def as_seen_by(reference: np.ndarray, video_to_reference: np.ndarray) -> np.ndarray:
    """What the moved camera records: the reference picture seen through the inverse change."""
    return cv2.warpPerspective(
        reference,
        np.linalg.inv(video_to_reference),
        FULL,
        borderValue=(BACKGROUND_GREY,) * 3,
    )


def grid() -> np.ndarray:
    xs, ys = np.meshgrid(np.linspace(200, 1720, 9), np.linspace(150, 930, 7))
    return np.column_stack([xs.ravel(), ys.ravel()])


def point_errors(alignment: Alignment, truth: np.ndarray) -> np.ndarray:
    """How far the alignment puts grid points from where the true change of view puts them."""
    points = grid()
    moved = alignment.to_reference(points)
    return np.linalg.norm(moved - apply_homography(truth, points), axis=1)


def test_a_small_change_of_view_is_recovered_in_full_resolution_pixels():
    reference = scene()
    truth = view_moved(dx=40.0, dy=-25.0, degrees=0.8, zoom=1.012)
    alignment = estimate_alignment(as_seen_by(reference, truth), reference, PARAMS, FULL, FULL)

    assert alignment.ok
    assert alignment.inliers >= PARAMS["min_inliers"]
    assert alignment.error_px < 2.0
    # Matching runs at half size, yet points land within 2 full-resolution pixels.
    assert point_errors(alignment, truth).max() < 2.0


def test_pictures_of_other_sizes_still_give_full_resolution_pixels():
    reference = scene()
    truth = view_moved(dx=30.0, dy=10.0, degrees=0.0, zoom=1.0)
    video = cv2.resize(as_seen_by(reference, truth), (1280, 720), interpolation=cv2.INTER_AREA)
    alignment = estimate_alignment(video, reference, PARAMS, FULL, FULL)
    assert point_errors(alignment, truth).max() < 3.0


def test_an_implausible_change_falls_back_to_no_shift():
    reference = scene()
    truth = view_moved(dx=0.3 * FULL[0], dy=0.0, degrees=0.0, zoom=1.0)  # beyond max_shift
    alignment = estimate_alignment(as_seen_by(reference, truth), reference, PARAMS, FULL, FULL)
    assert not alignment.ok
    np.testing.assert_array_equal(alignment.homography, np.eye(3))


def test_a_picture_without_features_falls_back_to_no_shift():
    blank = np.full((FULL[1], FULL[0], 3), BACKGROUND_GREY, np.uint8)
    alignment = estimate_alignment(blank, scene(), PARAMS, FULL, FULL)
    assert not alignment.ok
    np.testing.assert_array_equal(alignment.to_reference(grid()), grid())


def test_view_change_reports_shift_rotation_and_zoom():
    change = view_change(view_moved(dx=40.0, dy=-25.0, degrees=0.0, zoom=1.0), FULL)
    assert (change["centre_dx_px"], change["centre_dy_px"]) == pytest.approx((40.0, -25.0))
    assert change["rotation_deg"] == pytest.approx(0.0, abs=1e-9)
    assert change["zoom"] == pytest.approx(1.0)
    rotated = view_change(view_moved(dx=0.0, dy=0.0, degrees=-1.0, zoom=1.02), FULL)
    assert rotated["rotation_deg"] == pytest.approx(1.0)  # OpenCV's angles turn the other way
    assert rotated["zoom"] == pytest.approx(1.02)


def test_an_alignment_saved_to_json_loads_back(tmp_path):
    alignment = Alignment(view_moved(12.5, -3.0, 0.4, 1.01), ok=True, inliers=321, error_px=0.87654)
    path = tmp_path / "v.mp4" / "alignment.json"
    save_alignment(alignment, path, extra={"video": "v.mp4"})
    loaded = load_alignment(path)
    np.testing.assert_allclose(loaded.homography, alignment.homography)
    assert (loaded.ok, loaded.inliers) == (True, 321)
    assert loaded.error_px == pytest.approx(0.877)  # stored to 3 decimals


def test_the_identity_leaves_points_where_they_are():
    identity = Alignment.identity()
    assert not identity.ok
    np.testing.assert_array_equal(identity.to_reference(grid()), grid())
    assert identity.to_reference(np.zeros((0, 2))).shape == (0, 2)
