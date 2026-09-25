"""VideoReader: one decoding pass, frames numbered like the harness, correct geometry."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.video.reader import FrameGeometry, VideoReader


def test_frames_are_numbered_and_timed_like_the_harness(tiny_video):
    frames = list(VideoReader(tiny_video.path))
    expected = range(tiny_video.n_frames)
    assert [frame.index for frame in frames] == list(expected)
    expected_times = [i / tiny_video.fps for i in expected]
    assert [frame.t_sec for frame in frames] == pytest.approx(expected_times)


def test_frames_match_what_the_harness_decodes(tiny_video):
    cap = cv2.VideoCapture(str(tiny_video.path))
    for frame in VideoReader(tiny_video.path):
        ok, harness_frame = cap.read()
        assert ok
        assert np.abs(frame.image.astype(np.int16) - harness_frame).mean() < 3
    assert not cap.read()[0]  # and the harness sees no extra frames
    cap.release()


def test_stride_keeps_every_nth_frame(tiny_video):
    indices = [frame.index for frame in VideoReader(tiny_video.path, stride=3)]
    assert indices == list(range(0, tiny_video.n_frames, 3))


def test_working_image_is_cropped_then_shrunk(tiny_video):
    reader = VideoReader(tiny_video.path, width=16, crop=(16, 8, 48, 40))
    assert next(iter(reader)).image.shape == (16, 16, 3)
    assert reader.geometry == FrameGeometry(crop=(16, 8, 48, 40), scale_x=0.5, scale_y=0.5)


def test_geometry_maps_boxes_back_to_full_resolution():
    geometry = FrameGeometry(crop=(100, 50, 1380, 770), scale_x=0.5, scale_y=0.5)
    boxes = np.array([[0, 0, 640, 360], [10, 20, 30, 40]])
    np.testing.assert_allclose(
        geometry.to_full_res(boxes), [[100, 50, 1380, 770], [120, 90, 160, 130]]
    )


def test_rois_are_full_resolution_copies(tiny_video):
    full = next(iter(VideoReader(tiny_video.path))).image
    reader = VideoReader(tiny_video.path, width=16, rois={"light": (2, 3, 12, 9)})
    light = next(iter(reader)).rois["light"]
    np.testing.assert_array_equal(light, full[3:9, 2:12])
    assert light.flags.owndata  # a copy, so it doesn't keep the whole 4K frame alive


@pytest.mark.parametrize(
    "settings",
    [{"stride": 0}, {"crop": (0, 0, 65, 48)}, {"rois": {"light": (5, 5, 5, 9)}}],
)
def test_invalid_settings_are_rejected(tiny_video, settings):
    with pytest.raises(ValueError):
        VideoReader(tiny_video.path, **settings)
