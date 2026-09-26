"""Decode a video once and hand out the frames the pipeline analyses.

Every frame must be decoded (each H.264 frame depends on earlier ones), but only every
`stride`-th frame is converted to an image, then cropped and shrunk into the working image the
detector sees. Small full-resolution crops (e.g. traffic lights) come from the same frames.
Frames are numbered by counting decoded frames from 0, exactly like the harness's OpenCV loop,
so timestamps agree: t_sec = index / fps. tools/check_frame_parity.py checks this on real videos.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import av
import cv2
import numpy as np

from src.video.probe import VideoInfo, probe_video

# A rectangle in full-resolution pixels: x0, y0, x1, y1, with x1 and y1 exclusive.
Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class FrameGeometry:
    """How working-image pixels map back to full-resolution frame pixels."""

    crop: Box  # the part of the full frame that the working image shows
    scale_x: float  # working-image pixels per full-resolution pixel, horizontally
    scale_y: float  # the same, vertically

    def to_full_res(self, boxes: np.ndarray) -> np.ndarray:
        """Map (N, 4) boxes [x0, y0, x1, y1] from working-image to full-resolution pixels."""
        scale = np.array([self.scale_x, self.scale_y, self.scale_x, self.scale_y])
        offset = np.array([self.crop[0], self.crop[1], self.crop[0], self.crop[1]])
        return np.asarray(boxes, dtype=float) / scale + offset


@dataclass(frozen=True)
class SampledFrame:
    index: int  # frame number, counted from 0 the way the harness counts frames
    t_sec: float  # index / fps: the harness's timestamp for this frame
    image: np.ndarray  # working image: BGR uint8, cropped and shrunk
    rois: dict[str, np.ndarray] = field(default_factory=dict)  # full-resolution BGR crops
    grey: np.ndarray | None = None  # the whole frame in grey, grey_width wide, on some frames


class VideoReader:
    """One decoding pass over a video, yielding every `stride`-th frame.

    Args:
        path: the video file.
        stride: analyse frames 0, stride, 2 * stride, ... (every frame is decoded regardless).
        width: width of the working image after cropping; None keeps full resolution.
        crop: the part of the full frame the working image shows; None means all of it.
        rois: named full-resolution rectangles to cut out of every yielded frame.
        threads: decoder threads; 0 lets FFmpeg use one per CPU core.
        grey_samples: this many of the yielded frames, spread evenly over the video, also carry
            the whole frame in grey (SampledFrame.grey): Part A builds the video's empty-road
            background from them for camera alignment, without a second decode. 0 for none.
        grey_width: width of those grey frames, never more than the frame's own.
    """

    def __init__(
        self,
        path: str | Path,
        stride: int = 1,
        width: int | None = None,
        crop: Box | None = None,
        rois: Mapping[str, Box] | None = None,
        threads: int = 0,
        grey_samples: int = 0,
        grey_width: int = 1920,
    ) -> None:
        if stride < 1:
            raise ValueError(f"stride must be at least 1, got {stride}")
        self.path = str(path)
        self.info: VideoInfo = probe_video(path)
        self.stride = stride
        self.threads = threads
        self.crop: Box = crop or (0, 0, self.info.width, self.info.height)
        self.rois = dict(rois or {})
        for name, box in {"crop": self.crop, **self.rois}.items():
            self._check_inside_frame(name, box)

        crop_w, crop_h = self.crop[2] - self.crop[0], self.crop[3] - self.crop[1]
        if width is None:
            self.size = (crop_w, crop_h)  # working image (width, height)
        else:
            self.size = (width, max(1, round(crop_h * width / crop_w)))
        self.geometry = FrameGeometry(self.crop, self.size[0] / crop_w, self.size[1] / crop_h)

        # Every grey_every-th yielded frame carries a grey copy (0: none).
        yielded = -(-self.info.n_frames // stride)
        self.grey_every = max(1, yielded // grey_samples) if grey_samples > 0 else 0
        grey_w = min(grey_width, self.info.width)
        self.grey_size = (grey_w, max(1, round(self.info.height * grey_w / self.info.width)))

    def __iter__(self) -> Iterator[SampledFrame]:
        with av.open(self.path) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_type = "AUTO"
            stream.codec_context.thread_count = self.threads
            for index, frame in enumerate(container.decode(stream)):
                if index % self.stride == 0:
                    yield self._sample(index, frame.to_ndarray(format="bgr24"))

    def _sample(self, index: int, full: np.ndarray) -> SampledFrame:
        x0, y0, x1, y1 = self.crop
        region = full[y0:y1, x0:x1]
        if (region.shape[1], region.shape[0]) == self.size:
            image = np.ascontiguousarray(region)
        else:
            shrinking = self.size[0] < region.shape[1]
            interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
            image = cv2.resize(region, self.size, interpolation=interpolation)
        rois = {name: full[b[1] : b[3], b[0] : b[2]].copy() for name, b in self.rois.items()}
        grey = None
        if self.grey_every and (index // self.stride) % self.grey_every == 0:
            small = cv2.resize(full, self.grey_size, interpolation=cv2.INTER_AREA)
            grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return SampledFrame(index, index / self.info.fps, image, rois, grey)

    def _check_inside_frame(self, name: str, box: Box) -> None:
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= self.info.width and 0 <= y0 < y1 <= self.info.height):
            raise ValueError(
                f"{name} {box} is not inside the {self.info.width}x{self.info.height} frame"
            )
