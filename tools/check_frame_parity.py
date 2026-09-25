"""Does our decoder number frames exactly like the harness?

The harness timestamps frame i as i / fps, counting the frames cv2.VideoCapture.read() returns.
Our VideoReader counts the frames PyAV decodes. If one of them dropped or added a frame at the
start (these files open with two B-frames that come before the first keyframe), every timestamp
of ours would be shifted. This decodes the first --frames frames both ways and, for each shift
between the two sequences, measures how different the paired images are: the smallest
difference must be at shift 0. What remains at shift 0 is the two libraries converting colours
slightly differently, which is the same for every shift.

    python -m tools.check_frame_parity VIDEO [--frames 90] [--full]

--full also counts every frame with both decoders (one full decode each). Exit code 0 = PASS.
"""
from __future__ import annotations

import argparse
import sys
from itertools import islice

import av
import cv2
import numpy as np

from src.video.probe import probe_video
from src.video.reader import VideoReader

COMPARE_WIDTH = 480  # frames are compared at this width: fast, and still detailed enough
MAX_SHIFT = 3  # frames


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.int16)


def harness_frames(path: str, n: int, width: int) -> list[np.ndarray]:
    """The first n frames as the harness sees them (OpenCV read()), grey and shrunk."""
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            break
        height = round(frame.shape[0] * width / frame.shape[1])
        frames.append(to_gray(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)))
    cap.release()
    return frames


def reader_frames(path: str, n: int, width: int) -> list[np.ndarray]:
    """The first n frames as our VideoReader sees them, grey and shrunk the same way."""
    return [to_gray(frame.image) for frame in islice(VideoReader(path, width=width), n)]


def shift_differences(ours: list, theirs: list, max_shift: int) -> dict[int, float]:
    """Mean absolute pixel difference between ours[i + shift] and theirs[i], for each shift."""
    differences = {}
    for shift in range(-max_shift, max_shift + 1):
        pairs = [
            (ours[i + shift], theirs[i])
            for i in range(len(theirs))
            if 0 <= i + shift < len(ours)
        ]
        if pairs:
            differences[shift] = float(np.mean([np.abs(a - b).mean() for a, b in pairs]))
    return differences


def count_frames(path: str) -> tuple[int, int]:
    """Frames returned by a full pass with OpenCV (as the harness does it) and with PyAV."""
    cap = cv2.VideoCapture(path)
    n_opencv = 0
    while cap.grab():
        n_opencv += 1
    cap.release()
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_type = "AUTO"
        n_pyav = sum(1 for _ in container.decode(stream))
    return n_opencv, n_pyav


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("video")
    parser.add_argument("--frames", type=int, default=90, help="frames to compare (default 90)")
    parser.add_argument("--full", action="store_true", help="also count every frame both ways")
    args = parser.parse_args()

    info = probe_video(args.video)
    width = min(COMPARE_WIDTH, info.width)
    theirs = harness_frames(args.video, args.frames, width)
    ours = reader_frames(args.video, args.frames, width)
    differences = shift_differences(ours, theirs, MAX_SHIFT)
    best = min(differences, key=lambda shift: (differences[shift], abs(shift)))

    print(
        f"{info.name}: first {len(theirs)} frames from the harness's decoder, "
        f"{len(ours)} from ours"
    )
    for shift, difference in differences.items():
        marker = "   <- best" if shift == best else ""
        print(f"  shift {shift:+d}: mean pixel difference {difference:6.2f}{marker}")
    passed = best == 0 and len(ours) == len(theirs)

    if args.full:
        n_opencv, n_pyav = count_frames(args.video)
        print(
            f"  frame counts: container {info.n_frames}, "
            f"harness (OpenCV) {n_opencv}, ours (PyAV) {n_pyav}"
        )
        passed = passed and n_opencv == n_pyav

    print("PASS" if passed else "FAIL: our frame numbers would not match the harness's timestamps")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
