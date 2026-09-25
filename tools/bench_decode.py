"""How fast does this machine decode our videos? Run it on the GPU PC.

Decoding 4K H.264 on the CPU takes most of the time budget (3x the video's duration, for Part A
and Part B together), so these numbers decide the frame stride and the working-image width.
On the first --seconds of VIDEO it measures:

    harness   OpenCV read() on every frame: what run_submission.py spends on Part B
    decode    PyAV decoding every frame without converting it: the floor for Part A
    reader    our VideoReader at several (stride, width) settings: Part A's real decode cost
    reformat  PyAV converting and shrinking in one call: a possibly faster alternative

"x duration" is processing seconds per second of video.

    python -m tools.bench_decode VIDEO [--seconds 60] [--threads N]

To mimic the judges' 8 CPU cores:  taskset -c 0-7 python -m tools.bench_decode VIDEO
"""
from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable
from itertools import islice

import av
import cv2

from src.config import load_params
from src.video.probe import probe_video
from src.video.reader import VideoReader

BUDGET = 3.0  # run_submission.py's default time factor

# A method decodes the first n frames of a video with a given thread count and returns how many
# frames it decoded.
Method = Callable[[str, int, int], int]


def harness_read(path: str, n: int, threads: int) -> int:
    """OpenCV read() on every frame, as run_submission.run_risk() does (threads: OpenCV's own)."""
    cap = cv2.VideoCapture(path)
    count = 0
    while count < n and cap.read()[0]:
        count += 1
    cap.release()
    return count


def decode_only(path: str, n: int, threads: int) -> int:
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_type = "AUTO"
        stream.codec_context.thread_count = threads
        return sum(1 for _ in islice(container.decode(stream), n))


def reader(stride: int, width: int | None) -> Method:
    def run(path: str, n: int, threads: int) -> int:
        frames = VideoReader(path, stride=stride, width=width, threads=threads)
        last_index = -1
        for frame in islice(frames, -(-n // stride)):  # ceil(n / stride) samples
            last_index = frame.index
        return last_index + 1

    return run


def reformat(stride: int, width: int) -> Method:
    def run(path: str, n: int, threads: int) -> int:
        with av.open(path) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_type = "AUTO"
            stream.codec_context.thread_count = threads
            height = round(stream.height * width / stream.width)
            count = 0
            for index, frame in enumerate(islice(container.decode(stream), n)):
                if index % stride == 0:
                    frame.reformat(
                        width=width, height=height, format="bgr24", interpolation="AREA"
                    ).to_ndarray()
                count += 1
            return count

    return run


def frames_per_second(method: Method, path: str, n: int, threads: int) -> float:
    start = time.perf_counter()
    decoded = method(path, n, threads)
    return decoded / (time.perf_counter() - start)


def reader_name(stride: int, width: int | None) -> str:
    return f"reader: stride {stride}, width {width or 'full'}"


def main() -> None:
    current = load_params()["video"]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("video")
    parser.add_argument("--seconds", type=float, default=60.0, help="length to decode (default 60)")
    parser.add_argument(
        "--threads",
        type=int,
        default=current["decode_threads"],
        help="PyAV decoder threads, 0 = one per CPU (default: params.yaml video.decode_threads)",
    )
    args = parser.parse_args()

    info = probe_video(args.video)
    n = min(info.n_frames, round(args.seconds * info.fps))
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    print(f"{info.name}: {info.width}x{info.height} @ {info.fps:.2f} fps, first {n} frames, "
          f"{cpus} CPUs available\n")

    methods: dict[str, Method] = {
        "harness: OpenCV read(), every frame": harness_read,
        "decode only (PyAV)": decode_only,
    }
    settings = [(1, None), (3, None), (3, 1280), (3, 1920), (2, 1280)]
    for stride, width in [*settings, (current["stride"], current["width"])]:
        methods[reader_name(stride, width)] = reader(stride, width)
    methods["reformat: stride 3, width 1280"] = reformat(3, 1280)

    print("| method | frames/s | x duration |")
    print("| --- | ---: | ---: |")
    speed = {}
    for name, method in methods.items():
        try:
            speed[name] = frames_per_second(method, args.video, n, args.threads)
        except Exception as error:  # report it and keep measuring the other methods
            print(f"| {name} | failed: {error} | |")
            continue
        print(f"| {name} | {speed[name]:.1f} | {info.fps / speed[name]:.2f} |")

    harness = speed.get("harness: OpenCV read(), every frame")
    ours = speed.get(reader_name(current["stride"], current["width"]))
    if harness and ours:
        total = info.fps / harness + info.fps / ours
        print(f"\nDecoding alone, harness + our reader at the current settings (stride "
              f"{current['stride']}, width {current['width']}): {total:.2f}x duration "
              f"of the {BUDGET:.0f}x budget")


if __name__ == "__main__":
    main()
