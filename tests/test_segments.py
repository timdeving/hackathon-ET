"""finalize_events() must always return a list the harness keeps in full."""
from __future__ import annotations

import random

from evaluate import validate
from src.events import CLASSES, Segment
from src.postprocess.segments import finalize_events, merge_segments

PARAMS = {"default": {"merge_gap_sec": 1.0, "min_duration_sec": 0.5}, "per_class": {}}


def test_overlaps_are_merged_even_without_a_gap_allowance():
    segments = [Segment(0.0, 5.0, "jaywalking"), Segment(4.0, 8.0, "jaywalking")]
    assert merge_segments(segments, max_gap=0.0) == [Segment(0.0, 8.0, "jaywalking")]


def test_touching_segments_stay_separate_without_a_gap_allowance():
    segments = [Segment(0.0, 5.0, "jaywalking"), Segment(5.0, 8.0, "jaywalking")]
    assert merge_segments(segments, max_gap=0.0) == segments


def test_short_gaps_are_bridged_and_long_gaps_kept():
    segments = [
        Segment(0.0, 2.0, "congestion"),
        Segment(2.5, 4.0, "congestion"),
        Segment(6.0, 9.0, "congestion"),
    ]
    assert finalize_events(segments, 60.0, PARAMS) == [
        [0.0, 4.0, "congestion"],
        [6.0, 9.0, "congestion"],
    ]


def test_classes_are_merged_independently():
    segments = [Segment(0.0, 2.0, "near_miss"), Segment(2.2, 4.0, "accident")]
    assert finalize_events(segments, 60.0, PARAMS) == [
        [0.0, 2.0, "near_miss"],
        [2.2, 4.0, "accident"],
    ]


def test_fragments_are_merged_before_short_ones_are_dropped():
    # Three 0.3 s fragments are each too short alone, but together they form one 1.3 s event.
    segments = [Segment(t, t + 0.3, "jaywalking") for t in (10.0, 10.5, 11.0)]
    assert finalize_events(segments, 60.0, PARAMS) == [[10.0, 11.3, "jaywalking"]]


def test_blips_are_dropped():
    assert finalize_events([Segment(3.0, 3.2, "wrong_way")], 60.0, PARAMS) == []


def test_segments_are_clipped_to_the_video():
    segments = [
        Segment(-1.0, 2.0, "congestion"),
        Segment(58.0, 65.0, "stopped_vehicle"),
        Segment(70.0, 80.0, "fire_smoke"),
    ]
    assert finalize_events(segments, 60.0, PARAMS) == [
        [0.0, 2.0, "congestion"],
        [58.0, 60.0, "stopped_vehicle"],
    ]


def test_per_class_parameters_override_the_default():
    params = {**PARAMS, "per_class": {"stopped_vehicle": {"min_duration_sec": 10.0}}}
    segments = [Segment(0.0, 5.0, "stopped_vehicle"), Segment(0.0, 5.0, "congestion")]
    assert finalize_events(segments, 60.0, params) == [[0.0, 5.0, "congestion"]]


def test_unknown_labels_are_dropped():
    assert finalize_events([Segment(0.0, 5.0, "speeding")], 60.0, PARAMS) == []


def test_random_rule_output_always_passes_the_official_format_check():
    rng = random.Random(0)
    duration = 120.0
    segments = []
    for _ in range(500):
        start = rng.uniform(-5.0, duration + 5.0)
        segments.append(Segment(start, start + rng.uniform(0.0, 20.0), rng.choice(CLASSES)))

    events = finalize_events(segments, duration, PARAMS)

    assert events
    assert all(0.0 <= start < end <= duration for start, end, _ in events)
    pred = {"team": "test", "videos": {"v.mp4": {"events": events}}}
    gt = {"v.mp4": {"duration": duration, "fps": 25.0, "events": []}}
    errors, _ = validate(pred, gt)
    assert errors == []
