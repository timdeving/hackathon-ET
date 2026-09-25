"""Traffic-event detection for the WIUT Hackathon 2026 computer-vision track.

Package map:
    part_a.py     Part A entry point: detect_events()
    video/        video metadata identical to the harness's; decoding and frame sampling
    perception/   object detection (exported YOLO) and multi-object tracking (ByteTrack)
    scene/        scene map, camera alignment, image-to-ground homography, signal state
    features/     per-track features in ground-plane units: speed, heading, lane, zone
    rules/        one rule per event class: tracks and scene features in, segments out
    postprocess/  from raw rule output to an event list the harness accepts in full
    risk/         Part B: the causal accident-risk estimator
"""
