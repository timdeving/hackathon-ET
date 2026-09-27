"""The website's data: EDA of the sample videos and our results on them, from the perception
caches, the camera alignments, our predictions and the dev labels. Writes JSON for the charts
and pictures (heat maps, trajectories) into the website's folder.

    python -m tools.website_data [--predictions predictions_samples.json] [--out website]

Needs the caches (tools/cache_tracks.py); runs on any machine in under a minute.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

import evaluate
from src.config import CONFIG_DIR, REPO_ROOT, load_params
from src.features.tracks import PERSON, VEHICLES, compute_features
from src.perception.cache import load_result
from src.perception.stitching import stitch_tracks
from src.rules.common import AMBER, GREEN, RED
from src.scene.alignment import load_alignment, view_change
from src.scene.scene_map import SceneMap
from src.scene.signals import arm_phases
from tools.eval_from_cache import cache_folder

BIN_SEC = 5.0  # counts over time, per bin
RISK_STEP_SEC = 0.5  # the risk curve on the page: its highest value in each step
GROUPS = {"people": (PERSON,), "cars": (2,), "buses and trucks": (5, 7), "two-wheelers": (1, 3)}
# What the recordings show (seen in the videos); the brightness is measured.
TIME_OF_DAY = {"C3896": "morning, low sun", "C3897": "morning, low sun",
               "C3902": "afternoon, in shade", "C3905": "evening, dusk"}
PHASE_NAMES = {RED: "red", AMBER: "amber", GREEN: "green"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path, default=REPO_ROOT / "predictions_samples.json")
    parser.add_argument("--labels", type=Path, default=REPO_ROOT / "data" / "dev_labels.json")
    parser.add_argument("--cache", type=Path, default=REPO_ROOT / "cache")
    parser.add_argument("--backgrounds", type=Path,
                        default=REPO_ROOT / "outputs" / "scene_backgrounds")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "website")
    parser.add_argument("--clips", type=Path, metavar="MEDIA",
                        help="only cut the example and failure clips (website/data/clips.json) "
                             "from the annotated videos in MEDIA (tools/render_samples.py)")
    args = parser.parse_args()
    if args.clips:
        cut_clips(json.loads((args.out / "data" / "clips.json").read_text()), args.clips)
        return 0
    (args.out / "data").mkdir(parents=True, exist_ok=True)
    (args.out / "img").mkdir(parents=True, exist_ok=True)

    params, scene = load_params(), SceneMap.load()
    reference = cv2.imread(str(CONFIG_DIR / "scene" / "reference.jpg"))
    predictions = json.loads(args.predictions.read_text())["videos"]
    labels = json.loads(args.labels.read_text())
    eda, results = {"videos": []}, {"videos": []}
    vehicle_points, people_points, paths = [], [], []
    for video in sorted(predictions):
        folder = cache_folder(args.cache, video)
        alignment = load_alignment(args.cache / video / "alignment.json")
        result = stitch_tracks(load_result(folder), params)
        features = compute_features(result, alignment, scene, params["features"])
        eda["videos"].append(describe(video, result, alignment, features, scene, params,
                                      args.backgrounds))
        rows = features.rows
        moving_vehicles = rows[np.isin(rows["track_id"], track_ids(features, VEHICLES))
                               & rows["moving"]]
        vehicle_points.append(np.column_stack([moving_vehicles["x"], moving_vehicles["y"]]))
        people = rows[np.isin(rows["track_id"], track_ids(features, (PERSON,))) & rows["moving"]]
        people_points.append(np.column_stack([people["x"], people["y"]]))
        paths += vehicle_paths(features)
        results["videos"].append(outcome(video, predictions[video], labels.get(video)))

    labelled = {video: labels[video] for video in labels if video in predictions}
    report = evaluate.evaluate_part_a(labelled, {v: predictions[v] for v in labelled})
    results["scores"] = {
        "videos": sorted(labelled),
        "score_a": round(report["score_a"], 4),
        "classes": {c: {tau: round(report["per_class"][c][tau]["f1"], 3)
                        for tau in ("0.3", "0.5", "0.7")}
                    | {"mean": round(report["per_class"][c]["f1_mean"], 3)}
                    for c in report["classes"]},
    }
    (args.out / "data" / "eda.json").write_text(json.dumps(eda))
    (args.out / "data" / "results.json").write_text(json.dumps(results))
    heat_map(reference, np.concatenate(vehicle_points), args.out / "img" / "heat_vehicles.jpg")
    heat_map(reference, np.concatenate(people_points), args.out / "img" / "heat_walking.jpg")
    trajectories(reference, paths, args.out / "img" / "trajectories.jpg")
    print(f"wrote {args.out / 'data'} and {args.out / 'img'}; Score A on the labelled videos "
          f"{report['score_a']:.4f}")
    return 0


def track_ids(features, class_ids) -> np.ndarray:
    return features.tracks["track_id"][np.isin(features.tracks["class_id"], class_ids)]


def describe(video, result, alignment, features, scene, params, backgrounds: Path) -> dict:
    """One video's facts and time series for the EDA section."""
    info, rows = result.info, features.rows
    stem = Path(video).stem
    bins = np.arange(0.0, info.duration + BIN_SEC, BIN_SEC)
    frames_per_bin = np.maximum(np.histogram(np.unique(rows["t"]), bins)[0], 1)
    counts = {}
    for group, class_ids in GROUPS.items():
        mine = rows[np.isin(rows["track_id"], track_ids(features, class_ids))]
        counts[group] = np.round(np.histogram(mine["t"], bins)[0] / frames_per_bin, 2).tolist()
    vehicles = rows[np.isin(rows["track_id"], track_ids(features, VEHICLES))]
    stopped = vehicles[vehicles["stationary"]]
    counts["vehicles standing"] = np.round(
        np.histogram(stopped["t"], bins)[0] / frames_per_bin, 2).tolist()

    lights = []
    phases = arm_phases(result, alignment, scene, params).get("top")
    if phases is not None:
        lights = [[round(start, 2), round(end, 2), PHASE_NAMES.get(code, "unknown")]
                  for start, end, code in phases.spans(info.fps)]
    # A cycle runs from one green to the next; short greens are the flashing green's blinks.
    greens = [start for start, end, name in lights if name == "green" and end - start >= 10]
    background = backgrounds / stem / "background.png"
    brightness = None
    if background.exists():
        brightness = round(float(cv2.imread(str(background), cv2.IMREAD_GRAYSCALE).mean()), 1)
    moved = view_change(alignment.homography, (info.width, info.height))
    return {
        "video": video,
        "width": info.width, "height": info.height, "fps": round(info.fps, 3),
        "duration": round(info.duration, 1), "frames": info.n_frames,
        "time_of_day": TIME_OF_DAY.get(stem, ""), "brightness": brightness,
        "camera_shift_px": round(moved["max_corner_shift_px"], 1),
        "camera_rotation_deg": round(moved["rotation_deg"], 2),
        "tracks": {group: int(len(track_ids(features, ids))) for group, ids in GROUPS.items()},
        "bins_sec": BIN_SEC, "counts": counts, "lights": lights,
        "cycle_sec": round(float(np.median(np.diff(greens))), 1) if len(greens) > 2 else None,
    }


def outcome(video: str, prediction: dict, labels: dict | None) -> dict:
    """One video's events, risk curve (its highest value per step) and labels, for the results
    section."""
    risk = np.array(prediction["risk"], dtype=float).reshape(-1, 2)
    steps = np.floor(risk[:, 0] / RISK_STEP_SEC).astype(int)
    curve = [[round(step * RISK_STEP_SEC, 2), round(float(risk[steps == step, 1].max()), 3)]
             for step in np.unique(steps)]
    return {"video": video, "events": prediction["events"], "risk": curve,
            "labels": labels["events"] if labels else None}


def vehicle_paths(features) -> list[np.ndarray]:
    """Ground paths (reference pixels) of vehicles that travel, for the trajectories picture."""
    paths = []
    for _, rows in features.tracks_of(VEHICLES):
        moving = rows[rows["moving"]]
        if len(moving) < 20:
            continue
        points = np.column_stack([moving["x"], moving["y"]])
        if np.linalg.norm(points[-1] - points[0]) > 300:
            paths.append(points)
    return paths


def heat_map(reference: np.ndarray, points: np.ndarray, path: Path) -> None:
    """Where the points gather, as a heat map over the darkened reference picture."""
    height, width = reference.shape[:2]
    cell = 16
    grid, _, _ = np.histogram2d(points[:, 1], points[:, 0], bins=(height // cell, width // cell),
                                range=((0, height), (0, width)))
    grid = cv2.GaussianBlur(np.log1p(grid).astype(np.float32), (0, 0), 1.5)
    grid = cv2.resize(grid / max(float(grid.max()), 1e-6), (width, height))
    colours = cv2.applyColorMap((grid * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    alpha = np.clip(grid * 1.5, 0, 0.85)[..., None]
    picture = (reference * 0.45 * (1 - alpha) + colours * alpha).astype(np.uint8)
    cv2.imwrite(str(path), cv2.resize(picture, (1600, 900), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 85])


def trajectories(reference: np.ndarray, paths: list[np.ndarray], path: Path) -> None:
    """Vehicle paths over the darkened reference picture, coloured by their direction."""
    picture = (reference * 0.4).astype(np.uint8)
    for points in paths:
        dx, dy = points[-1] - points[0]
        hue = int((np.degrees(np.arctan2(dy, dx)) % 360) / 2)  # OpenCV hues run 0-179
        colour = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        cv2.polylines(picture, [points.astype(np.int32)], False, colour, 3, cv2.LINE_AA)
    cv2.imwrite(str(path), cv2.resize(picture, (1600, 900), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 85])


def cut_clips(clips: dict, media: Path) -> None:
    """Each listed stretch of an annotated video as its own small clip, next to the videos."""
    for kind in ("examples", "failures"):
        (media / kind).mkdir(exist_ok=True)
        for clip in clips[kind]:
            source = media / f"{Path(clip['video']).stem}.mp4"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(clip["start"]),
                            "-i", str(source), "-t", str(clip["end"] - clip["start"]),
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-an",
                            "-movflags", "+faststart", str(media / kind / f"{clip['id']}.mp4")],
                           check=True)
            print(f"{kind}/{clip['id']}.mp4")


if __name__ == "__main__":
    raise SystemExit(main())
