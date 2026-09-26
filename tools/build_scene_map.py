"""Build configs/scene_map.json from the LabelMe drawing and the rules file.

    python -m tools.build_scene_map [--labelme configs/scene/labelme.json]
        [--rules configs/scene/scene_rules.yaml] [--image PICTURE]
        [--out configs/scene_map.json] [--scene-dir configs/scene]

Checks the drawing and the rules file against each other and lists every problem at once
(exit code 1), so a drawing can be fixed in one pass. When all is well it writes the scene map,
<scene-dir>/overlay.jpg (every shape drawn on the picture, lane flows as arrows, for a second
person to review) and <scene-dir>/reference.jpg (the picture 1920 px wide, which camera
alignment compares each video's background with). --image defaults to the picture the LabelMe
file points to.

The drawing's labels are `kind:name`; the kinds and the shape each must be drawn with are in
SHAPES below. The rules file holds what the picture can't show: `reference` (the video drawn
on), `arms` (name: description), `lanes` (name: arm, role in/out, signal, exits), `lights`
(name: controls, layout, pedestrian) and `ground_points` (name: [x, y] in metres).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.config import CONFIG_DIR
from src.scene.geometry import apply_homography, distance_to_polyline
from src.scene.scene_map import FORMAT, SceneMap

REFERENCE_WIDTH = 1920
NAME = re.compile(r"[a-z0-9_]+")
SHAPES = {  # label kind -> the LabelMe shape types it may be drawn with
    "road": {"polygon"},
    "island": {"polygon"},
    "crossing": {"polygon"},
    "junction": {"polygon"},
    "lane": {"polygon"},
    "flow": {"linestrip", "line"},
    "stop": {"line", "linestrip"},
    "solid": {"linestrip", "line"},
    "light": {"rectangle"},
    "parking": {"polygon"},
    "bus_stop": {"polygon"},
    "no_uturn": {"polygon"},
    "exit": {"polygon"},  # where a road leaves the picture; the name is the road's arm
    "ground": {"point"},
}
FILLS = {  # overlay colours (BGR) of the filled areas, drawn in this order
    "road": (128, 128, 128),
    "island": (0, 110, 0),
    "junction": (0, 165, 255),
    "crossing": (255, 255, 255),
    "parking": (0, 200, 0),
    "bus_stop": (200, 0, 200),
    "no_uturn": (0, 0, 255),
    "exit": (255, 120, 0),
}


def parse_drawing(labelme: dict, problems: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """The drawing as {kind: {name: points}}; anything malformed goes into `problems`."""
    shapes: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for shape in labelme.get("shapes", []):
        label = shape["label"].strip()
        kind, _, name = label.partition(":")
        if label == "junction":
            name = "junction"
        where = f"shape {label!r}"
        if kind not in SHAPES:
            problems.append(f"{where}: unknown kind {kind!r}; the kinds are {', '.join(SHAPES)}")
        elif not NAME.fullmatch(name):
            problems.append(f"{where}: after ':' use only lowercase letters, digits and _")
        elif shape["shape_type"] not in SHAPES[kind]:
            allowed = " or ".join(sorted(SHAPES[kind]))
            problems.append(f"{where}: drawn as a {shape['shape_type']}; a {kind} is a {allowed}")
        elif name in shapes[kind]:
            problems.append(f"{where}: drawn twice")
        else:
            points = np.asarray(shape["points"], dtype=np.float64)
            if kind == "light":  # a rectangle's two corners, in either order
                points = np.array([points.min(axis=0), points.max(axis=0)])
            shapes[kind][name] = points
    return shapes


def check(shapes: dict, rules: dict, problems: list[str]) -> None:
    """Check the drawing and the rules file against each other."""
    arms = set(rules.get("arms") or {})
    lanes = rules.get("lanes") or {}
    lights = rules.get("lights") or {}
    if not shapes["road"]:
        problems.append("no road:<name> polygon is drawn")
    if len(shapes["junction"]) != 1:
        problems.append("draw exactly one junction polygon")

    _compare(set(shapes["lane"]), set(shapes["flow"]), "lane:{} has no flow:{} line",
             "flow:{} has no lane:{} polygon", problems)
    _compare(set(shapes["lane"]), set(lanes), "lane:{} is drawn but missing from the rules file",
             "lane {} is in the rules file but not drawn", problems)
    for name, lane in lanes.items():
        if lane.get("arm") not in arms:
            problems.append(f"lane {name}: arm {lane.get('arm')!r} is not one of the arms")
        if lane.get("role") not in ("in", "out"):
            problems.append(f"lane {name}: role must be in or out, not {lane.get('role')!r}")
        for arm in lane.get("exits") or []:
            if arm not in arms:
                problems.append(f"lane {name}: exit {arm!r} is not one of the arms")
        stop, flow = shapes["stop"].get(lane.get("arm")), shapes["flow"].get(name)
        if lane.get("role") == "in" and stop is not None and flow is not None:
            first, last = distance_to_polyline(flow[[0, -1]], stop)
            if last >= first:
                problems.append(f"flow:{name} points away from the {lane['arm']} stop line: "
                                "draw it from upstream to downstream")
    for arm in shapes["stop"]:
        if arm not in arms:
            problems.append(f"stop:{arm}: {arm!r} is not one of the arms")
    for arm in shapes["exit"]:
        if arm not in arms:
            problems.append(f"exit:{arm}: {arm!r} is not one of the arms")

    _compare(set(shapes["light"]), set(lights), "light:{} is drawn but missing from the rules file",
             "light {} is in the rules file but not drawn", problems)
    for name, light in lights.items():
        for arm in light.get("controls") or []:
            if arm not in arms:
                problems.append(f"light {name}: controls {arm!r}, which is not one of the arms")
        if light.get("layout", "vertical") not in ("vertical", "horizontal"):
            problems.append(f"light {name}: layout must be vertical or horizontal")
        if not isinstance(light.get("pedestrian", False), bool):
            problems.append(f"light {name}: pedestrian must be true or false")

    ground = rules.get("ground_points") or {}
    _compare(set(shapes["ground"]), set(ground), "ground:{} is drawn but has no metres in the "
             "rules file", "ground point {} is in the rules file but not drawn", problems)
    if shapes["ground"] and len(shapes["ground"]) < 4:
        problems.append("draw at least 4 ground points, or none")


def _compare(drawn: set, listed: set, only_drawn: str, only_listed: str, problems: list) -> None:
    problems += [only_drawn.format(n, n) for n in sorted(drawn - listed)]
    problems += [only_listed.format(n, n) for n in sorted(listed - drawn)]


def fit_ground(shapes: dict, rules: dict, problems: list[str]) -> tuple[np.ndarray | None, dict]:
    """The picture-to-metres homography from the ground points, and each point's error (m)."""
    metres = rules.get("ground_points") or {}
    names = sorted(set(shapes["ground"]) & set(metres))
    if len(names) < 4:
        return None, {}
    image = np.array([shapes["ground"][name][0] for name in names])
    ground = np.array([metres[name] for name in names], dtype=np.float64)
    spread = np.linalg.svd(image - image.mean(axis=0), compute_uv=False)
    if spread[1] < 0.05 * spread[0]:
        problems.append("the ground points are nearly in one line: spread them over the road")
        return None, {}
    homography, _ = cv2.findHomography(image, ground, 0)
    residuals = np.linalg.norm(apply_homography(homography, image) - ground, axis=1)
    return homography, dict(zip(names, residuals.round(2).tolist(), strict=True))


def build(shapes: dict, rules: dict, width: int, height: int, homography) -> dict:
    """The scene map, in the format src.scene.scene_map reads."""

    def rounded(points: np.ndarray) -> list:
        return np.round(points, 1).tolist()

    def named(kind: str) -> dict:
        return {name: rounded(points) for name, points in shapes[kind].items()}

    lanes = rules.get("lanes") or {}
    lights = rules.get("lights") or {}
    ground = rules.get("ground_points") or {}
    return {
        "format": FORMAT,
        "reference": {"video": rules.get("reference"), "width": width, "height": height},
        "roads": named("road"),
        "islands": named("island"),
        "crossings": named("crossing"),
        "junction": rounded(shapes["junction"]["junction"]),
        "lanes": {
            name: {
                "arm": lane["arm"],
                "role": lane["role"],
                "signal": bool(lane.get("signal", lane["role"] == "in")),
                "exits": lane.get("exits"),  # None = unknown: rules make no call
                "polygon": rounded(shapes["lane"][name]),
                "flow": rounded(shapes["flow"][name]),
            }
            for name, lane in lanes.items()
        },
        "stop_lines": named("stop"),
        "solid_lines": named("solid"),
        "lights": {
            name: {
                "box": np.rint(shapes["light"][name]).astype(int).ravel().tolist(),
                "controls": list(light.get("controls") or []),
                "layout": light.get("layout", "vertical"),
                "pedestrian": light.get("pedestrian", False),
            }
            for name, light in lights.items()
        },
        "parking": named("parking"),
        "bus_stops": named("bus_stop"),
        "no_uturn": named("no_uturn"),
        "exit_zones": named("exit"),
        "ground_points": {
            name: {"image": rounded(shapes["ground"][name][0]), "ground": list(ground[name])}
            for name in sorted(shapes["ground"])
        },
        "image_to_ground": None if homography is None else homography.tolist(),
    }


def draw_overlay(image: np.ndarray, shapes: dict) -> np.ndarray:
    """Every shape on the picture: areas filled, lanes outlined, flows as arrows, names written."""
    fill = image.copy()
    for kind, colour in FILLS.items():
        for points in shapes[kind].values():
            cv2.fillPoly(fill, [np.rint(points).astype(np.int32)], colour)
    canvas = cv2.addWeighted(image, 0.55, fill, 0.45, 0)
    thick = max(2, image.shape[1] // 960)
    scale = image.shape[1] / 2400

    def outline(points: np.ndarray, colour: tuple, closed: bool) -> None:
        cv2.polylines(canvas, [np.rint(points).astype(np.int32)], closed, colour, thick)

    def write(text: str, at: np.ndarray, colour: tuple) -> None:
        cv2.putText(canvas, text, tuple(np.rint(at).astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, colour, max(1, thick // 2))

    for name, points in shapes["lane"].items():
        outline(points, (255, 128, 0), closed=True)
        write(name, points.mean(axis=0), (255, 255, 0))
    for points in shapes["flow"].values():
        outline(points, (255, 255, 0), closed=False)
        tail, head = np.rint(points[-2:]).astype(int)
        cv2.arrowedLine(canvas, tuple(tail), tuple(head), (255, 255, 0), thick * 2, tipLength=0.3)
    for kind, colour in (("stop", (0, 0, 255)), ("solid", (0, 255, 255))):
        for name, points in shapes[kind].items():
            outline(points, colour, closed=False)
            write(f"{kind}:{name}", points[0], colour)
    for name, (corner, opposite) in shapes["light"].items():
        cv2.rectangle(canvas, tuple(np.rint(corner).astype(int)),
                      tuple(np.rint(opposite).astype(int)), (255, 0, 255), thick)
        write(name, corner, (255, 0, 255))
    for name, points in shapes["exit"].items():
        write(f"exit:{name}", points.mean(axis=0), (255, 255, 255))
    for name, points in shapes["ground"].items():
        cv2.circle(canvas, tuple(np.rint(points[0]).astype(int)), thick * 3, (255, 255, 255), -1)
        write(name, points[0], (255, 255, 255))
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--labelme", type=Path, default=CONFIG_DIR / "scene" / "labelme.json")
    parser.add_argument("--rules", type=Path, default=CONFIG_DIR / "scene" / "scene_rules.yaml")
    parser.add_argument("--image", type=Path, help="the picture drawn on (default: from LabelMe)")
    parser.add_argument("--out", type=Path, default=CONFIG_DIR / "scene_map.json")
    parser.add_argument("--scene-dir", type=Path, default=CONFIG_DIR / "scene")
    args = parser.parse_args()

    labelme = json.loads(args.labelme.read_text())
    rules = yaml.safe_load(args.rules.read_text()) or {}
    image_path = args.image or args.labelme.parent / labelme["imagePath"]
    image = cv2.imread(str(image_path))
    problems: list[str] = []
    if image is None:
        problems.append(f"cannot read the picture {image_path}; pass it with --image")
    elif image.shape[:2] != (labelme["imageHeight"], labelme["imageWidth"]):
        problems.append(f"{image_path} is {image.shape[1]}x{image.shape[0]}, but the drawing was "
                        f"made on a {labelme['imageWidth']}x{labelme['imageHeight']} picture")
    shapes = parse_drawing(labelme, problems)
    check(shapes, rules, problems)
    homography, residuals = fit_ground(shapes, rules, problems)
    if problems:
        for problem in problems:
            print(f"  {problem}")
        print(f"{len(problems)} problem(s): nothing written")
        return 1

    height, width = image.shape[:2]
    scene = build(shapes, rules, width, height, homography)
    SceneMap(scene)  # the rules must be able to load what we write
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scene, indent=1))
    args.scene_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.scene_dir / "overlay.jpg"), draw_overlay(image, shapes),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    small_width = min(REFERENCE_WIDTH, width)
    small = (small_width, round(height * small_width / width))
    reference = cv2.resize(image, small, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(args.scene_dir / "reference.jpg"), reference, [cv2.IMWRITE_JPEG_QUALITY, 90])

    counts = ", ".join(f"{len(shapes[kind])} {kind}" for kind in SHAPES if shapes[kind])
    print(f"wrote {args.out}: {counts}")
    if residuals:
        print(f"ground points, fitting error in metres: {residuals}")
    print(f"review {args.scene_dir / 'overlay.jpg'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
