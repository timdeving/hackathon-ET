"""The converter from a LabelMe drawing + rules file to the scene map."""
from __future__ import annotations

import copy
import json

import cv2
import numpy as np
import yaml

from src.scene.geometry import apply_homography
from src.scene.scene_map import SceneMap

RULES = {
    "reference": "tiny.mp4",
    "arms": {"west": "the road on the left"},
    "lanes": {
        "west_in_1": {"arm": "west", "role": "in", "exits": ["west"]},
        "west_out_1": {"arm": "west", "role": "out"},
    },
    "lights": {"west_main": {"controls": ["west"], "layout": "vertical"}},
}


def shape(label: str, shape_type: str, points: list) -> dict:
    """One shape the way LabelMe saves it."""
    return {"label": label, "shape_type": shape_type, "points": points, "group_id": None,
            "flags": {}}


def drawing() -> list[dict]:
    """The layout of tests/test_scene_map.py's small scene, as it would be drawn in LabelMe."""
    return [
        shape("road:main", "polygon", [[0, 20], [200, 20], [200, 80], [0, 80]]),
        shape("island:median", "polygon", [[0, 48], [200, 48], [200, 52], [0, 52]]),
        shape("crossing:cw", "polygon", [[80, 20], [88, 20], [88, 80], [80, 80]]),
        shape("junction", "polygon", [[90, 20], [110, 20], [110, 80], [90, 80]]),
        shape("lane:west_in_1", "polygon", [[0, 52], [80, 52], [80, 80], [0, 80]]),
        shape("flow:west_in_1", "linestrip", [[5, 66], [40, 66], [75, 66]]),
        shape("lane:west_out_1", "polygon", [[0, 20], [80, 20], [80, 48], [0, 48]]),
        shape("flow:west_out_1", "line", [[75, 34], [5, 34]]),
        shape("stop:west", "line", [[80, 52], [80, 80]]),
        shape("light:west_main", "rectangle", [[156, 18], [150, 5]]),  # corners in either order
    ]


def build(folder, shapes: list[dict], rules: dict, run_python):
    """Write the picture, the drawing and the rules into folder, then run the converter."""
    cv2.imwrite(str(folder / "background.png"), np.full((100, 200, 3), 90, np.uint8))
    labelme = {"version": "5.5.0", "shapes": shapes, "imagePath": "background.png",
               "imageData": None, "imageHeight": 100, "imageWidth": 200}
    (folder / "labelme.json").write_text(json.dumps(labelme))
    (folder / "scene_rules.yaml").write_text(yaml.safe_dump(rules))
    return run_python("-m", "tools.build_scene_map", "--labelme", folder / "labelme.json",
                      "--rules", folder / "scene_rules.yaml", "--out", folder / "scene_map.json",
                      "--scene-dir", folder)


def test_a_correct_drawing_becomes_a_scene_map(tmp_path, run_python):
    result = build(tmp_path, drawing(), RULES, run_python)
    assert result.returncode == 0, result.stdout + result.stderr

    scene = SceneMap.load(tmp_path / "scene_map.json")
    assert [lane.name for lane in scene.lanes] == ["west_in_1", "west_out_1"]
    assert scene.lanes[0].signal is True  # incoming lanes default to signal-controlled
    assert (scene.lanes[1].signal, scene.lanes[1].exits) == (False, None)
    assert scene.lights["west_main"].box == (150, 5, 156, 18)
    assert scene.on_road(np.array([[10, 30], [10, 50]])).tolist() == [True, False]
    assert scene.image_to_ground is None
    assert (tmp_path / "overlay.jpg").exists()
    assert cv2.imread(str(tmp_path / "reference.jpg")).shape == (100, 200, 3)  # never enlarged


def test_a_pedestrian_light_is_kept_as_one(tmp_path, run_python):
    rules = copy.deepcopy(RULES)
    rules["lights"]["west_walk"] = {"controls": ["west"], "layout": "vertical", "pedestrian": True}
    shapes = [*drawing(), shape("light:west_walk", "rectangle", [[160, 5], [166, 15]])]
    result = build(tmp_path, shapes, rules, run_python)
    assert result.returncode == 0, result.stdout + result.stderr
    lights = SceneMap.load(tmp_path / "scene_map.json").lights
    assert (lights["west_main"].pedestrian, lights["west_walk"].pedestrian) == (False, True)


def test_every_problem_is_listed_and_nothing_is_written(tmp_path, run_python):
    shapes = drawing()
    shapes[5] = shape("flow:west_in_1", "linestrip", [[75, 66], [5, 66]])  # arrow reversed
    shapes.append(shape("tree:big", "polygon", [[1, 1], [2, 1], [2, 2]]))
    shapes.append(shape("crossing:cw2", "line", [[1, 1], [2, 2]]))
    rules = copy.deepcopy(RULES)
    del rules["lanes"]["west_out_1"]

    result = build(tmp_path, shapes, rules, run_python)

    assert result.returncode == 1
    for expected in (
        "flow:west_in_1 points away from the west stop line",
        "unknown kind 'tree'",
        "a crossing is a polygon",
        "lane:west_out_1 is drawn but missing from the rules file",
    ):
        assert expected in result.stdout
    assert not (tmp_path / "scene_map.json").exists()


def test_ground_points_give_a_picture_to_metres_homography(tmp_path, run_python):
    corners = [[0, 20], [200, 20], [200, 80], [0, 80]]
    shapes = drawing() + [shape(f"ground:g{i}", "point", [c]) for i, c in enumerate(corners)]
    metres = {"g0": [0, 0], "g1": [40, 0], "g2": [40, 12], "g3": [0, 12]}  # 0.2 m per pixel
    result = build(tmp_path, shapes, {**RULES, "ground_points": metres}, run_python)
    assert result.returncode == 0, result.stdout + result.stderr

    scene = SceneMap.load(tmp_path / "scene_map.json")
    np.testing.assert_allclose(
        apply_homography(scene.image_to_ground, np.array([[100.0, 50.0]])), [[20, 6]], atol=1e-6
    )
