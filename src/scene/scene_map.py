"""The scene map: the road layout of the reference picture, and fast questions about it.

Loads configs/scene_map.json (written by tools/build_scene_map.py) and answers what the rules
ask about a point: is it on the road, in which lane, in which crossing, in the junction. Every
zone is drawn once into a mask the size of the reference picture, so asking about a million
points is one array lookup. All coordinates are pixels of the reference picture; other videos
are mapped onto it by camera alignment first.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.config import CONFIG_DIR
from src.scene.geometry import direction_along

FORMAT = 1  # the scene_map.json layout this code reads


@dataclass(frozen=True)
class Lane:
    name: str
    arm: str  # the road it belongs to
    role: str  # "in": towards the junction; "out": leaving it
    signal: bool  # governed by the light at its arm's stop line
    exits: tuple[str, ...] | None  # arms it may leave the junction by; None = unknown
    polygon: np.ndarray
    flow: np.ndarray  # polyline in the direction traffic moves, first point upstream


@dataclass(frozen=True)
class Light:
    name: str
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 around the lamps
    controls: tuple[str, ...]  # arms whose stop line it governs
    layout: str  # "vertical" (red on top) or "horizontal"


def _points(value: list) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1, 2)


class SceneMap:
    def __init__(self, data: dict) -> None:
        if data.get("format") != FORMAT:
            raise ValueError(f"scene map format {data.get('format')!r}; this code reads {FORMAT}")
        self.video = data["reference"]["video"]
        self.width, self.height = data["reference"]["width"], data["reference"]["height"]
        self.lanes = [
            Lane(
                name=name,
                arm=lane["arm"],
                role=lane["role"],
                signal=lane["signal"],
                exits=None if lane["exits"] is None else tuple(lane["exits"]),
                polygon=_points(lane["polygon"]),
                flow=_points(lane["flow"]),
            )
            for name, lane in data["lanes"].items()
        ]
        self.crossings = list(data["crossings"])
        self.stop_lines = {arm: _points(line) for arm, line in data["stop_lines"].items()}
        self.solid_lines = {name: _points(line) for name, line in data["solid_lines"].items()}
        self.lights = {
            name: Light(name, tuple(light["box"]), tuple(light["controls"]), light["layout"])
            for name, light in data["lights"].items()
        }
        ground = data["image_to_ground"]
        self.image_to_ground = None if ground is None else np.asarray(ground, dtype=np.float64)

        self._road = self._mask(data["roads"].values())
        self._road[self._mask(data["islands"].values())] = False
        self._junction = self._mask([data["junction"]] if data["junction"] else [])
        self._parking = self._mask(data["parking"].values())
        self._bus_stops = self._mask(data["bus_stops"].values())
        self._no_uturn = self._mask(data["no_uturn"].values())
        self._lane_ids = self._index_mask([lane.polygon for lane in self.lanes])
        self._crossing_ids = self._index_mask([_points(p) for p in data["crossings"].values()])

    @classmethod
    def load(cls, path: str | Path = CONFIG_DIR / "scene_map.json") -> SceneMap:
        return cls(json.loads(Path(path).read_text()))

    def on_road(self, points: np.ndarray) -> np.ndarray:
        """On the carriageway: inside a road polygon and not on an island."""
        return self._lookup(self._road, points)

    def in_junction(self, points: np.ndarray) -> np.ndarray:
        return self._lookup(self._junction, points)

    def in_parking(self, points: np.ndarray) -> np.ndarray:
        return self._lookup(self._parking, points)

    def at_bus_stop(self, points: np.ndarray) -> np.ndarray:
        return self._lookup(self._bus_stops, points)

    def in_no_uturn_zone(self, points: np.ndarray) -> np.ndarray:
        return self._lookup(self._no_uturn, points)

    def lane_index(self, points: np.ndarray) -> np.ndarray:
        """Index into self.lanes of the lane each point is in, or -1."""
        return self._lookup(self._lane_ids, points).astype(np.int64) - 1

    def crossing_index(self, points: np.ndarray) -> np.ndarray:
        """Index into self.crossings of the crossing each point is in, or -1."""
        return self._lookup(self._crossing_ids, points).astype(np.int64) - 1

    def flow_direction(self, lane_indices: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Unit direction traffic should move in, at each point of its lane; zero outside lanes."""
        points = _points(points)
        directions = np.zeros((len(points), 2))
        for index in np.unique(lane_indices[lane_indices >= 0]):
            here = lane_indices == index
            directions[here] = direction_along(self.lanes[index].flow, points[here])
        return directions

    def _mask(self, polygons) -> np.ndarray:
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        for polygon in polygons:
            cv2.fillPoly(mask, [np.rint(_points(polygon)).astype(np.int32)], 1)
        return mask.astype(bool)

    def _index_mask(self, polygons: list[np.ndarray]) -> np.ndarray:
        """Each pixel holds 1 + the index of the polygon covering it (0 = none; later wins)."""
        mask = np.zeros((self.height, self.width), dtype=np.int16)
        for index, polygon in enumerate(polygons):
            cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], index + 1)
        return mask

    def _lookup(self, mask: np.ndarray, points: np.ndarray) -> np.ndarray:
        """The mask's value at each point; points outside the picture get 0 / False."""
        xy = np.rint(_points(points)).astype(np.int64)
        x, y = xy[:, 0], xy[:, 1]
        inside = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        values = np.zeros(len(xy), dtype=mask.dtype)
        values[inside] = mask[y[inside], x[inside]]
        return values
