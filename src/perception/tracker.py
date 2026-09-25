"""Multi-object tracking: ByteTrack (Zhang et al., ECCV 2022) in numpy + scipy.

Follows the algorithm of the MIT-licensed reference implementation
(github.com/ifzhang/ByteTrack), with two changes for this project:

- Detections only match tracks of the same class group. A pedestrian passing in front of a car
  never takes the car's ID, while a vehicle the detector calls "car" in one frame and "truck" in
  the next keeps its ID.
- Track IDs belong to one tracker instance, not to the whole process, so Part A, Part B and
  every video are tracked independently and deterministically.

Each update: predict every track with a Kalman filter; match confident detections to tracked
and lost tracks by IoU weighted by score; match weak detections to the tracks still unmatched
(a partly hidden object's score drops, and this keeps it on its track); give each new track one
more update to be confirmed; start tracks from confident detections nobody claimed; and forget
tracks lost for too long.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.perception.boxes import Detections, box_iou
from src.perception.kalman import KalmanFilter, xyah_to_xyxy

NO_MATCH = 1e6  # matching cost between different class groups: above any threshold


class TrackState(Enum):
    TENTATIVE = "tentative"  # started in the latest update; one more match confirms it
    TRACKED = "tracked"  # confirmed and matched in the latest update
    LOST = "lost"  # confirmed but unmatched lately; can still be recovered
    REMOVED = "removed"


@dataclass(frozen=True)
class TrackedBox:
    """Where one track is in the current update."""

    track_id: int
    box: np.ndarray  # (4,) x0, y0, x1, y1 after the Kalman correction
    score: float  # score of the detection matched in this update
    class_id: int  # COCO class id of that detection
    detection: int  # index of that detection in this update's input
    confirmed: bool  # False only in a track's first update: it may vanish in the next one


@dataclass(eq=False)
class _Track:
    track_id: int
    mean: np.ndarray  # Kalman state
    covariance: np.ndarray
    score: float
    class_id: int
    group: int
    state: TrackState
    start: int  # number of the update that created the track
    last_seen: int  # number of the update with its latest match
    detection: int = -1  # detection matched in the current update, or -1

    @property
    def box(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean[:4])


class ByteTracker:
    """Tracks the detections of one video. Use a new instance per video and per part.

    Args:
        updates_per_second: how often update() is called per second of video (fps / stride);
            turns `lost_seconds` into a number of updates.
        params: the `tracker` section of configs/params.yaml.
    """

    def __init__(self, updates_per_second: float, params: Mapping[str, Any]) -> None:
        self.params = params
        self.max_lost_updates = max(1, round(params["lost_seconds"] * updates_per_second))
        self._group_of = {
            class_id: group
            for group, class_ids in enumerate(params["class_groups"].values())
            for class_id in class_ids
        }
        self._kalman = KalmanFilter()
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._updates = 0

    def update(self, detections: Detections) -> list[TrackedBox]:
        """Match this update's detections to tracks; return the tracks seen in it, by ID."""
        p = self.params
        self._updates += 1
        for track in self._tracks:
            track.detection = -1
            self._predict(track)
        groups = np.array([self._group(c) for c in detections.class_ids], dtype=np.int64)
        scores = detections.scores
        confident = np.flatnonzero(scores >= p["high_score"])
        weak = np.flatnonzero((scores >= p["low_score"]) & (scores < p["high_score"]))
        confirmed = [t for t in self._tracks if t.state in (TrackState.TRACKED, TrackState.LOST)]
        tentative = [t for t in self._tracks if t.state is TrackState.TENTATIVE]

        # 1. Confident detections vs every confirmed track; lost tracks can come back here.
        unmatched, confident = self._match(
            confirmed, detections, groups, confident, p["match_cost_high"], use_scores=True
        )
        # 2. Weak detections vs the tracks that were tracked until now. Weak detections never
        #    start tracks.
        still_tracked = [t for t in unmatched if t.state is TrackState.TRACKED]
        unmatched, _ = self._match(
            still_tracked, detections, groups, weak, p["match_cost_low"], use_scores=False
        )
        for track in unmatched:
            track.state = TrackState.LOST
        # 3. Tracks started in the previous update get one chance to be confirmed.
        unconfirmed, confident = self._match(
            tentative, detections, groups, confident, p["match_cost_new"], use_scores=True
        )
        for track in unconfirmed:
            track.state = TrackState.REMOVED
        # 4. Confident detections nobody claimed start new tracks.
        for index in confident:
            if scores[index] >= p["new_track_score"]:
                self._start_track(detections, groups, index)
        # 5. Forget tracks lost for too long, and lost tracks duplicating a tracked one.
        for track in self._tracks:
            if (
                track.state is TrackState.LOST
                and self._updates - track.last_seen > self.max_lost_updates
            ):
                track.state = TrackState.REMOVED
        self._remove_duplicates()
        self._tracks = [t for t in self._tracks if t.state is not TrackState.REMOVED]

        seen = [t for t in self._tracks if t.detection >= 0]
        return [self._output(t) for t in sorted(seen, key=lambda t: t.track_id)]

    def _group(self, class_id: int) -> int:
        """The class group a COCO class belongs to; classes in no group form their own."""
        return self._group_of.get(int(class_id), -1 - int(class_id))

    def _predict(self, track: _Track) -> None:
        mean = track.mean.copy()
        if track.state is not TrackState.TRACKED:
            mean[7] = 0.0  # as in ByteTrack: an unmatched box keeps its size
        track.mean, track.covariance = self._kalman.predict(mean, track.covariance)

    def _match(
        self,
        tracks: list[_Track],
        detections: Detections,
        groups: np.ndarray,
        candidates: np.ndarray,
        max_cost: float,
        use_scores: bool,
    ) -> tuple[list[_Track], np.ndarray]:
        """Assign candidate detections (indices) to tracks, minimising the total cost.

        Cost is 1 - IoU, or 1 - IoU x detection score when use_scores. Pairs from different
        class groups, or costing more than max_cost, are never matched. Matched tracks are
        updated; returns the unmatched tracks and the unmatched candidate indices.
        """
        if not tracks or len(candidates) == 0:
            return list(tracks), candidates
        similarity = box_iou(np.array([t.box for t in tracks]), detections.boxes[candidates])
        if use_scores:
            similarity = similarity * detections.scores[candidates][None, :]
        cost = 1.0 - similarity
        track_groups = np.array([t.group for t in tracks])
        cost[track_groups[:, None] != groups[candidates][None, :]] = NO_MATCH

        rows, cols = linear_sum_assignment(cost)
        accepted = cost[rows, cols] <= max_cost
        for row, col in zip(rows[accepted], cols[accepted], strict=True):
            self._update_track(tracks[row], detections, int(candidates[col]))
        matched_rows, matched_cols = set(rows[accepted]), set(cols[accepted])
        unmatched_tracks = [t for i, t in enumerate(tracks) if i not in matched_rows]
        unmatched_candidates = np.array(
            [c for j, c in enumerate(candidates) if j not in matched_cols], dtype=np.int64
        )
        return unmatched_tracks, unmatched_candidates

    def _update_track(self, track: _Track, detections: Detections, index: int) -> None:
        track.mean, track.covariance = self._kalman.update(
            track.mean, track.covariance, detections.boxes[index]
        )
        track.score = float(detections.scores[index])
        track.class_id = int(detections.class_ids[index])
        track.state = TrackState.TRACKED
        track.last_seen = self._updates
        track.detection = index

    def _start_track(self, detections: Detections, groups: np.ndarray, index: int) -> None:
        mean, covariance = self._kalman.initiate(detections.boxes[index])
        # In the very first update there is nothing to confirm against, so (as in ByteTrack)
        # its tracks are confirmed straight away.
        state = TrackState.TRACKED if self._updates == 1 else TrackState.TENTATIVE
        self._tracks.append(
            _Track(
                track_id=self._next_id,
                mean=mean,
                covariance=covariance,
                score=float(detections.scores[index]),
                class_id=int(detections.class_ids[index]),
                group=int(groups[index]),
                state=state,
                start=self._updates,
                last_seen=self._updates,
                detection=int(index),
            )
        )
        self._next_id += 1

    def _remove_duplicates(self) -> None:
        """A lost track lying on a tracked track of the same group is the same object: keep
        whichever has been tracked for longer."""
        tracked = [t for t in self._tracks if t.state is TrackState.TRACKED]
        lost = [t for t in self._tracks if t.state is TrackState.LOST]
        if not tracked or not lost:
            return
        overlap = box_iou(np.array([t.box for t in tracked]), np.array([t.box for t in lost]))
        for i, j in zip(*np.nonzero(overlap > self.params["duplicate_iou"]), strict=True):
            a, b = tracked[i], lost[j]
            if a.group != b.group:
                continue
            if a.last_seen - a.start > b.last_seen - b.start:
                b.state = TrackState.REMOVED
            else:
                a.state = TrackState.REMOVED

    @staticmethod
    def _output(track: _Track) -> TrackedBox:
        return TrackedBox(
            track_id=track.track_id,
            box=track.box,
            score=track.score,
            class_id=track.class_id,
            detection=track.detection,
            confirmed=track.state is TrackState.TRACKED,
        )
