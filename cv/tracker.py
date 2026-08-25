"""
cv/tracker.py — Multi-object tracking across video frames.

Assigns persistent track IDs to detections (from cv.detector.Detection)
across consecutive frames, so "the same object" can be followed over time
(e.g. for the ISS live-feed pipeline: "unknown object near ISS, tracked
for the last N frames" rather than re-detecting it fresh every frame with
no continuity).

Method: IOU (Intersection-over-Union) matching between the current
frame's detections and each active track's last-known box, solved as a
linear assignment problem (Hungarian algorithm via
scipy.optimize.linear_sum_assignment) rather than greedy nearest-match —
this avoids the common greedy-matching failure mode where an early
suboptimal match blocks a better match for a later detection in the same
frame. Tracks that go unmatched for more than `max_age` consecutive
frames are dropped (the object left the frame, or detection missed it
briefly — a few missed frames shouldn't kill the track).

This is the classic "SORT" (Simple Online and Realtime Tracking)
approach, without the Kalman-filter motion prediction step SORT adds —
IOU-only matching is simpler and works well when frame rate is high
relative to object motion (consecutive-frame boxes overlap
substantially), which is the assumption here; a fast-moving object
across low-frame-rate video would need the motion-prediction extension.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from cv.detector import Detection


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """
    Intersection-over-Union of two (x1, y1, x2, y2) boxes.

    Returns:
        IOU in [0, 1]; 0 if the boxes don't overlap at all.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


@dataclass
class Track:
    """A single tracked object's state across frames."""

    track_id: int
    detection: Detection
    age: int = 0            # consecutive frames since last matched (0 = matched this frame)
    hits: int = 1           # total number of frames this track has been matched
    history: list[tuple[float, float]] = field(default_factory=list)  # center positions over time

    def __post_init__(self):
        if not self.history:
            self.history = [self.detection.center]


class MultiObjectTracker:
    """IOU-based multi-object tracker with Hungarian-algorithm frame-to-frame assignment."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5):
        """
        Args:
            iou_threshold: minimum IOU for a detection-to-track match to
                be accepted; pairs below this are treated as non-matches
                even if they're the best available pairing.
            max_age: number of consecutive unmatched frames before a
                track is dropped.
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        """
        Advance the tracker by one frame.

        Args:
            detections: this frame's detections (e.g. from
                SpaceObjectDetector.detect()).

        Returns:
            The current list of active tracks (after this frame's update
            — newly spawned tracks included, dropped ones excluded).
        """
        if not self.tracks:
            for det in detections:
                self.tracks.append(Track(track_id=self._next_id, detection=det))
                self._next_id += 1
            return self.tracks

        if not detections:
            for track in self.tracks:
                track.age += 1
            self.tracks = [t for t in self.tracks if t.age <= self.max_age]
            return self.tracks

        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                cost_matrix[i, j] = 1.0 - iou(track.detection.box_xyxy, det.box_xyxy)

        track_indices, det_indices = linear_sum_assignment(cost_matrix)

        matched_track_idxs = set()
        matched_det_idxs = set()
        for ti, di in zip(track_indices, det_indices):
            if cost_matrix[ti, di] <= 1.0 - self.iou_threshold:
                track = self.tracks[ti]
                track.detection = detections[di]
                track.age = 0
                track.hits += 1
                track.history.append(detections[di].center)
                matched_track_idxs.add(ti)
                matched_det_idxs.add(di)

        for i, track in enumerate(self.tracks):
            if i not in matched_track_idxs:
                track.age += 1

        for j, det in enumerate(detections):
            if j not in matched_det_idxs:
                self.tracks.append(Track(track_id=self._next_id, detection=det))
                self._next_id += 1

        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return self.tracks

    def reset(self) -> None:
        """Clear all tracks and restart ID assignment from 1."""
        self.tracks = []
        self._next_id = 1
