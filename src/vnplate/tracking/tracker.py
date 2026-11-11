from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .association import associate, cxcywh_to_xyxy, xyxy_to_cxcywh
from .kalman import KalmanBoxFilter


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    score: float
    kalman: KalmanBoxFilter | None = None
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    history: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        if not self.history:
            self.history.append(self.bbox.copy())

    def predict(self) -> np.ndarray:
        if self.kalman is None:
            return self.bbox
        return cxcywh_to_xyxy(self.kalman.predict())

    def update(self, bbox: np.ndarray, score: float) -> None:
        self.bbox = np.asarray(bbox, dtype=np.float64)
        self.score = float(score)
        if self.kalman is not None:
            self.kalman.update(xyxy_to_cxcywh(self.bbox))
        self.history.append(self.bbox.copy())
        self.time_since_update = 0
        self.hits += 1

    def is_confirmed(self, min_hits: int) -> bool:
        return self.hits >= min_hits

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox.tolist(),
            "score": self.score,
            "hits": self.hits,
            "age": self.age,
            "time_since_update": self.time_since_update,
        }


class MultiObjectTracker:
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        use_kalman: bool = True,
        byte_track: bool = False,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        low_iou_threshold: float = 0.2,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.use_kalman = use_kalman
        self.byte_track = byte_track
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.new_track_thresh = new_track_thresh
        self.low_iou_threshold = low_iou_threshold
        self.tracks: list[Track] = []
        self._next_id = 1
        self.frame_index = 0

    def _spawn(self, bbox: np.ndarray, score: float) -> Track:
        track = Track(
            track_id=self._next_id,
            bbox=np.asarray(bbox, dtype=np.float64),
            score=score,
            kalman=KalmanBoxFilter(xyxy_to_cxcywh(bbox)) if self.use_kalman else None,
        )
        self._next_id += 1
        return track

    def _update_standard(self, detections: np.ndarray, scores: np.ndarray, predicted: np.ndarray) -> None:
        matches, unmatched_dets, unmatched_trks = associate(detections, predicted, self.iou_threshold)
        for d, t in matches:
            self.tracks[t].update(detections[d], scores[d])
        for t in unmatched_trks:
            self.tracks[t].time_since_update += 1
        for d in unmatched_dets:
            self.tracks.append(self._spawn(detections[d], scores[d]))

    def _update_bytetrack(self, detections: np.ndarray, scores: np.ndarray, predicted: np.ndarray) -> None:
        high_idx = np.flatnonzero(scores >= self.high_thresh)
        low_idx = np.flatnonzero((scores >= self.low_thresh) & (scores < self.high_thresh))

        matches_high, unmatched_high, unmatched_trks = associate(
            detections[high_idx], predicted, self.iou_threshold
        )
        matched_tracks: set[int] = set()
        for d, t in matches_high:
            di = int(high_idx[d])
            self.tracks[t].update(detections[di], scores[di])
            matched_tracks.add(int(t))

        if len(low_idx) and len(unmatched_trks):
            low_dets = detections[low_idx]
            trk_pred = predicted[unmatched_trks]
            matches_low, _, _ = associate(low_dets, trk_pred, self.low_iou_threshold)
            for d, t in matches_low:
                di = int(low_idx[d])
                ti = int(unmatched_trks[t])
                self.tracks[ti].update(detections[di], scores[di])
                matched_tracks.add(ti)

        for t in range(len(self.tracks)):
            if t not in matched_tracks:
                self.tracks[t].time_since_update += 1

        for d in unmatched_high:
            di = int(high_idx[d])
            if scores[di] >= self.new_track_thresh:
                self.tracks.append(self._spawn(detections[di], scores[di]))

    def update(self, detections: np.ndarray, scores: np.ndarray | None = None) -> list[Track]:
        self.frame_index += 1
        detections = np.asarray(detections, dtype=np.float64).reshape(-1, 4)
        if scores is None:
            scores = np.ones(len(detections))
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)

        predicted = np.array([t.predict() for t in self.tracks]) if self.tracks else np.zeros((0, 4))
        if self.byte_track:
            self._update_bytetrack(detections, scores, predicted)
        else:
            self._update_standard(detections, scores, predicted)

        for track in self.tracks:
            track.age += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return [t for t in self.tracks if t.is_confirmed(self.min_hits) and t.time_since_update == 0]

    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.is_confirmed(self.min_hits)]
