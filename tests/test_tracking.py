import numpy as np

from vnplate.fusion.temporal import TemporalFusion
from vnplate.tracking.tracker import MultiObjectTracker


def test_tracker_keeps_identity_through_occlusion():
    tracker = MultiObjectTracker(max_age=3, min_hits=1, iou_threshold=0.2)
    boxes = [
        [10, 10, 60, 40],
        [14, 11, 64, 41],
        [18, 12, 68, 42],
        [22, 13, 72, 43],
    ]
    ids = []
    for box in boxes:
        tracks = tracker.update(np.array([box], dtype=np.float64), np.array([0.9]))
        ids.extend(t.track_id for t in tracks)
    assert len(set(ids)) == 1

    for _ in range(2):
        tracker.update(np.zeros((0, 4)), np.zeros((0,)))
    tracks = tracker.update(np.array([[30, 14, 80, 44]], dtype=np.float64), np.array([0.9]))
    assert tracks and tracks[0].track_id == ids[0]


def test_temporal_fusion_majority_vote():
    fusion = TemporalFusion(min_votes=2)
    for text, conf in [("30A12345", 0.75), ("30A12345", 0.89), ("30A12845", 0.61), ("30A12345", 0.95)]:
        fusion.add(1, text, conf)
    result = fusion.fuse(1)
    assert result is not None
    assert result.text == "30A12345"
    assert result.observations == 4
    assert result.confidence > 0.7


def test_tracker_use_kalman_false_disables_kalman():
    tracker = MultiObjectTracker(use_kalman=False)
    tracker.update(np.array([[10, 10, 60, 40]], dtype=np.float64), np.array([0.9]))
    assert tracker.tracks[0].kalman is None


def test_tracker_use_kalman_true_creates_kalman():
    tracker = MultiObjectTracker(use_kalman=True)
    tracker.update(np.array([[10, 10, 60, 40]], dtype=np.float64), np.array([0.9]))
    assert tracker.tracks[0].kalman is not None
