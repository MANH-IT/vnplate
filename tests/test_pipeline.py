import numpy as np

from vnplate.fusion.temporal import TemporalFusion
from vnplate.pipeline.video_pipeline import VideoVehiclePipeline
from vnplate.tracking.tracker import MultiObjectTracker


def test_pipeline_with_oracle_boxes_tracks_object():
    pipeline = VideoVehiclePipeline(
        detector=None,
        device="cpu",
        image_size=(96, 160),
        tracker=MultiObjectTracker(max_age=3, min_hits=2, iou_threshold=0.2),
        fusion=TemporalFusion(min_votes=1),
    )
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    confirmed = False
    for i in range(5):
        box = np.array([[20 + i * 5, 30, 100 + i * 5, 70]], dtype=np.float64)
        result = pipeline.process_frame(frame, i, gt_boxes=box)
        confirmed = confirmed or bool(result.tracks)
    assert confirmed
    assert len(pipeline.tracker.confirmed_tracks()) == 1


def test_pipeline_vehicle_oracle_labels_fuse():
    pipeline = VideoVehiclePipeline(
        detector=None,
        device="cpu",
        image_size=(96, 160),
        tracker=MultiObjectTracker(max_age=3, min_hits=1, iou_threshold=0.2),
        fusion=TemporalFusion(min_votes=1),
    )
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    box = np.array([[20, 30, 100, 70]], dtype=np.float64)
    result = pipeline.process_frame(frame, 0, gt_boxes=box, gt_labels=np.array([0]))
    assert any(r["text"] == "car" for r in result.readings)
    assert pipeline.fusion.fuse(result.readings[0]["track_id"]).text == "car"


def test_pipeline_track_labels_stay_distinct_for_overlapping_objects():
    pipeline = VideoVehiclePipeline(
        detector=None,
        device="cpu",
        image_size=(96, 160),
        tracker=MultiObjectTracker(max_age=5, min_hits=1, iou_threshold=0.2, use_kalman=False),
        fusion=TemporalFusion(min_votes=1),
    )
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    boxes = np.array([[0, 0, 100, 50], [20, 10, 120, 60]], dtype=np.float64)
    labels = np.array([1, 2])
    for i in range(3):
        pipeline.process_frame(frame, i, gt_boxes=boxes, gt_labels=labels)
    assert len(pipeline._track_labels) == 2
    assert set(pipeline._track_labels.values()) == {1, 2}
