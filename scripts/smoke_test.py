from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np
import torch

from vnplate.config import load_config
from vnplate.data.synthetic import VEHICLE_CLASSES, SyntheticDetectionDataset
from vnplate.fusion.temporal import TemporalFusion
from vnplate.models.detector import DETRDetector
from vnplate.models.matcher import HungarianMatcher, SetCriterion
from vnplate.tracking.tracker import MultiObjectTracker


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} torch={torch.__version__}")

    detector = DETRDetector(
        backbone="tiny",
        pretrained=False,
        hidden_dim=64,
        nheads=4,
        enc_layers=1,
        dec_layers=1,
        dim_feedforward=128,
        num_queries=8,
        num_classes=len(VEHICLE_CLASSES),
    ).to(device)
    criterion = SetCriterion(num_classes=len(VEHICLE_CLASSES), matcher=HungarianMatcher()).to(device)
    ds = SyntheticDetectionDataset(length=4, image_size=(128, 192), seed=7)
    batch = [ds[i] for i in range(2)]
    images = torch.stack([b["image"] for b in batch]).to(device)
    targets = [{"boxes": b["boxes"].to(device), "labels": b["labels"].to(device)} for b in batch]
    outputs = detector(images)
    losses = criterion(outputs, targets)
    losses["loss"].backward()
    print(f"detector loss={float(losses['loss']):.4f} logits={tuple(outputs['pred_logits'].shape)}")

    tracker = MultiObjectTracker(max_age=5, min_hits=1, iou_threshold=0.2)
    for i in range(6):
        box = np.array([[20 + i * 4, 30, 80 + i * 4, 60]], dtype=np.float64)
        tracks = tracker.update(box, np.array([0.9]))
    print(f"tracker confirmed={[t.track_id for t in tracks]} active={len(tracker.tracks)}")

    fusion = TemporalFusion(min_votes=2)
    for text, conf in [("car", 0.75), ("car", 0.89), ("bus", 0.61), ("car", 0.95)]:
        fusion.add(1, text, conf)
    result = fusion.fuse(1)
    print(f"fusion text={result.text} conf={result.confidence:.3f} method={result.method}")

    cfg = load_config("configs/default.yaml")
    print(f"config ok: queries={cfg.detector.num_queries} vehicle_classes={cfg.data.vehicle_classes}")
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
