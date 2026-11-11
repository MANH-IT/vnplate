from __future__ import annotations

from pathlib import Path

from .synthetic import SyntheticDetectionDataset
from .yolo import IMAGE_EXTS, YOLODetectionDataset


def _yolo_root(root: Path) -> Path | None:
    candidates = [root] + list(root.iterdir() if root.exists() else [])
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        for image_root in (candidate / "images" / "train", candidate / "train" / "images", candidate / "train"):
            if image_root.exists() and any(p.suffix.lower() in IMAGE_EXTS for p in image_root.rglob("*")):
                return candidate
    return None


def _num_classes(cfg) -> int:
    classes = cfg.data.get("vehicle_classes") or []
    return int(cfg.data.get("num_classes", cfg.detector.get("num_classes", len(classes) or 7)))


def _class_map(cfg) -> dict[int, int] | None:
    raw = cfg.data.get("class_map")
    if not raw:
        return None
    return {int(k): int(v) for k, v in raw.items()}


def build_detection_dataset(cfg, split: str | None = None, synthetic_length: int = 512):
    image_size = tuple(cfg.data.image_size)
    root = Path(cfg.paths.get("data_root", "data/vehicles"))
    split = split or cfg.data.get("split", "train")
    yolo_root = _yolo_root(root)
    if yolo_root is not None:
        return YOLODetectionDataset(
            root=yolo_root,
            split=split,
            image_size=image_size,
            class_map=_class_map(cfg),
            num_classes=_num_classes(cfg),
        )
    return SyntheticDetectionDataset(
        length=synthetic_length,
        image_size=image_size,
        seed=0,
        max_vehicles=int(cfg.data.get("max_vehicles", 2)),
    )
