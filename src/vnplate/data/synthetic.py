from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .taxonomy import ROAD_USER_CLASSES, ROAD_USER_COLORS
from .transforms import normalize_boxes_cxcywh, to_tensor

VEHICLE_CLASSES = ROAD_USER_CLASSES
VEHICLE_COLORS = ROAD_USER_COLORS


def _background(size: tuple[int, int], rng: random.Random) -> np.ndarray:
    h, w = size
    base = np.array([rng.randint(20, 90) for _ in range(3)], dtype=np.uint8)
    img = np.tile(base, (h, w, 1))
    noise = np.random.normal(0, 12, (h, w, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


class SyntheticDetectionDataset(Dataset):
    def __init__(self, length: int = 512, image_size: tuple[int, int] = (384, 640), seed: int = 0, max_vehicles: int = 2):
        self.length = length
        self.image_size = image_size
        self.seed = seed
        self.max_vehicles = max_vehicles

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed + idx)
        h, w = self.image_size
        scene = _background(self.image_size, rng)
        boxes, labels = [], []
        for _ in range(rng.randint(1, self.max_vehicles)):
            cls = rng.randrange(len(VEHICLE_CLASSES))
            if cls in (0, 1):  # person, bicycle
                pw = rng.randint(20, min(60, w - 1))
                ph = rng.randint(40, min(120, h - 1))
            else:
                pw = rng.randint(60, min(180, w - 1))
                ph = rng.randint(40, min(120, h - 1))
            x1 = rng.randint(0, max(0, w - pw - 1))
            y1 = rng.randint(0, max(0, h - ph - 1))
            color = VEHICLE_COLORS[cls]
            cv2.rectangle(scene, (x1, y1), (x1 + pw, y1 + ph), color, -1)
            cv2.rectangle(scene, (x1, y1), (x1 + pw, y1 + ph), (0, 0, 0), 2)
            boxes.append((x1, y1, x1 + pw, y1 + ph))
            labels.append(cls)
        if rng.random() < 0.5:
            scene = cv2.GaussianBlur(scene, (3, 3), 0)

        boxes_np = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        boxes_norm = normalize_boxes_cxcywh(boxes_np, w, h)
        return {
            "image": to_tensor(scene),
            "boxes": torch.from_numpy(boxes_norm).float(),
            "boxes_xyxy": torch.from_numpy(boxes_np).float(),
            "labels": torch.tensor(labels, dtype=torch.long),
            "video_id": f"synthetic/{idx}",
            "frame_id": "frame1",
            "orig_size": torch.tensor([h, w]),
        }


def collate_detr(batch: list[dict]) -> dict:
    images = torch.stack([b["image"] for b in batch], dim=0)
    targets = [
        {"boxes": b["boxes"], "labels": b["labels"], "video_id": b["video_id"], "frame_id": b["frame_id"]}
        for b in batch
    ]
    return {"images": images, "targets": targets}


def _simulate_night(frame: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dark = (frame.astype(np.float32) * 0.22).clip(0, 255)
    noise = rng.normal(0, 7, dark.shape)
    out = np.clip(dark + noise, 0, 255).astype(np.uint8)
    out[..., 2] = np.clip(out[..., 2].astype(np.int16) + 10, 0, 255).astype(np.uint8)
    return out


def write_demo_video(
    path: str | Path,
    labels: list[str] | None = None,
    frames: int = 90,
    image_size: tuple[int, int] = (384, 640),
    fps: int = 30,
    night: bool = False,
) -> list[dict]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = image_size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if labels is None:
        labels = ["car", "motorbike"]
    rng = random.Random(42)
    base_scene = _background(image_size, rng)
    annotations = []
    n = max(1, len(labels))
    pw = min(160, max(80, w // 3))
    ph = min(52, max(20, h // (n + 1)))
    for f in range(frames):
        scene = base_scene.copy()
        frame_objs = []
        for i, label in enumerate(labels):
            progress = f / max(1, frames - 1)
            x1 = int(20 + progress * (w - pw - 40) + (i % 3) * 12)
            y1 = int((i + 0.5) * (h / n) - ph / 2 + 10 * np.sin(2 * np.pi * progress * 2 + i))
            x1 = max(0, min(w - pw, x1))
            y1 = max(0, min(h - ph, y1))
            color = VEHICLE_COLORS[VEHICLE_CLASSES.index(label) % len(VEHICLE_COLORS)] if label in VEHICLE_CLASSES else VEHICLE_COLORS[i % len(VEHICLE_COLORS)]
            cv2.rectangle(scene, (x1, y1), (x1 + pw, y1 + ph), color, -1)
            cv2.rectangle(scene, (x1, y1), (x1 + pw, y1 + ph), (0, 0, 0), 2)
            cv2.putText(scene, str(label), (x1 + 4, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            frame_objs.append({"track_id": i + 1, "bbox": [x1, y1, x1 + pw, y1 + ph], "label": str(label)})
        output = _simulate_night(scene, seed=f) if night else scene
        writer.write(cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        annotations.append({"frame": f, "objects": frame_objs})
    writer.release()
    return annotations
