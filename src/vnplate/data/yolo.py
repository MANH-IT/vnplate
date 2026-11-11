from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import clip_boxes, normalize_boxes_cxcywh, to_tensor

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class YOLODetectionDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: tuple[int, int] = (384, 640),
        class_map: dict[int, int] | None = None,
        num_classes: int = 9,
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = tuple(image_size)
        self.class_map = class_map
        self.num_classes = num_classes
        self.samples = self._index()

    def _index(self) -> list[dict]:
        splits = [self.split]
        if self.split == "val":
            splits.append("valid")
        elif self.split == "valid":
            splits.append("val")
        candidates = []
        for sp in splits:
            candidates.append((self.root / "images" / sp, self.root / "labels" / sp))
            candidates.append((self.root / sp / "images", self.root / sp / "labels"))
            candidates.append((self.root / sp, self.root / sp))
        image_root, label_root = next(((img, lbl) for img, lbl in candidates if img.exists() and lbl.exists()), (None, None))
        if image_root is None:
            return []
        samples = []
        for image_path in sorted(p for p in image_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("._") and not p.name.startswith(".")):
            label_path = label_root / f"{image_path.stem}.txt"
            samples.append({"image_path": image_path, "label_path": label_path})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]
        image = cv2.imread(str(item["image_path"]))
        if image is None:
            raise FileNotFoundError(item["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        new_h, new_w = self.image_size
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        boxes, labels = [], []
        if item["label_path"].exists():
            for raw in item["label_path"].read_text(encoding="utf-8").splitlines():
                parts = raw.split()
                if len(parts) < 5:
                    continue
                src, cx, cy, w, h = parts[:5]
                try:
                    src = int(float(src))
                except ValueError:
                    continue
                label = self.class_map[src] if self.class_map is not None and src in self.class_map else src
                if label < 0 or label >= self.num_classes:
                    continue
                cx, cy, w, h = (float(cx), float(cy), float(w), float(h))
                x1 = (cx - w / 2) * new_w
                y1 = (cy - h / 2) * new_h
                x2 = (cx + w / 2) * new_w
                y2 = (cy + h / 2) * new_h
                boxes.append((x1, y1, x2, y2))
                labels.append(label)

        boxes_np = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        boxes_np = clip_boxes(boxes_np, new_w, new_h)
        boxes_norm = normalize_boxes_cxcywh(boxes_np, new_w, new_h)
        return {
            "image": to_tensor(image),
            "boxes": torch.from_numpy(boxes_norm).float(),
            "boxes_xyxy": torch.from_numpy(boxes_np).float(),
            "labels": torch.tensor(labels, dtype=torch.long),
            "video_id": f"{self.root.name}/{self.split}",
            "frame_id": item["image_path"].stem,
            "orig_size": torch.tensor([orig_h, orig_w]),
        }
