from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    arr = np.asarray(boxes, dtype=np.float64)
    single = arr.ndim == 1
    arr = arr.reshape(-1, 4)
    if arr.size == 0:
        return arr.reshape(0, 4)
    x1, y1, x2, y2 = arr.T
    out = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)
    return out[0] if single else out


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    arr = np.asarray(boxes, dtype=np.float64)
    single = arr.ndim == 1
    arr = arr.reshape(-1, 4)
    if arr.size == 0:
        return arr.reshape(0, 4)
    cx, cy, w, h = arr.T
    out = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return out[0] if single else out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-6, None)


def associate(detections: np.ndarray, trackers: np.ndarray, iou_threshold: float = 0.3):
    """Match detections to trackers. Returns (matches, unmatched_dets, unmatched_trks)."""
    if len(trackers) == 0:
        return [], list(range(len(detections))), []
    if len(detections) == 0:
        return [], [], list(range(len(trackers)))
    iou = iou_matrix(detections, trackers)
    row, col = linear_sum_assignment(-iou)
    matches = [(d, t) for d, t in zip(row, col) if iou[d, t] >= iou_threshold]
    matched_d = {d for d, _ in matches}
    matched_t = {t for _, t in matches}
    unmatched_dets = [d for d in range(len(detections)) if d not in matched_d]
    unmatched_trks = [t for t in range(len(trackers)) if t not in matched_t]
    return matches, unmatched_dets, unmatched_trks
