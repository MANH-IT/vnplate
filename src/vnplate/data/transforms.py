from __future__ import annotations

import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    arr = image_rgb.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def from_tensor(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def rescale_boxes(boxes: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    return boxes


def rescale_poly(poly: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2).copy()
    poly[:, 0] *= scale_x
    poly[:, 1] *= scale_y
    return poly


def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return boxes


def polygon_to_box(poly: np.ndarray) -> np.ndarray:
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    return np.array([poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max()])


def order_polygon(poly: np.ndarray) -> np.ndarray:
    poly = np.asarray(poly, dtype=np.float32).reshape(4, 2)
    s = poly.sum(axis=1)
    d = np.diff(poly, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = poly[np.argmin(s)]
    ordered[2] = poly[np.argmax(s)]
    ordered[1] = poly[np.argmin(d)]
    ordered[3] = poly[np.argmax(d)]
    return ordered


def perspective_crop(image_rgb: np.ndarray, poly: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
    w, h = out_size
    src = order_polygon(poly)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image_rgb, matrix, (w, h))


def normalize_boxes_cxcywh(boxes_xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4).copy()
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    boxes[:, [0, 2]] /= width
    boxes[:, [1, 3]] /= height
    x1, y1, x2, y2 = boxes.T
    return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)
