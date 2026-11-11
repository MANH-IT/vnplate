from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.synthetic import collate_detr
from ..models.box_ops import box_cxcywh_to_xyxy
from ..tracking.association import iou_matrix


@torch.no_grad()
def evaluate_detection(model, dataset, device="cuda", iou_threshold=0.5, max_batches=None) -> dict:
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_detr)
    tp = fp = fn = 0
    for i, batch in enumerate(tqdm(loader, desc="eval detector")):
        images = batch["images"].to(device)
        preds = model.detect(images, conf_threshold=0.05)[0]
        gt = batch["targets"][0]["boxes"].numpy()
        if gt.size:
            gt = box_cxcywh_to_xyxy(torch.from_numpy(gt)).numpy()
        pred = preds["boxes"].cpu().numpy()
        pred = box_cxcywh_to_xyxy(torch.from_numpy(pred)).numpy() if pred.size else pred
        if pred.size == 0:
            fn += len(gt)
        elif gt.size == 0:
            fp += len(pred)
        else:
            iou = iou_matrix(pred, gt)
            row, col = linear_sum_assignment(-iou)
            matched = sum(1 for r, c in zip(row, col) if iou[r, c] >= iou_threshold)
            tp += matched
            fp += len(pred) - matched
            fn += len(gt) - matched
        if max_batches is not None and i + 1 >= max_batches:
            break
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def tracking_metrics(gt_frames: list[list[tuple[int, np.ndarray]]], pred_frames: list[list[tuple[int, np.ndarray]]], iou_threshold: float = 0.5) -> dict:
    overlaps: dict[tuple[int, int], float] = defaultdict(float)
    gt_counts: dict[int, int] = defaultdict(int)
    pred_counts: dict[int, int] = defaultdict(int)
    id_switches = 0
    last_match: dict[int, int] = {}

    for gt_objs, pred_objs in zip(gt_frames, pred_frames):
        for gt_id, _ in gt_objs:
            gt_counts[gt_id] += 1
        for pred_id, _ in pred_objs:
            pred_counts[pred_id] += 1
        if not gt_objs or not pred_objs:
            continue
        gt_boxes = np.stack([b for _, b in gt_objs])
        pred_boxes = np.stack([b for _, b in pred_objs])
        iou = iou_matrix(gt_boxes, pred_boxes)
        row, col = linear_sum_assignment(-iou)
        for r, c in zip(row, col):
            if iou[r, c] < iou_threshold:
                continue
            gt_id = gt_objs[r][0]
            pred_id = pred_objs[c][0]
            overlaps[(gt_id, pred_id)] += 1
            if gt_id in last_match and last_match[gt_id] != pred_id:
                id_switches += 1
            last_match[gt_id] = pred_id

    total_gt = sum(gt_counts.values())
    matched = sum(overlaps.values())
    idf1 = 2 * matched / max(1e-6, total_gt + sum(pred_counts.values()))
    return {"idf1": idf1, "id_switches": id_switches, "matched": matched, "total_gt": total_gt}
