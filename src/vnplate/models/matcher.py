from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list[dict]):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        indices = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"]
            tgt_bbox = targets[b]["boxes"]
            if tgt_ids.numel() == 0:
                indices.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue
            prob_b = out_prob[b * num_queries : (b + 1) * num_queries]
            bbox_b = out_bbox[b * num_queries : (b + 1) * num_queries]
            cost_class = -prob_b[:, tgt_ids]
            cost_bbox = torch.cdist(bbox_b, tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(bbox_b), box_cxcywh_to_xyxy(tgt_bbox))
            cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            cost = cost.detach().cpu()
            row, col = linear_sum_assignment(cost)
            indices.append((torch.as_tensor(row, dtype=torch.long), torch.as_tensor(col, dtype=torch.long)))
        return indices


def _sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    prob = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum()


class SetCriterion(nn.Module):
    def __init__(self, num_classes: int, matcher: HungarianMatcher, weight_dict=None, eos_coef: float = 0.1, aux_loss_weight: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict or {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
        self.eos_coef = eos_coef
        self.aux_loss_weight = aux_loss_weight
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def _loss_labels(self, outputs, targets, indices):
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        idx = self._get_src_permutation_idx(indices, device)
        target_classes = torch.full(pred_logits.shape[:2], self.num_classes, dtype=torch.long, device=device)
        if idx[0].numel():
            tgt_classes = torch.cat([t["labels"][j].to(device) for t, (_, j) in zip(targets, indices) if len(j)])
            target_classes[idx] = tgt_classes
        return F.cross_entropy(pred_logits.transpose(1, 2), target_classes, self.empty_weight.to(device))

    def _loss_boxes(self, outputs, targets, indices, num_boxes: int):
        device = outputs["pred_boxes"].device
        idx = self._get_src_permutation_idx(indices, device)
        src_boxes = outputs["pred_boxes"][idx]
        matched = [t["boxes"][j].to(device) for t, (_, j) in zip(targets, indices) if len(j)]
        if not matched:
            zero = src_boxes.sum()
            return zero, zero
        tgt_boxes = torch.cat(matched)
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="sum") / num_boxes
        giou = generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes))
        loss_giou = (1 - torch.diag(giou)).sum() / num_boxes if giou.numel() else src_boxes.sum()
        return loss_bbox, loss_giou

    def _get_src_permutation_idx(self, indices, device=None):
        device = device or torch.device("cpu")
        srcs = [src for (src, _) in indices if len(src)]
        if not srcs:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty
        batch_idx = torch.cat([torch.full_like(src, b) for b, (src, _) in enumerate(indices) if len(src)]).to(device)
        src_idx = torch.cat(srcs).to(device)
        return batch_idx, src_idx

    def _compute(self, outputs, targets, indices, num_boxes: int) -> dict:
        loss_ce = self._loss_labels(outputs, targets, indices)
        loss_bbox, loss_giou = self._loss_boxes(outputs, targets, indices, num_boxes)
        return {
            "loss_ce": loss_ce * self.weight_dict["loss_ce"],
            "loss_bbox": loss_bbox * self.weight_dict["loss_bbox"],
            "loss_giou": loss_giou * self.weight_dict["loss_giou"],
        }

    def forward(self, outputs, targets):
        num_boxes = max(sum(int(t["labels"].numel()) for t in targets), 1)
        indices = self.matcher(outputs, targets)
        losses = self._compute(outputs, targets, indices, num_boxes)
        aux_total = 0.0
        for i, aux in enumerate(outputs.get("aux_outputs", [])):
            aux_indices = self.matcher(aux, targets)
            aux_losses = self._compute(aux, targets, aux_indices, num_boxes)
            for key, value in aux_losses.items():
                losses[f"{key}_aux{i}"] = value * self.aux_loss_weight
            aux_total += (aux_losses["loss_ce"] + aux_losses["loss_bbox"] + aux_losses["loss_giou"]) * self.aux_loss_weight
        losses["loss"] = losses["loss_ce"] + losses["loss_bbox"] + losses["loss_giou"] + aux_total
        return losses
