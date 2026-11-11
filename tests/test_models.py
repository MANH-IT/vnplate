import torch

from vnplate.models.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from vnplate.models.detector import DETRDetector
from vnplate.models.matcher import HungarianMatcher, SetCriterion


def _tiny_detector(dec_layers: int = 1) -> DETRDetector:
    return DETRDetector(backbone="tiny", pretrained=False, hidden_dim=32, nheads=4, enc_layers=1, dec_layers=dec_layers, dim_feedforward=64, num_queries=5)


def test_detector_forward():
    model = DETRDetector(backbone="tiny", pretrained=False, hidden_dim=32, nheads=4, enc_layers=1, dec_layers=1, dim_feedforward=64, num_queries=5)
    out = model(torch.randn(1, 3, 96, 128))
    assert out["pred_logits"].shape == (1, 5, 5)
    assert out["pred_boxes"].shape == (1, 5, 4)


def test_giou_range():
    a = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    b = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    giou = generalized_box_iou(box_cxcywh_to_xyxy(a), box_cxcywh_to_xyxy(b))
    assert torch.allclose(giou, torch.ones_like(giou), atol=1e-5)


def test_loss_labels_supervises_background_queries():
    crit = SetCriterion(num_classes=4, matcher=HungarianMatcher())
    logits = torch.zeros(1, 6, 5)
    logits[..., 0] = 5.0
    outputs = {"pred_logits": logits, "pred_boxes": torch.rand(1, 6, 4)}
    targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]), "labels": torch.tensor([0])}]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    loss = crit._loss_labels(outputs, targets, indices)
    assert loss > 1.0


def test_criterion_empty_targets_finite():
    model = _tiny_detector()
    crit = SetCriterion(num_classes=4, matcher=HungarianMatcher())
    outputs = model(torch.randn(2, 3, 64, 96))
    targets = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)} for _ in range(2)]
    losses = crit(outputs, targets)
    assert torch.isfinite(losses["loss"])


def test_criterion_uses_aux_outputs():
    model = _tiny_detector(dec_layers=2)
    crit = SetCriterion(num_classes=4, matcher=HungarianMatcher())
    outputs = model(torch.randn(1, 3, 64, 96))
    targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]), "labels": torch.tensor([2])}]
    losses = crit(outputs, targets)
    assert any(k.startswith("loss_ce_aux") for k in losses)
    main = losses["loss_ce"] + losses["loss_bbox"] + losses["loss_giou"]
    assert losses["loss"] >= main
