from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.factory import build_detection_dataset
from ..data.synthetic import collate_detr
from ..engine.evaluate import evaluate_detection
from ..models.detector import DETRDetector
from ..models.matcher import HungarianMatcher, SetCriterion


def build_detector(cfg) -> DETRDetector:
    d = cfg.detector
    return DETRDetector(
        backbone=d.backbone,
        pretrained=bool(d.get("pretrained", True)),
        hidden_dim=int(d.hidden_dim),
        nheads=int(d.nheads),
        enc_layers=int(d.enc_layers),
        dec_layers=int(d.dec_layers),
        dim_feedforward=int(d.dim_feedforward),
        dropout=float(d.dropout),
        num_queries=int(d.num_queries),
        num_classes=int(d.get("num_classes", len(cfg.data.get("vehicle_classes", [])) or 9)),
    )


def _evaluate_val(cfg, model, device, max_batches: int):
    val = build_detection_dataset(cfg, split="val", synthetic_length=64)
    if len(val) == 0 or max_batches <= 0:
        return None
    return evaluate_detection(model, val, device=device, max_batches=max_batches)


def train_detector(cfg, max_steps: int | None = None, synthetic_length: int = 512, eval_batches: int | None = None) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_detection_dataset(cfg, synthetic_length=synthetic_length)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.detector.batch_size),
        shuffle=True,
        num_workers=int(cfg.data.get("num_workers", 0)),
        collate_fn=collate_detr,
        drop_last=False,
    )
    model = build_detector(cfg).to(device)
    matcher = HungarianMatcher()
    criterion = SetCriterion(num_classes=model.num_classes, matcher=matcher).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(cfg.detector.lr), weight_decay=float(cfg.detector.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    ckpt_dir = Path(cfg.paths.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    smoke = max_steps is not None
    eval_batches = int(eval_batches if eval_batches is not None else cfg.detector.get("eval_batches", 20))
    best_loss = float("inf")
    best_f1 = -1.0
    step = 0
    start = time.time()
    for epoch in range(int(cfg.detector.epochs)):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        progress = tqdm(loader, desc=f"detector epoch {epoch + 1}")
        for batch in progress:
            images = batch["images"].to(device)
            targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in batch["targets"]]
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(images)
                losses = criterion(outputs, targets)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 0.1)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(losses["loss"].detach())
            epoch_steps += 1
            step += 1
            progress.set_postfix(loss=f"{float(losses['loss']):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            if max_steps is not None and step >= max_steps:
                break
        mean_loss = epoch_loss / max(1, epoch_steps)
        metrics = _evaluate_val(cfg, model, device, eval_batches)
        save_best = False
        if metrics is not None:
            f1 = float(metrics.get("f1", -1.0))
            save_best = f1 > best_f1
            if save_best:
                best_f1 = f1
        else:
            save_best = mean_loss < best_loss
            if save_best:
                best_loss = mean_loss
        if smoke:
            torch.save({"model": model.state_dict(), "config": dict(cfg.detector), "epoch": epoch, "loss": mean_loss}, ckpt_dir / "detector_smoke.pt")
        else:
            if save_best:
                torch.save(
                    {"model": model.state_dict(), "config": dict(cfg.detector), "epoch": epoch, "loss": mean_loss, "val_metrics": metrics},
                    ckpt_dir / "detector_best.pt",
                )
            torch.save({"model": model.state_dict(), "config": dict(cfg.detector), "epoch": epoch, "loss": mean_loss}, ckpt_dir / "detector_last.pt")
        if metrics is not None:
            print(f"epoch {epoch + 1}: loss={mean_loss:.4f} val_f1={metrics['f1']:.4f} val_p={metrics['precision']:.4f} val_r={metrics['recall']:.4f} elapsed={time.time() - start:.1f}s")
        else:
            print(f"epoch {epoch + 1}: loss={mean_loss:.4f} elapsed={time.time() - start:.1f}s")
        if max_steps is not None and step >= max_steps:
            break
    return ckpt_dir / ("detector_smoke.pt" if smoke else "detector_last.pt")


def load_detector(path: str | Path, device: str | torch.device = "cuda") -> DETRDetector:
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = DETRDetector(
        backbone=cfg.get("backbone", "resnet18"),
        pretrained=False,
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        nheads=int(cfg.get("nheads", 8)),
        enc_layers=int(cfg.get("enc_layers", 3)),
        dec_layers=int(cfg.get("dec_layers", 3)),
        dim_feedforward=int(cfg.get("dim_feedforward", 1024)),
        dropout=float(cfg.get("dropout", 0.1)),
        num_queries=int(cfg.get("num_queries", 30)),
        num_classes=int(cfg.get("num_classes", 9)),
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()
