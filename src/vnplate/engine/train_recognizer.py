from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.crops import collate_crops
from ..data.factory import build_crop_dataset, vocab_from_config
from ..models.recognizer import PlateRecognizer


def build_recognizer(cfg, vocab) -> PlateRecognizer:
    r = cfg.recognizer
    return PlateRecognizer(
        vocab_size=len(vocab),
        hidden_dim=int(r.hidden_dim),
        nheads=int(r.nheads),
        enc_layers=int(r.enc_layers),
        dec_layers=int(r.dec_layers),
        dim_feedforward=int(r.dim_feedforward),
        dropout=float(r.dropout),
        use_stn=bool(r.get("use_stn", True)),
        plate_size=tuple(cfg.data.plate_size),
        pad_id=vocab.pad_id,
        bos_id=vocab.sos_id,
        eos_id=vocab.eos_id,
    )


def _encode_batch(texts: list[str], vocab, max_len: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [vocab.encode(t, add_sos_eos=True, max_len=max_len) for t in texts]
    width = max(len(ids) for ids in encoded)
    dec_out = torch.full((len(encoded), width), vocab.pad_id, dtype=torch.long)
    for i, ids in enumerate(encoded):
        dec_out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    dec_in = torch.cat([torch.full((len(encoded), 1), vocab.sos_id, dtype=torch.long), dec_out[:, :-1]], dim=1)
    return dec_in.to(device), dec_out.to(device)


def train_recognizer(cfg, max_steps: int | None = None, synthetic_length: int = 2048) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = vocab_from_config(cfg)
    dataset = build_crop_dataset(cfg, synthetic_length=synthetic_length)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.recognizer.batch_size),
        shuffle=True,
        num_workers=int(cfg.data.get("num_workers", 0)),
        collate_fn=collate_crops,
        drop_last=False,
    )
    model = build_recognizer(cfg, vocab).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.recognizer.lr), weight_decay=float(cfg.recognizer.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ckpt_dir = Path(cfg.paths.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    max_len = int(cfg.data.get("max_plate_len", 12))
    best = float("inf")
    step = 0
    start = time.time()
    for epoch in range(int(cfg.recognizer.epochs)):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        correct = 0
        total = 0
        progress = tqdm(loader, desc=f"recognizer epoch {epoch + 1}")
        for batch in progress:
            if not batch["images"].numel():
                continue
            images = batch["images"].to(device)
            dec_in, dec_out = _encode_batch(batch["texts"], vocab, max_len, device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images, dec_in)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), dec_out.reshape(-1), ignore_index=vocab.pad_id)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach())
            epoch_steps += 1
            pred = logits.argmax(-1)
            mask = dec_out != vocab.pad_id
            correct += int(((pred == dec_out) & mask).sum())
            total += int(mask.sum())
            step += 1
            progress.set_postfix(loss=f"{float(loss):.4f}", acc=f"{correct / max(1, total):.3f}")
            if max_steps is not None and step >= max_steps:
                break
        mean_loss = epoch_loss / max(1, epoch_steps)
        token_acc = correct / max(1, total)
        state = {"model": model.state_dict(), "config": dict(cfg.recognizer), "data": dict(cfg.data), "vocab": vocab.state_dict(), "epoch": epoch, "loss": mean_loss}
        if mean_loss < best:
            best = mean_loss
            torch.save(state, ckpt_dir / "recognizer_best.pt")
        torch.save(state, ckpt_dir / "recognizer_last.pt")
        print(f"epoch {epoch + 1}: loss={mean_loss:.4f} token_acc={token_acc:.3f} elapsed={time.time() - start:.1f}s")
        if max_steps is not None and step >= max_steps:
            break
    return ckpt_dir / "recognizer_last.pt"


def load_recognizer(path: str | Path, device: str | torch.device = "cuda"):
    from ..data.vocab import PlateVocab

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    vocab = PlateVocab.from_state_dict(ckpt["vocab"])
    cfg = ckpt.get("config", {})
    data = ckpt.get("data", {})
    model = PlateRecognizer(
        vocab_size=len(vocab),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        nheads=int(cfg.get("nheads", 8)),
        enc_layers=int(cfg.get("enc_layers", 2)),
        dec_layers=int(cfg.get("dec_layers", 2)),
        dim_feedforward=int(cfg.get("dim_feedforward", 1024)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_stn=bool(cfg.get("use_stn", True)),
        plate_size=tuple(data.get("plate_size", (64, 192))),
        pad_id=vocab.pad_id,
        bos_id=vocab.sos_id,
        eos_id=vocab.eos_id,
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), vocab
