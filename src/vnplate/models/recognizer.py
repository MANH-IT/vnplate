from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sacnn import SACNN
from .stn import SpatialTransformer
from .transformer import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)


class PlateRecognizer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        nheads: int = 8,
        enc_layers: int = 2,
        dec_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_stn: bool = True,
        plate_size: tuple[int, int] = (64, 192),
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        lstm_layers: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.plate_size = plate_size
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.use_stn = use_stn

        self.stn = SpatialTransformer(3, 64, plate_size) if use_stn else nn.Identity()
        self.backbone = SACNN(3, hidden_dim, nhead=max(4, nheads // 2), dropout=dropout)
        self.input_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.memory_norm = nn.LayerNorm(hidden_dim)

        enc_layer = TransformerEncoderLayer(hidden_dim, nheads, dim_feedforward, dropout)
        self.encoder = _SeqEncoder(enc_layer, enc_layers)
        dec_layer = TransformerDecoderLayer(hidden_dim, nheads, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(dec_layer, dec_layers)
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(64, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def _features(self, images: torch.Tensor):
        if self.use_stn:
            images, _ = self.stn(images)
        feat, pos = self.backbone(images)
        feat = self.input_proj(feat)
        b, c, h, w = feat.shape
        seq = feat.flatten(2).transpose(1, 2)
        pos = pos.flatten(2).transpose(1, 2)
        seq = seq + pos
        lstm_out, _ = self.lstm(seq)
        memory = self.memory_norm(lstm_out)
        return memory

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self._features(images)

    def forward(self, images: torch.Tensor, dec_in: torch.Tensor) -> torch.Tensor:
        memory = self.encode(images)
        b, t = dec_in.shape
        t = min(t, self.pos_embed.num_embeddings)
        dec_in = dec_in[:, :t]
        positions = torch.arange(t, device=images.device).unsqueeze(0).expand(b, t)
        tgt = self.token_embed(dec_in) + self.pos_embed(positions)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(t, device=images.device)
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        return self.output(out)

    @torch.no_grad()
    def greedy_decode(self, images: torch.Tensor, max_len: int = 12, return_confidence: bool = False):
        memory = self.encode(images)
        b = images.shape[0]
        ids = torch.full((b, 1), self.bos_id, dtype=torch.long, device=images.device)
        finished = torch.zeros(b, dtype=torch.bool, device=images.device)
        step_conf = []
        for _ in range(max_len):
            t = min(ids.shape[1], self.pos_embed.num_embeddings)
            positions = torch.arange(t, device=images.device).unsqueeze(0).expand(b, t)
            tgt = self.token_embed(ids[:, :t]) + self.pos_embed(positions)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(t, device=images.device)
            out = self.decoder(tgt, memory, tgt_mask=tgt_mask)
            logits = self.output(out[:, -1])
            prob = logits.softmax(-1)
            conf, next_id = prob.max(-1)
            conf = conf.masked_fill(finished, 0.0)
            step_conf.append(conf)
            ids = torch.cat([ids, next_id.unsqueeze(1)], dim=1)
            finished |= next_id == self.eos_id
            if finished.all():
                break
        ids = ids[:, 1:]
        if not return_confidence:
            return ids
        confs = torch.stack(step_conf, dim=1)
        lens = (confs > 0).sum(dim=1).clamp(min=1)
        return ids, confs.sum(dim=1) / lens

    def recognize_texts(self, images: torch.Tensor, vocab, max_len: int = 12) -> list[str]:
        ids = self.greedy_decode(images, max_len=max_len)
        return [vocab.decode(row.tolist()) for row in ids]


class _SeqEncoder(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])

    def forward(self, src):
        out = src
        for layer in self.layers:
            out = layer(out)
        return out
