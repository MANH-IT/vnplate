from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats: int = 128, temperature: float = 10000.0, normalize: bool = True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        not_mask = (~mask).to(torch.float32)
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * 2 * math.pi
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * 2 * math.pi

        dim_t = torch.arange(self.num_pos_feats, device=mask.device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def _get_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    return nn.ReLU()


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation(activation)

    def forward(self, src, src_key_padding_mask=None, pos=None):
        q = k = src if pos is None else src + pos
        src2 = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)[0]
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout3(src2))
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, layer: TransformerEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, mask=None, pos=None):
        key_padding_mask = mask.flatten(1) if mask is not None else None
        flat_pos = pos.flatten(2).transpose(1, 2) if pos is not None else None
        out = src
        for layer in self.layers:
            out = layer(out, src_key_padding_mask=key_padding_mask, pos=flat_pos)
        return out


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation(activation)

    def forward(self, tgt, memory, memory_key_padding_mask=None, pos=None, query_pos=None, tgt_mask=None):
        q = k = tgt if query_pos is None else tgt + query_pos
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        q2 = tgt if query_pos is None else tgt + query_pos
        k2 = memory if pos is None else memory + pos
        tgt2 = self.cross_attn(q2, k2, value=memory, key_padding_mask=memory_key_padding_mask)[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, layer: TransformerDecoderLayer, num_layers: int, return_intermediate: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, memory_mask=None, pos=None, query_pos=None, tgt_mask=None):
        key_padding_mask = memory_mask.flatten(1) if memory_mask is not None else None
        flat_pos = pos.flatten(2).transpose(1, 2) if pos is not None else None
        out = tgt
        intermediate = []
        for layer in self.layers:
            out = layer(out, memory, memory_key_padding_mask=key_padding_mask, pos=flat_pos, query_pos=query_pos, tgt_mask=tgt_mask)
            if self.return_intermediate:
                intermediate.append(out)
        return torch.stack(intermediate) if self.return_intermediate else out


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x
