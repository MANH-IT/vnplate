from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .transformer import (
    MLP,
    PositionEmbeddingSine,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)


class DETRDetector(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        hidden_dim: int = 256,
        nheads: int = 8,
        enc_layers: int = 3,
        dec_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_queries: int = 30,
        num_classes: int = 9,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.backbone = build_backbone(backbone, pretrained=pretrained, out_channels=hidden_dim)
        self.pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        enc_layer = TransformerEncoderLayer(hidden_dim, nheads, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(enc_layer, enc_layers)
        dec_layer = TransformerDecoderLayer(hidden_dim, nheads, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(dec_layer, dec_layers, return_intermediate=True)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.class_embed.weight)
        nn.init.constant_(self.class_embed.bias, 0)
        nn.init.normal_(self.query_embed.weight)

    def forward(self, images: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
        feat = self.backbone(images)
        b, _, h, w = feat.shape
        if mask is None:
            mask = torch.zeros(b, h, w, dtype=torch.bool, device=images.device)
        else:
            mask = F.interpolate(mask.float().unsqueeze(1), size=(h, w), mode="nearest").squeeze(1).bool()
        pos = self.pos_embed(mask)
        src = feat.flatten(2).transpose(1, 2)
        memory = self.encoder(src, mask, pos)
        tgt = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)
        hs = self.decoder(tgt, memory, mask, pos)
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        return {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1], "aux_outputs": [{"pred_logits": a, "pred_boxes": c} for a, c in zip(outputs_class[:-1], outputs_coord[:-1])]}

    @torch.no_grad()
    def detect(self, images: torch.Tensor, conf_threshold: float = 0.5) -> list[dict]:
        outputs = self.forward(images)
        prob = outputs["pred_logits"].softmax(-1)
        scores, labels = prob[..., :-1].max(-1)
        boxes = outputs["pred_boxes"]
        results = []
        for i in range(images.shape[0]):
            keep = scores[i] > conf_threshold
            results.append({"boxes": boxes[i][keep], "scores": scores[i][keep], "labels": labels[i][keep]})
        return results


# Backward/forward-compatible aliases
VehicleDetector = DETRDetector
