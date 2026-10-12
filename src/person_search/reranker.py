"""Cross-modal token matcher used to rerank CLIP retrieval candidates."""

from __future__ import annotations

import torch
from torch import nn


class ImageTextMatcher(nn.Module):
    """Fuse frozen CLIP image/text tokens and classify whether they match."""

    def __init__(
        self,
        embed_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.input_norm = nn.LayerNorm(embed_dim)
        self.type_embeddings = nn.Embedding(2, embed_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, 1)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        image_tokens: torch.Tensor,
        image_mask: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch = image_tokens.shape[0]
        image_tokens = self.input_norm(image_tokens) + self.type_embeddings.weight[0]
        text_tokens = self.input_norm(text_tokens) + self.type_embeddings.weight[1]
        cls = self.cls_token.expand(batch, -1, -1)
        sequence = torch.cat((cls, image_tokens, text_tokens), dim=1)
        cls_mask = torch.ones((batch, 1), dtype=torch.bool, device=sequence.device)
        padding_mask = ~torch.cat((cls_mask, image_mask, text_mask), dim=1)
        fused = self.fusion(sequence, src_key_padding_mask=padding_mask)
        return self.classifier(self.output_norm(fused[:, 0])).squeeze(-1)


def load_reranker(path: str, device: str):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"Unsupported reranker checkpoint: {path}")
    model = ImageTextMatcher(
        embed_dim=int(payload.get("embed_dim", 512)),
        num_layers=int(payload.get("num_layers", 2)),
        num_heads=int(payload.get("num_heads", 8)),
        dropout=float(payload.get("dropout", 0.1)),
    )
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload
