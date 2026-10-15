"""OpenCLIP loading, checkpoint, and token-feature helpers."""

from __future__ import annotations

import torch
import open_clip
from PIL import Image
from pathlib import Path

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


def load_model(
    device: str | None = None,
    checkpoint: str | Path | None = None,
    model_name: str = MODEL_NAME,
    pretrained: str | None = PRETRAINED,
    image_size: int | tuple[int, int] | None = None,
):
    """Load OpenCLIP, optionally restoring a fine-tuned checkpoint."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if checkpoint is not None:
        pretrained = None
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, force_image_size=image_size
    )
    if checkpoint is not None:
        state = _read_state_dict(checkpoint)
        # The RSTPReid CLIP baseline checkpoint also stores task-specific
        # classification heads. They are not part of OpenCLIP and are not
        # used by this retrieval pipeline; the wrapped base_model weights
        # contain the complete CLIP encoders.
        auxiliary_heads = ("classifier_bge.", "mlp_bge.", "classifier_id_bge.")
        state = {
            key: value
            for key, value in state.items()
            if not key.startswith(auxiliary_heads)
        }
        missing, unexpected = model.load_state_dict(state, strict=False)
        # Some fine-tuned CLIP checkpoints save the encoder under a wrapper
        # module and omit logit_scale. The latter is only used to scale the
        # contrastive logits in ``forward``; this project uses encode_image /
        # encode_text and their token features, so the initialized value is
        # irrelevant for retrieval.
        missing = [key for key in missing if key != "logit_scale"]
        if missing or unexpected:
            raise ValueError(
                "Checkpoint does not match the requested OpenCLIP architecture "
                f"{model_name!r}. Missing keys: {missing[:8]}; "
                f"unexpected keys: {unexpected[:8]}"
            )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer, device


def _read_state_dict(checkpoint: str | Path) -> dict[str, torch.Tensor]:
    path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"CLIP checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was added.
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, dict) and value and all(
                isinstance(k, str) for k in value
            ):
                payload = value
                break
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    state = {}
    for key, value in payload.items():
        if not isinstance(value, torch.Tensor):
            continue
        clean_key = key
        changed = True
        while changed:
            changed = False
            for prefix in (
                "module.",
                "model.",
                "base_model.",
                "clip_model.",
                "clip.",
            ):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix) :]
                    changed = True
                    break
        state[clean_key] = value
    if not state:
        raise ValueError(f"No model tensors found in checkpoint: {path}")
    return state


@torch.inference_mode()
def encode_text(model, tokenizer, texts: list[str], device: str) -> torch.Tensor:
    tokens = tokenizer(texts).to(device)
    features = model.encode_text(tokens)
    return torch.nn.functional.normalize(features.float(), dim=-1)


@torch.inference_mode()
def encode_image(model, preprocess, image: Image.Image, device: str) -> torch.Tensor:
    pixels = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    features = model.encode_image(pixels)
    return torch.nn.functional.normalize(features.float(), dim=-1)


@torch.inference_mode()
def encode_images(model, preprocess, images: list[Image.Image], device: str) -> torch.Tensor:
    pixels = torch.stack([preprocess(image.convert("RGB")) for image in images]).to(device)
    features = model.encode_image(pixels)
    return torch.nn.functional.normalize(features.float(), dim=-1)


@torch.no_grad()
def encode_image_tokens(model, preprocess, images: list[Image.Image], device: str):
    """Return projected ViT tokens and a valid-token mask for an image batch."""
    visual = model.visual
    if not all(hasattr(visual, name) for name in ("conv1", "class_embedding", "positional_embedding", "transformer", "ln_post")):
        raise TypeError("Token reranking currently requires an OpenCLIP ViT visual encoder.")
    pixels = torch.stack([preprocess(image.convert("RGB")) for image in images]).to(device)
    x = visual.conv1(pixels)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype).reshape(1, 1, -1).expand(x.shape[0], 1, -1)
    x = torch.cat((cls, x), dim=1)
    pos = visual.positional_embedding.to(x.dtype)
    if pos.ndim == 3:
        pos = pos.squeeze(0)
    if pos.shape[0] != x.shape[1]:
        raise ValueError(
            "Input image resolution does not match the CLIP positional embeddings; "
            "use the checkpoint's expected image size."
        )
    x = x + pos.unsqueeze(0)
    if hasattr(visual, "ln_pre"):
        x = visual.ln_pre(x)
    transformer = visual.transformer
    if getattr(transformer, "batch_first", False):
        x = transformer(x)
    else:
        x = transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
    x = visual.ln_post(x)
    projection = getattr(visual, "proj", None)
    if projection is not None:
        x = x @ projection
    return x.float(), torch.ones(x.shape[:2], dtype=torch.bool, device=device)


@torch.no_grad()
def encode_text_tokens(model, tokenizer, texts: list[str], device: str):
    """Return projected CLIP token states and a padding mask."""
    tokens = tokenizer(texts).to(device)
    text_model = getattr(model, "text", model)
    required = ("token_embedding", "positional_embedding", "transformer", "ln_final", "text_projection")
    if not all(hasattr(text_model, name) for name in required):
        raise TypeError("Token reranking requires an OpenCLIP Transformer text encoder.")
    x = text_model.token_embedding(tokens)
    x = x + text_model.positional_embedding[: x.shape[1]].to(x.dtype)
    transformer = text_model.transformer
    if getattr(transformer, "batch_first", False):
        x = transformer(x, attn_mask=getattr(text_model, "attn_mask", None))
    else:
        x = transformer(
            x.permute(1, 0, 2), attn_mask=getattr(text_model, "attn_mask", None)
        ).permute(1, 0, 2)
    x = text_model.ln_final(x)
    projection = text_model.text_projection
    if projection is not None:
        x = x @ projection
    mask = tokens.ne(0)
    return x.float(), mask

