"""OpenCLIP loading and embedding helpers."""

from __future__ import annotations

import torch
import open_clip
from PIL import Image

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


def load_model(device: str | None = None):
    """Load the baseline model; OpenCLIP downloads its weights on first use."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, preprocess, tokenizer, device


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

