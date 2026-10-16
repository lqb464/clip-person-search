"""Training and evaluation helpers for the RSTPReid CLIP reranker."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .model import (
    MODEL_NAME,
    encode_image_tokens,
    encode_images,
    encode_text,
    encode_text_tokens,
    load_model,
)
from .reranker import ImageTextMatcher, load_reranker


def load_records(root: str | Path, split: str) -> list[dict]:
    root = Path(root).expanduser()
    annotation_path = root / "data_captions.json"
    if not annotation_path.is_file() or not (root / "imgs").is_dir():
        raise FileNotFoundError(
            f"Expected data_captions.json and imgs/ under RSTPReid root: {root}"
        )
    records = json.loads(annotation_path.read_text(encoding="utf-8"))
    filtered = [r for r in records if str(r.get("split", "")).lower() == split]
    if not filtered:
        raise ValueError(f"No RSTPReid records found for split={split!r}")
    missing = [root / "imgs" / r["img_path"] for r in filtered if not (root / "imgs" / r["img_path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images are missing; first: {missing[0]}")
    return filtered


class CaptionPairDataset(Dataset):
    def __init__(self, root: str | Path, records: list[dict]):
        self.image_dir = Path(root).expanduser() / "imgs"
        self.items = []
        self.captions_by_pid: dict[int, list[str]] = defaultdict(list)
        for record in records:
            pid = int(record["id"])
            for caption in record["captions"]:
                caption = str(caption)
                self.items.append((record["img_path"], caption, pid))
                self.captions_by_pid[pid].append(caption)
        if not self.items:
            raise ValueError("The selected split contains no image-caption pairs")
        self.pids = list(self.captions_by_pid)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        relative_path, caption, pid = self.items[index]
        with Image.open(self.image_dir / relative_path) as image:
            image = image.convert("RGB")
        negative_pid = random.choice(self.pids)
        while negative_pid == pid:
            negative_pid = random.choice(self.pids)
        negative_caption = random.choice(self.captions_by_pid[negative_pid])
        return image, caption, negative_caption, pid


def _collate_pairs(batch):
    images, captions, negative_captions, pids = zip(*batch)
    return list(images), list(captions), list(negative_captions), torch.tensor(pids)


def _encode_train_batch(model, preprocess, tokenizer, batch, device):
    images, captions, negative_captions, pids = batch
    image_tokens, image_mask = encode_image_tokens(model, preprocess, images, device)
    text_tokens, text_mask = encode_text_tokens(
        model, tokenizer, list(captions) + list(negative_captions), device
    )
    paired_images = torch.cat((image_tokens, image_tokens), dim=0)
    paired_image_masks = torch.cat((image_mask, image_mask), dim=0)
    return paired_images, paired_image_masks, text_tokens, text_mask, pids.to(device)


def train_reranker(
    root: str | Path,
    clip_checkpoint: str | Path,
    output: str | Path,
    device: str | None = None,
    batch_size: int = 32,
    epochs: int = 5,
    learning_rate: float = 2e-4,
    num_workers: int = 0,
    seed: int = 7,
    model_name: str = MODEL_NAME,
    image_size: tuple[int, int] = (384, 128),
    image_preprocess: str = "open_clip",
    encoder_learning_rate: float = 1e-6,
):
    random.seed(seed)
    torch.manual_seed(seed)
    train_records = load_records(root, "train")
    val_records = load_records(root, "val")
    train_data = CaptionPairDataset(root, train_records)
    val_data = CaptionPairDataset(root, val_records)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_pairs,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_pairs,
        pin_memory=torch.cuda.is_available(),
    )
    model, preprocess, tokenizer, device = load_model(
        device=device,
        checkpoint=clip_checkpoint,
        model_name=model_name,
        image_size=image_size,
        image_preprocess=image_preprocess,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    text_projection = getattr(model, "text_projection", None)
    if text_projection is None and hasattr(model, "text"):
        text_projection = getattr(model.text, "text_projection", None)
    if text_projection is None:
        raise TypeError("Could not infer the CLIP text embedding dimension")
    embed_dim = int(text_projection.shape[-1])
    matcher = ImageTextMatcher(embed_dim=embed_dim).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": encoder_learning_rate},
            {"params": matcher.parameters(), "lr": learning_rate},
        ],
        weight_decay=0.01,
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    best_val_loss = float("inf")
    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        matcher.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            image_tokens, image_mask, text_tokens, text_mask, _ = _encode_train_batch(
                model, preprocess, tokenizer, batch, device
            )
            batch_count = len(batch[1])
            labels = torch.cat(
                (
                    torch.ones(batch_count, device=device),
                    torch.zeros(batch_count, device=device),
                )
            )
            logits = matcher(image_tokens, image_mask, text_tokens, text_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(matcher.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item() * len(labels)
            train_count += len(labels)

        model.eval()
        matcher.eval()
        val_loss_sum = 0.0
        val_count = 0
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
                image_tokens, image_mask, text_tokens, text_mask, _ = _encode_train_batch(
                    model, preprocess, tokenizer, batch, device
                )
                batch_count = len(batch[1])
                labels = torch.cat(
                    (
                        torch.ones(batch_count, device=device),
                        torch.zeros(batch_count, device=device),
                    )
                )
                logits = matcher(image_tokens, image_mask, text_tokens, text_mask)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * len(labels)
                val_count += len(labels)
                correct += (logits.ge(0) == labels.bool()).sum().item()
        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = val_loss_sum / max(val_count, 1)
        val_accuracy = correct / max(val_count, 1)
        print(
            f"epoch {epoch}/{epochs} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "state_dict": matcher.state_dict(),
                    "encoder_state_dict": model.state_dict(),
                    "encoder_finetuned": True,
                    "encoder_learning_rate": encoder_learning_rate,
                    "matcher_learning_rate": learning_rate,
                    "embed_dim": embed_dim,
                    "num_layers": 2,
                    "num_heads": 8,
                    "dropout": 0.1,
                    "clip_model_name": model_name,
                    "image_size": image_size,
                    "image_preprocess": image_preprocess,
                    "clip_checkpoint": str(Path(clip_checkpoint).expanduser()),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                output_path,
            )
    print(f"Saved best validation checkpoint: {output_path}")


def _metrics(rankings: torch.Tensor, gallery_pids: torch.Tensor, query_pids: torch.Tensor):
    matches = gallery_pids[rankings].eq(query_pids[:, None])
    metrics = {}
    for k in (1, 5, 10):
        metrics[f"R@{k}"] = (
            matches[:, : min(k, matches.shape[1])].any(dim=1).float().mean().item() * 100
        )
    aps = []
    for row in matches:
        relevant = row.nonzero(as_tuple=False).flatten()
        precision = torch.arange(1, len(relevant) + 1, dtype=torch.float32) / (relevant + 1)
        aps.append(precision.mean().item() if len(relevant) else 0.0)
    metrics["mAP"] = sum(aps) / max(len(aps), 1) * 100
    return metrics, matches, aps


@torch.no_grad()
def evaluate_reranked(
    root: str | Path,
    clip_checkpoint: str | Path,
    reranker_checkpoint: str | Path,
    top_m: int = 100,
    batch_size: int = 32,
    device: str | None = None,
    split: str = "test",
    per_query_output: str | Path | None = None,
    model_name: str = MODEL_NAME,
    image_size: tuple[int, int] = (384, 128),
    image_preprocess: str = "open_clip",
):
    records = load_records(root, split)
    root = Path(root).expanduser()
    image_paths, gallery_pids, captions, query_pids = [], [], [], []
    for record in records:
        image_paths.append(root / "imgs" / record["img_path"])
        gallery_pids.append(int(record["id"]))
        for caption in record["captions"]:
            captions.append(str(caption))
            query_pids.append(int(record["id"]))
    model, preprocess, tokenizer, device = load_model(
        device=device,
        checkpoint=clip_checkpoint,
        model_name=model_name,
        image_size=image_size,
        image_preprocess=image_preprocess,
    )
    matcher, matcher_metadata = load_reranker(str(reranker_checkpoint), device)
    if matcher_metadata.get("clip_model_name", model_name) != model_name:
        raise ValueError("Reranker and CLIP checkpoint use different model architectures")
    if tuple(matcher_metadata.get("image_size", image_size)) != tuple(image_size):
        raise ValueError("Evaluation image size differs from the reranker's training configuration")
    if matcher_metadata.get("image_preprocess", image_preprocess) != image_preprocess:
        raise ValueError("Evaluation preprocessing differs from the reranker's training configuration")
    encoder_state = matcher_metadata.pop("encoder_state_dict", None)
    if matcher_metadata.get("encoder_finetuned", False) and encoder_state is None:
        raise ValueError("Fine-tuned reranker checkpoint is missing its encoder weights")
    if encoder_state is not None:
        model.load_state_dict(encoder_state, strict=True)
        del encoder_state
        print("Restored fine-tuned encoder from the reranker checkpoint.")
    model.eval()

    image_features = []
    image_tokens, image_masks = [], []
    for start in range(0, len(image_paths), batch_size):
        batch_images = []
        for path in image_paths[start : start + batch_size]:
            with Image.open(path) as image:
                batch_images.append(image.convert("RGB"))
        image_features.append(encode_images(model, preprocess, batch_images, device).cpu())
        tokens, mask = encode_image_tokens(model, preprocess, batch_images, device)
        image_tokens.append(tokens.cpu())
        image_masks.append(mask.cpu())
    image_features = torch.cat(image_features)
    image_tokens = torch.cat(image_tokens)
    image_masks = torch.cat(image_masks)

    text_features, text_tokens, text_masks = [], [], []
    for start in range(0, len(captions), batch_size):
        batch_texts = captions[start : start + batch_size]
        text_features.append(encode_text(model, tokenizer, batch_texts, device).cpu())
        tokens, mask = encode_text_tokens(model, tokenizer, batch_texts, device)
        text_tokens.append(tokens.cpu())
        text_masks.append(mask.cpu())
    text_features = torch.cat(text_features)
    text_tokens = torch.cat(text_tokens)
    text_masks = torch.cat(text_masks)
    similarities = text_features @ image_features.T
    baseline_rankings = similarities.argsort(dim=1, descending=True)
    gallery_pids_tensor = torch.tensor(gallery_pids)
    query_pids_tensor = torch.tensor(query_pids)
    baseline_metrics, baseline_matches, baseline_aps = _metrics(
        baseline_rankings, gallery_pids_tensor, query_pids_tensor
    )

    candidate_count = len(gallery_pids) if top_m <= 0 else min(top_m, len(gallery_pids))
    baseline_metrics[f"R@{candidate_count}"] = (
        baseline_matches[:, :candidate_count].any(dim=1).float().mean().item() * 100
    )
    reranked_rankings = baseline_rankings.clone()
    matcher.eval()
    with torch.no_grad():
        for query_index in range(len(captions)):
            candidates = baseline_rankings[query_index, :candidate_count]
            pair_scores = []
            for start in range(0, candidate_count, batch_size):
                image_ids = candidates[start : start + batch_size]
                count = len(image_ids)
                scores = matcher(
                    image_tokens[image_ids].to(device),
                    image_masks[image_ids].to(device),
                    text_tokens[query_index : query_index + 1]
                    .expand(count, -1, -1)
                    .to(device),
                    text_masks[query_index : query_index + 1]
                    .expand(count, -1)
                    .to(device),
                )
                pair_scores.append(scores.cpu())
            local_order = torch.cat(pair_scores).argsort(descending=True)
            reordered_candidates = candidates[local_order]
            reranked_rankings[query_index, :candidate_count] = reordered_candidates

    reranked_metrics, reranked_matches, reranked_aps = _metrics(
        reranked_rankings, gallery_pids_tensor, query_pids_tensor
    )
    reranked_metrics[f"R@{candidate_count}"] = (
        reranked_matches[:, :candidate_count].any(dim=1).float().mean().item() * 100
    )
    harmed = {}
    for k in (1, 5, 10):
        before = baseline_matches[:, : min(k, baseline_matches.shape[1])].any(dim=1)
        after = reranked_matches[:, : min(k, reranked_matches.shape[1])].any(dim=1)
        harmed[f"R@{k}"] = int((before & ~after).sum().item())
    ap_delta = torch.tensor(reranked_aps) - torch.tensor(baseline_aps)
    summary = {
        "split": split,
        "encoder_source": "reranker_checkpoint" if matcher_metadata.get("encoder_finetuned", False) else "clip_checkpoint",
        "baseline_description": "Global retrieval using the same encoder as the reranker",
        "num_gallery_images": len(gallery_pids),
        "num_queries": len(captions),
        "rerank_top_m": candidate_count,
        "baseline": baseline_metrics,
        "reranked": reranked_metrics,
        "delta": {
            key: reranked_metrics[key] - baseline_metrics[key] for key in baseline_metrics
        },
        "queries_harmed": harmed,
        "queries_with_lower_ap": int((ap_delta < 0).sum().item()),
    }
    if per_query_output:
        output = Path(per_query_output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["query_index", "pid", "baseline_top1_correct", "reranked_top1_correct", "baseline_ap", "reranked_ap"]
            )
            for index, pid in enumerate(query_pids):
                writer.writerow(
                    [
                        index,
                        pid,
                        bool(baseline_matches[index, 0]),
                        bool(reranked_matches[index, 0]),
                        f"{baseline_aps[index]:.6f}",
                        f"{reranked_aps[index]:.6f}",
                    ]
                )
    return summary
