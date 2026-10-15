"""Command line interface for downloading, smoke-testing, and searching."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from .model import PRETRAINED, MODEL_NAME, encode_image, encode_images, encode_text, load_model
from .training import evaluate_reranked, train_reranker

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def command_download(_args: argparse.Namespace) -> int:
    print(f"Loading {MODEL_NAME} ({PRETRAINED}); first run downloads and caches weights.")
    _model, _preprocess, _tokenizer, device = load_model()
    print(f"Model ready on {device}.")
    return 0


def command_train_reranker(args: argparse.Namespace) -> int:
    train_reranker(
        root=args.root,
        clip_checkpoint=args.clip_checkpoint,
        output=args.output,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        seed=args.seed,
        model_name=args.clip_model,
        image_size=tuple(args.image_size),
    )
    return 0


def command_smoke(_args: argparse.Namespace) -> int:
    prompts = [
        "a photo of a person wearing a red jacket and black trousers",
        "a photo of a person wearing a white shirt and blue jeans",
    ]
    model, _preprocess, tokenizer, device = load_model()
    features = encode_text(model, tokenizer, prompts, device)
    similarity = (features @ features.T).cpu()
    print(f"Model: {MODEL_NAME} ({PRETRAINED}) | device: {device}")
    print(f"Text embedding shape: {tuple(features.shape)}")
    print(f"Embedding norms: {features.norm(dim=-1).cpu().tolist()}")
    print(f"Prompt-to-prompt cosine similarity: {similarity[0, 1].item():.4f}")
    return 0


def command_search(args: argparse.Namespace) -> int:
    image_dir = Path(args.images).expanduser()
    if not image_dir.is_dir():
        raise SystemExit(f"Image directory does not exist: {image_dir}")
    paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise SystemExit(f"No supported images found under: {image_dir}")

    model, preprocess, tokenizer, device = load_model()
    text_features = encode_text(model, tokenizer, [args.query], device)
    results: list[tuple[float, str]] = []
    skipped = 0
    for path in paths:
        try:
            with Image.open(path) as image:
                image_features = encode_image(model, preprocess, image, device)
            score = (text_features @ image_features.T).item()
            results.append((score, str(path)))
        except (OSError, UnidentifiedImageError) as exc:
            skipped += 1
            print(f"Skipping unreadable image {path}: {exc}")

    results.sort(reverse=True)
    top = results[: args.top_k]
    print(f"Query: {args.query}\nScanned: {len(paths)} | ranked: {len(results)} | skipped: {skipped}")
    for rank, (score, path) in enumerate(top, 1):
        print(f"{rank:>3}. {score:.4f}  {path}")

    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["rank", "cosine_similarity", "image_path"])
            writer.writerows((rank, f"{score:.6f}", path) for rank, (score, path) in enumerate(top, 1))
        print(f"Saved results: {output}")
    return 0


def command_evaluate_rstpreid(args: argparse.Namespace) -> int:
    if args.reranker_checkpoint:
        if not args.clip_checkpoint:
            raise SystemExit("--clip-checkpoint is required when --reranker-checkpoint is used")
        summary = evaluate_reranked(
            root=args.root,
            clip_checkpoint=args.clip_checkpoint,
            reranker_checkpoint=args.reranker_checkpoint,
            top_m=args.rerank_top_m,
            batch_size=args.batch_size,
            device=args.device,
            split=args.split,
            per_query_output=args.per_query_output,
            model_name=args.clip_model,
            image_size=tuple(args.image_size),
        )
        print(json.dumps(summary, indent=2))
        if args.output:
            output = Path(args.output).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"Saved metrics: {output}")
        return 0

    root = Path(args.root).expanduser()
    annotation_path = root / "data_captions.json"
    image_dir = root / "imgs"
    if not annotation_path.is_file() or not image_dir.is_dir():
        raise SystemExit(
            "Expected RSTPReid root containing data_captions.json and imgs/. "
            f"Not found under: {root}"
        )
    records = json.loads(annotation_path.read_text(encoding="utf-8"))
    test_records = [r for r in records if str(r.get("split", "")).lower() == args.split]
    if not test_records:
        raise SystemExit(f"No records with split={args.split!r} found in data_captions.json")

    image_paths: list[Path] = []
    image_pids: list[int] = []
    captions: list[str] = []
    caption_pids: list[int] = []
    for record in test_records:
        pid = int(record["id"])
        image_paths.append(image_dir / record["img_path"])
        image_pids.append(pid)
        for caption in record["captions"]:
            captions.append(str(caption))
            caption_pids.append(pid)
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} image files referenced by annotations are missing; first: {missing[0]}")

    model, preprocess, tokenizer, device = load_model(
        device=args.device,
        checkpoint=args.clip_checkpoint,
        model_name=args.clip_model,
        pretrained=PRETRAINED if args.clip_checkpoint is None else None,
        image_size=tuple(args.image_size),
    )
    image_features_list: list[torch.Tensor] = []
    batch_size = args.batch_size
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        with_images = []
        for path in batch_paths:
            with Image.open(path) as image:
                with_images.append(image.convert("RGB"))
        image_features_list.append(encode_images(model, preprocess, with_images, device).cpu())
    image_features = torch.cat(image_features_list)

    text_features_list: list[torch.Tensor] = []
    for start in range(0, len(captions), batch_size):
        text_features_list.append(
            encode_text(model, tokenizer, captions[start : start + batch_size], device).cpu()
        )
    text_features = torch.cat(text_features_list)
    similarities = text_features @ image_features.T
    ranking = similarities.argsort(dim=1, descending=True)
    gallery_pids = torch.tensor(image_pids)
    query_pids = torch.tensor(caption_pids)
    matches = gallery_pids[ranking].eq(query_pids[:, None])

    metrics: dict[str, float] = {}
    for k in (1, 5, 10):
        metrics[f"R@{k}"] = matches[:, : min(k, matches.shape[1])].any(dim=1).float().mean().item() * 100
    average_precisions = []
    for row in matches:
        relevant_ranks = row.nonzero(as_tuple=False).flatten()
        precisions = torch.arange(1, len(relevant_ranks) + 1, dtype=torch.float32) / (relevant_ranks + 1)
        average_precisions.append(precisions.mean().item() if len(relevant_ranks) else 0.0)
    metrics["mAP"] = sum(average_precisions) / len(average_precisions) * 100

    model_source = args.clip_checkpoint or PRETRAINED
    print(f"Model: {args.clip_model} ({model_source}) | device: {device}")
    print(f"{args.split.title()} identities: {len(set(image_pids))} | images: {len(image_pids)} | captions: {len(captions)}")
    for name, value in metrics.items():
        print(f"{name}: {value:.2f}%")
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Saved metrics: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="person-search", description="CLIP text-to-image person retrieval baseline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="download/cache model weights and verify model loading")
    download.set_defaults(handler=command_download)

    smoke = subparsers.add_parser("smoke", help="encode sample descriptions and check embeddings")
    smoke.set_defaults(handler=command_smoke)

    search = subparsers.add_parser("search", help="rank an image gallery by a text description")
    search.add_argument("--query", required=True, help="natural-language description of the target person")
    search.add_argument("--images", required=True, help="folder containing gallery images (searched recursively)")
    search.add_argument("--top-k", type=int, default=10, help="number of results to show")
    search.add_argument("--output", help="optional CSV path for ranked results")
    search.set_defaults(handler=command_search)

    evaluate = subparsers.add_parser("evaluate-rstpreid", help="evaluate zero-shot retrieval on the RSTPReid test split")
    evaluate.add_argument("--root", required=True, help="RSTPReid folder containing imgs/ and data_captions.json")
    evaluate.add_argument("--batch-size", type=int, default=32, help="inference batch size")
    evaluate.add_argument("--split", choices=("val", "test"), default="test", help="benchmark split to evaluate")
    evaluate.add_argument("--device", help="device such as cuda or cpu; defaults to CUDA when available")
    evaluate.add_argument("--clip-model", default=MODEL_NAME, help="OpenCLIP architecture matching the checkpoint")
    evaluate.add_argument("--image-size", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"), default=(384, 128), help="input image height and width; RSTPReid default: 384 128")
    evaluate.add_argument("--clip-checkpoint", help="fine-tuned CLIP .pth checkpoint; omit for the public LAION weights")
    evaluate.add_argument("--reranker-checkpoint", help="trained image-text matcher checkpoint; enables reranking")
    evaluate.add_argument("--rerank-top-m", type=int, default=100, help="rerank the CLIP top M; use 0 to rerank the full gallery")
    evaluate.add_argument("--per-query-output", help="optional CSV with per-query top-1/AP changes")
    evaluate.add_argument("--output", help="optional JSON path for metrics")
    evaluate.set_defaults(handler=command_evaluate_rstpreid)

    train = subparsers.add_parser("train-reranker-rstpreid", help="train a token-level image-caption matcher on RSTPReid train/val")
    train.add_argument("--root", required=True, help="RSTPReid folder containing imgs/ and data_captions.json")
    train.add_argument("--clip-checkpoint", required=True, help="fine-tuned CLIP checkpoint used by retrieval and matcher")
    train.add_argument("--clip-model", default=MODEL_NAME, help="OpenCLIP architecture matching the checkpoint")
    train.add_argument("--image-size", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"), default=(384, 128), help="input image height and width; RSTPReid default: 384 128")
    train.add_argument("--output", required=True, help="where to save the best validation reranker checkpoint")
    train.add_argument("--device", help="device such as cuda or cpu; defaults to CUDA when available")
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--seed", type=int, default=7)
    train.set_defaults(handler=command_train_reranker)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if getattr(args, "top_k", 1) < 1:
        raise SystemExit("--top-k must be at least 1")
    if getattr(args, "batch_size", 1) < 1:
        raise SystemExit("--batch-size must be at least 1")
    if getattr(args, "epochs", 1) < 1:
        raise SystemExit("--epochs must be at least 1")
    if getattr(args, "rerank_top_m", 1) < 0:
        raise SystemExit("--rerank-top-m must be 0 (full gallery) or positive")
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

