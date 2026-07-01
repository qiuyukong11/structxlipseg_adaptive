#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from clip import clip
from test import build_json_refseg_test_dataset, build_model, result_name
from utils.main_utils import load_cfg_from_cfg_file


@dataclass
class RunSpec:
    name: str
    label: str
    root: Path
    config: Path
    color_image: str
    color_text: str


RUNS = [
    RunSpec(
        name="clipseg",
        label="CLIPSeg latest",
        root=Path("output/sketchy_clipseg_100percent"),
        config=Path("configs/sketchy_clipseg_100percent.yaml"),
        color_image="#2166ac",
        color_text="#b2182b",
    ),
    RunSpec(
        name="structxlipseg",
        label="StructXLIPSeg latest",
        root=Path("output/sketchy_structxlipseg_100percent"),
        config=Path("configs/sketchy_structxlipseg_100percent.yaml"),
        color_image="#2166ac",
        color_text="#b2182b",
    ),
]


def device_from_arg(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return value


def sample_indices(total: int, max_samples: int, seed: int) -> list[int]:
    if max_samples <= 0 or max_samples >= total:
        return list(range(total))
    rng = random.Random(seed)
    return sorted(rng.sample(range(total), max_samples))


def checkpoint_path(run: RunSpec, cfg, seed: int) -> Path:
    ckpt_name = f"{run.root.name}_{result_name(cfg)}_latest.pth"
    return run.root / "trained_models" / f"seed{seed}" / ckpt_name


def load_run_model(run: RunSpec, device: str, seed: int):
    cfg = load_cfg_from_cfg_file(str(run.config))
    cfg.MODEL.DEVICE = device
    model = build_model(cfg)
    path = checkpoint_path(run, cfg, seed)
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)
    return cfg, model, path


def tokenize(model, texts: list[str], device: str) -> torch.Tensor:
    if hasattr(model, "_tokenize_texts"):
        return model._tokenize_texts(texts, device)
    return torch.cat([
        clip.tokenize(t, context_length=model.context_length, truncate=True) for t in texts
    ]).to(device)


def encode_batch(model, images: torch.Tensor, texts: list[str], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    tokenized = tokenize(model, texts, device)
    with torch.no_grad():
        prompts = model.clip_model.token_embedding(tokenized).type(model.dtype)
        image_features, text_features = model.encode_text_image(tokenized, prompts, images)

    patch_features = F.normalize(image_features.float(), dim=-1, eps=1e-6)
    image_embeds = F.normalize(patch_features.mean(dim=1), dim=-1, eps=1e-6)
    text_embeds = F.normalize(text_features.float(), dim=-1, eps=1e-6)
    return image_embeds, text_embeds


def compute_embeddings(
    run: RunSpec,
    device: str,
    seed: int,
    batch_size: int,
    indices: list[int] | None,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    cfg, model, ckpt = load_run_model(run, device, seed)
    dataset = build_json_refseg_test_dataset(cfg)
    if indices is None:
        indices = list(range(len(dataset)))
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    image_embeds = []
    text_embeds = []
    texts: list[str] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Embedding {run.label}"):
            images = batch["image"].to(device, non_blocking=True)
            batch_texts = list(batch["text_prompt"])
            img, txt = encode_batch(model, images, batch_texts, device)
            image_embeds.append(img.cpu())
            text_embeds.append(txt.cpu())
            texts.extend(batch_texts)

    image_np = torch.cat(image_embeds, dim=0).numpy()
    text_np = torch.cat(text_embeds, dim=0).numpy()
    similarities = (image_np * text_np).sum(axis=1)
    meta = {
        "checkpoint": str(ckpt),
        "config": str(run.config),
        "num_pairs": int(len(indices)),
        "mean_pair_cosine": float(similarities.mean()),
        "median_pair_cosine": float(np.median(similarities)),
        "std_pair_cosine": float(similarities.std()),
    }
    return image_np, text_np, texts, meta


def reduce_tsne(points: np.ndarray, seed: int, perplexity: float, max_iter: int) -> tuple[np.ndarray, str, float]:
    from sklearn.manifold import TSNE

    safe_perplexity = min(float(perplexity), max(5.0, (len(points) - 1) / 3.0))
    kwargs = dict(
        n_components=2,
        perplexity=safe_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    try:
        reducer = TSNE(**kwargs, max_iter=max_iter)
    except TypeError:
        reducer = TSNE(**kwargs, n_iter=max_iter)
    return reducer.fit_transform(points), "t-SNE", safe_perplexity


def axis_limits(*arrays: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    points = np.concatenate(arrays, axis=0)
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    pad_x = max((x1 - x0) * 0.06, 1e-3)
    pad_y = max((y1 - y0) * 0.06, 1e-3)
    return (float(x0 - pad_x), float(x1 + pad_x)), (float(y0 - pad_y), float(y1 + pad_y))


def plot_run(
    run: RunSpec,
    image_xy: np.ndarray,
    text_xy: np.ndarray,
    metrics: dict,
    output_dir: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    line_alpha: float,
    show_legend: bool = False,
    max_lines: int = 500,
    rng: np.random.Generator | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=300)
    rng = rng or np.random.default_rng(42)
    n = image_xy.shape[0]
    n_lines = n if max_lines <= 0 else min(max_lines, n)
    for i in rng.choice(n, size=n_lines, replace=False):
        ax.plot(
            [image_xy[i, 0], text_xy[i, 0]],
            [image_xy[i, 1], text_xy[i, 1]],
            color="0.55",
            alpha=line_alpha,
            linewidth=0.3,
            zorder=1,
        )

    ax.scatter(image_xy[:, 0], image_xy[:, 1], s=8, c="tab:blue", alpha=0.65, marker="o", linewidths=0, label="Image", zorder=2)
    ax.scatter(text_xy[:, 0], text_xy[:, 1], s=8, c="tab:red", marker="^", alpha=0.65, linewidths=0, label="Text", zorder=3)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(run.label, fontsize=14, fontweight="bold")
    xticks = np.linspace(xlim[0], xlim[1], 5)
    yticks = np.linspace(ylim[0], ylim[1], 5)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{x:.1f}" for x in xticks], fontsize=8)
    ax.set_yticklabels([f"{y:.1f}" for y in yticks], fontsize=8)
    ax.set_xlabel("t-SNE dim 1", fontsize=10)
    ax.set_ylabel("t-SNE dim 2", fontsize=10)
    ax.grid(True, alpha=0.22, linewidth=0.5)
    if show_legend:
        ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    path = output_dir / f"{run.name}_latest_embedding_tsne.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_pair_csv(path: Path, texts: list[str], image_xy: np.ndarray, text_xy: np.ndarray, similarities: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "similarity", "image_x", "image_y", "text_x", "text_y", "text_prompt"],
        )
        writer.writeheader()
        for idx, text in enumerate(texts):
            writer.writerow(
                {
                    "index": idx,
                    "similarity": f"{similarities[idx]:.8f}",
                    "image_x": f"{image_xy[idx, 0]:.8f}",
                    "image_y": f"{image_xy[idx, 1]:.8f}",
                    "text_x": f"{text_xy[idx, 0]:.8f}",
                    "text_y": f"{text_xy[idx, 1]:.8f}",
                    "text_prompt": text,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot latest checkpoint image/text embedding t-SNE for Sketchy 100% runs.")
    parser.add_argument("--output-dir", type=Path, default=Path("output/sketchy_latest_embedding_tsne"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means use the full test set.")
    parser.add_argument("--perplexity", type=float, default=40.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--line-alpha", type=float, default=0.035)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = device_from_arg(args.device)
    print(f"[info] device={device}")

    first_cfg = load_cfg_from_cfg_file(str(RUNS[0].config))
    first_dataset = build_json_refseg_test_dataset(first_cfg)
    indices = sample_indices(len(first_dataset), args.max_samples, args.seed)
    print(f"[info] using {len(indices)}/{len(first_dataset)} test pairs")

    embeddings = {}
    metrics = {}
    texts_by_run = {}
    for run in RUNS:
        image_embeds, text_embeds, texts, meta = compute_embeddings(run, device, args.seed, args.batch_size, indices)
        embeddings[run.name] = (image_embeds, text_embeds)
        metrics[run.name] = meta
        texts_by_run[run.name] = texts
        np.savez_compressed(args.output_dir / f"{run.name}_latest_embeddings.npz", image=image_embeds, text=text_embeds)

    combined = np.concatenate(
        [embeddings[run.name][0] for run in RUNS] + [embeddings[run.name][1] for run in RUNS],
        axis=0,
    )
    print(f"[info] fitting shared t-SNE on {combined.shape[0]} points")
    coords, method, safe_perplexity = reduce_tsne(combined, args.seed, args.perplexity, args.max_iter)
    metrics["_projection"] = {
        "method": method,
        "perplexity": safe_perplexity,
        "max_iter": args.max_iter,
        "shared_fit": True,
    }

    n = len(indices)
    offset = 0
    coords_by_run = {}
    for run in RUNS:
        img_xy = coords[offset : offset + n]
        offset += n
        coords_by_run.setdefault(run.name, {})["image"] = img_xy
    for run in RUNS:
        txt_xy = coords[offset : offset + n]
        offset += n
        coords_by_run.setdefault(run.name, {})["text"] = txt_xy

    xlim, ylim = axis_limits(*[coords_by_run[run.name]["image"] for run in RUNS], *[coords_by_run[run.name]["text"] for run in RUNS])
    metrics["_axis_limits"] = {"xlim": xlim, "ylim": ylim}

    plot_paths = []
    for run in RUNS:
        img_xy = coords_by_run[run.name]["image"]
        txt_xy = coords_by_run[run.name]["text"]
        image_embeds, text_embeds = embeddings[run.name]
        sims = (image_embeds * text_embeds).sum(axis=1)
        save_pair_csv(args.output_dir / f"{run.name}_latest_tsne_points.csv", texts_by_run[run.name], img_xy, txt_xy, sims)
        plot_paths.append(plot_run(
            run,
            img_xy,
            txt_xy,
            metrics[run.name],
            args.output_dir,
            xlim,
            ylim,
            args.line_alpha,
            show_legend=(run == RUNS[0]),
            rng=np.random.default_rng(args.seed),
        ))

    with (args.output_dir / "embedding_tsne_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    for path in plot_paths:
        print(f"[saved] {path}")
    print(f"[saved] {args.output_dir / 'embedding_tsne_metrics.json'}")


if __name__ == "__main__":
    main()
