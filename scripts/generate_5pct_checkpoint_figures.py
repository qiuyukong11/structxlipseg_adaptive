from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset

from datasets.json_refseg_dataset import JsonRefSegDataset
from trainers import build_structxlip
from utils.main_utils import load_cfg_from_cfg_file


RUNS = {
    "structxlip_sketchy_5pct_no_3loss": {
        "config": "configs/structxlip_sketchy_5pct_no_3loss.yaml",
        "label": "StructXLIP no 3 losses",
    },
    "structxlip_sketchy_5pct_with_3loss": {
        "config": "configs/structxlip_sketchy_5pct_with_3loss.yaml",
        "label": "StructXLIP with 3 losses",
    },
}

RESULT_NAME = "CLIPSeg_structxlip_ViT-B-16_best_dice"
COMPONENT_LOSSES = [
    "seg_bce_loss",
    "seg_dice_loss",
    "clip_aux_loss",
    "loss_s_t",
    "loss_rs",
    "loss_chunk_align",
]
EPOCH_RE = re.compile(
    r"EPOCH:\s*(?P<epoch>\d+)\s*\|\s*Training Loss:\s*(?P<train>[0-9.]+)\s*\|\s*Validation Loss:\s*(?P<val>[0-9.]+)"
)
DICE_RE = re.compile(r"(?:Dice:\s*|New best Dice:\s*[-0-9.]+\s*.\s*)(?P<dice>[0-9.]+)")


@dataclass
class RunPaths:
    name: str
    label: str
    config: Path
    log: Path
    checkpoint: Path
    manifest: Path
    out_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate analysis figures for the two 5pct StructXLIP Sketchy checkpoints.")
    parser.add_argument("--output-dir", default="output/analysis_5pct_checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--test-json", default="/mnt/data/zruan/kqy/pami/segmentation/SKETCHY_test_instance_mask.json")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--max-embedding-samples", type=int, default=1200)
    parser.add_argument("--max-vis-per-group", type=int, default=4)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    return parser.parse_args()


def device_from_arg(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; using CPU.")
        return "cpu"
    return value


def make_run_paths(run_name: str, run_meta: dict, seed: int, root: Path) -> RunPaths:
    run_root = Path("output") / run_name
    return RunPaths(
        name=run_name,
        label=run_meta["label"],
        config=Path(run_meta["config"]),
        log=run_root / "trained_models" / f"seed{seed}" / "log.txt",
        checkpoint=run_root / "trained_models" / f"seed{seed}" / f"{RESULT_NAME}.pth",
        manifest=run_root / "seg_results" / f"seed{seed}" / RESULT_NAME / "manifest.csv",
        out_dir=root / run_name,
    )


def read_training_history(log_path: Path) -> list[dict]:
    entries: list[dict] = []
    current: dict | None = None
    with log_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                if current is not None:
                    entries.append(current)
                current = {
                    "epoch": int(epoch_match.group("epoch")),
                    "train_loss": float(epoch_match.group("train")),
                    "val_loss": float(epoch_match.group("val")),
                    "val_dice": math.nan,
                }
                continue

            dice_match = DICE_RE.search(line)
            if dice_match and current is not None:
                current["val_dice"] = float(dice_match.group("dice"))
                entries.append(current)
                current = None

    if current is not None:
        entries.append(current)

    by_epoch: dict[int, dict] = {}
    for entry in entries:
        by_epoch[entry["epoch"]] = entry
    return [by_epoch[k] for k in sorted(by_epoch)]


def write_training_csv(history: list[dict], path: Path) -> None:
    fieldnames = ["epoch", "train_loss", "val_loss", "val_dice", *COMPONENT_LOSSES]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            out = dict(row)
            for name in COMPONENT_LOSSES:
                out[name] = ""
            writer.writerow(out)


def plot_training_history(run: RunPaths, history: list[dict]) -> None:
    epochs = [x["epoch"] for x in history]
    train_loss = [x["train_loss"] for x in history]
    val_loss = [x["val_loss"] for x in history]
    val_dice = [x["val_dice"] for x in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=180)
    axes[0].plot(epochs, train_loss, marker="o", linewidth=1.8, label="train total loss")
    axes[0].plot(epochs, val_loss, marker="s", linewidth=1.8, label="val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title(run.label)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=True)

    axes[1].plot(epochs, val_dice, marker="o", color="#2271b2", linewidth=1.8, label="val Dice")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_ylim(0, max(0.45, max([x for x in val_dice if not math.isnan(x)], default=0.0) + 0.05))
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=True)
    axes[1].text(
        0.02,
        0.02,
        "Component losses were not logged in the historical training run.\n"
        "CSV keeps component columns empty instead of reconstructing them.",
        transform=axes[1].transAxes,
        fontsize=8,
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.9),
    )

    fig.tight_layout()
    fig.savefig(run.out_dir / "training_curves.png")
    plt.close(fig)


def plot_training_comparison(histories: dict[str, list[dict]], root: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=180)
    colors = {
        "structxlip_sketchy_5pct_no_3loss": "#3066be",
        "structxlip_sketchy_5pct_with_3loss": "#d1495b",
    }
    for run_name, history in histories.items():
        label = RUNS[run_name]["label"]
        epochs = [x["epoch"] for x in history]
        axes[0].plot(epochs, [x["train_loss"] for x in history], marker="o", linewidth=1.6, color=colors[run_name], label=label)
        axes[1].plot(epochs, [x["val_dice"] for x in history], marker="o", linewidth=1.6, color=colors[run_name], label=label)
    axes[0].set_title("Train total loss")
    axes[1].set_title("Validation Dice")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=True)
    axes[0].set_ylabel("loss")
    axes[1].set_ylabel("Dice")
    fig.tight_layout()
    fig.savefig(root / "training_curves_comparison.png")
    plt.close(fig)


def cfg_get(node, name: str, default=None):
    try:
        return getattr(node, name)
    except AttributeError:
        return default


def build_test_dataset(cfg, test_json: str) -> JsonRefSegDataset:
    struct_cfg = cfg_get(cfg, "STRUCTXLIP", None)
    return JsonRefSegDataset(
        test_json,
        train=False,
        data_root=cfg_get(cfg.DATASET, "DATA_ROOT", ""),
        image_size=int(cfg.DATASET.SIZE),
        hflip_prob=0.0,
        min_similarity=cfg_get(cfg.DATASET, "MIN_SIMILARITY", None),
        use_original_caption_prefix=bool(cfg_get(cfg.DATASET, "USE_ORIGINAL_CAPTION_PREFIX", False)),
        structure_image_field=cfg_get(struct_cfg, "STRUCTURE_IMAGE_FIELD", "filename_canny"),
        chunk_top_k=int(cfg_get(struct_cfg, "CHUNK_TOP_K", 3)),
        load_aux_images=False,
    )


def sample_indices(total: int, max_items: int, seed: int) -> list[int]:
    if max_items <= 0 or total <= max_items:
        return list(range(total))
    rng = random.Random(seed)
    return sorted(rng.sample(range(total), max_items))


def load_model(run: RunPaths, device: str):
    cfg = load_cfg_from_cfg_file(str(run.config))
    cfg.MODEL.DEVICE = device
    model = build_structxlip(cfg)
    checkpoint = torch.load(run.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)
    return cfg, model


def compute_embeddings(run: RunPaths, test_json: str, device: str, batch_size: int, max_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cfg, model = load_model(run, device)
    dataset = build_test_dataset(cfg, test_json)
    indices = sample_indices(len(dataset), max_samples, seed)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    image_embeds = []
    text_embeds = []
    texts: list[str] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            images = batch["image"].to(device)
            text = list(batch["text_prompt"])
            tokenized = model._tokenize_texts(text, device)
            prompts = model.clip_model.token_embedding(tokenized).type(model.dtype)
            _, text_features, cls_features = model.encode_text_image(
                tokenized,
                prompts,
                images,
                return_image_embed=True,
            )
            image_embeds.append(F.normalize(cls_features.float(), dim=-1).cpu())
            text_embeds.append(F.normalize(text_features.float(), dim=-1).cpu())
            texts.extend(text)
            if batch_idx % 10 == 0:
                print(f"[embedding] {run.name}: processed {min(batch_idx * batch_size, len(indices))}/{len(indices)}")

    return torch.cat(image_embeds).numpy(), torch.cat(text_embeds).numpy(), texts


def reduce_2d(points: np.ndarray, perplexity: float, seed: int) -> tuple[np.ndarray, str]:
    try:
        from sklearn.manifold import TSNE

        safe_perplexity = min(float(perplexity), max(5.0, (len(points) - 1) / 3.0))
        try:
            reducer = TSNE(
                n_components=2,
                perplexity=safe_perplexity,
                init="pca",
                learning_rate="auto",
                random_state=seed,
                max_iter=1000,
            )
        except TypeError:
            reducer = TSNE(
                n_components=2,
                perplexity=safe_perplexity,
                init="pca",
                learning_rate="auto",
                random_state=seed,
                n_iter=1000,
            )
        return reducer.fit_transform(points), "t-SNE"
    except Exception as exc:
        print(f"[warn] t-SNE failed ({exc}); falling back to PCA.")
        points = points - points.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(points, full_matrices=False)
        return points @ vt[:2].T, "PCA fallback"


def retrieval_metrics(image_embeds: np.ndarray, text_embeds: np.ndarray) -> dict:
    sim = image_embeds @ text_embeds.T
    target = np.arange(sim.shape[0])
    return {
        "mean_cos": float(np.diag(sim).mean()),
        "i2t_r1": float((sim.argmax(axis=1) == target).mean() * 100.0),
        "t2i_r1": float((sim.argmax(axis=0) == target).mean() * 100.0),
        "n": int(sim.shape[0]),
    }


def plot_embedding_space(run: RunPaths, image_embeds: np.ndarray, text_embeds: np.ndarray, seed: int, perplexity: float) -> dict:
    metrics = retrieval_metrics(image_embeds, text_embeds)
    points = np.concatenate([image_embeds, text_embeds], axis=0)
    coords, method = reduce_2d(points, perplexity, seed)
    n = image_embeds.shape[0]
    img_xy = coords[:n]
    txt_xy = coords[n:]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    for i in range(n):
        ax.plot([txt_xy[i, 0], img_xy[i, 0]], [txt_xy[i, 1], img_xy[i, 1]], color="0.75", alpha=0.08, linewidth=0.5)

    ax.scatter(img_xy[:, 0], img_xy[:, 1], s=10, alpha=0.8, c="#1f77b4", label="Instance")
    ax.scatter(txt_xy[:, 0], txt_xy[:, 1], s=12, alpha=0.72, c="#e53935", marker="^", label="Text")

    mean_img = img_xy.mean(axis=0)
    mean_txt = txt_xy.mean(axis=0)
    ax.scatter([mean_txt[0]], [mean_txt[1]], s=95, c="#e53935", edgecolors="black", marker="^", label="Mean text", zorder=5)
    ax.scatter([mean_img[0]], [mean_img[1]], s=95, c="#1f77b4", edgecolors="black", label="Mean instance", zorder=5)

    ax.set_title(f"{run.label} checkpoint embeddings", fontsize=14, weight="bold")
    ax.set_xlabel(f"{method} dim 1")
    ax.set_ylabel(f"{method} dim 2")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="lower right", frameon=True)
    metric_text = (
        f"mean cos: {metrics['mean_cos']:.4f}\n"
        f"N = {metrics['n']}\n"
        f"T2I R@1: {metrics['t2i_r1']:.2f}\n"
        f"I2T R@1: {metrics['i2t_r1']:.2f}"
    )
    ax.text(
        0.02,
        0.98,
        metric_text,
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="0.65", alpha=0.92),
    )

    fig.tight_layout()
    fig.savefig(run.out_dir / "embedding_space.png")
    plt.close(fig)

    with (run.out_dir / "embedding_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({**metrics, "projection": method}, handle, indent=2)
    return {**metrics, "projection": method}


def dice_score(pred_path: Path, gt_path: Path) -> float:
    pred = np.array(Image.open(pred_path).convert("L")) > 127
    gt = np.array(Image.open(gt_path).convert("L")) > 127
    if pred.shape != gt.shape:
        pred = np.array(Image.fromarray(pred.astype(np.uint8) * 255).resize((gt.shape[1], gt.shape[0]), Image.Resampling.NEAREST)) > 127
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def read_manifest_scores(run: RunPaths) -> list[dict]:
    rows: list[dict] = []
    with run.manifest.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pred_path = Path(row["prediction_path"])
            gt_path = Path(row["gt_path"])
            if not pred_path.exists() or not gt_path.exists():
                continue
            row["dice"] = dice_score(pred_path, gt_path)
            rows.append(row)
    rows.sort(key=lambda x: x["dice"], reverse=True)
    with (run.out_dir / "segmentation_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else ["dice"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def load_rgb(path: str, size: int = 224) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return np.array(image).astype(np.float32) / 255.0


def load_mask(path: str, size: int = 224) -> np.ndarray:
    image = Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST)
    return np.array(image) > 127


def transparent_overlay(image: np.ndarray, mask: np.ndarray, color: tuple[float, float, float], alpha: float = 0.42) -> np.ndarray:
    out = image.copy()
    color_arr = np.array(color, dtype=np.float32)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color_arr
    return np.clip(out, 0.0, 1.0)


def short_text(text: str, width: int = 52) -> str:
    return "\n".join(textwrap.wrap(text, width=width, max_lines=2, placeholder="..."))


def plot_segmentation_examples(run: RunPaths, rows: list[dict], per_group: int) -> None:
    if not rows:
        raise RuntimeError(f"No segmentation rows found for {run.name}")
    top = rows[:per_group]
    bottom = list(reversed(rows[-per_group:]))
    selected = [("success", row) for row in top] + [("failure", row) for row in bottom]

    fig = plt.figure(figsize=(9, 2.95 * len(selected)), dpi=180)
    height_ratios = []
    for _ in selected:
        height_ratios.extend([1.0, 0.18])
    grid = fig.add_gridspec(
        nrows=len(selected) * 2,
        ncols=3,
        height_ratios=height_ratios,
        hspace=0.08,
        wspace=0.03,
    )

    for row_idx, (kind, row) in enumerate(selected):
        image = load_rgb(row["image_path"])
        pred = load_mask(row["prediction_path"])
        gt = load_mask(row["gt_path"])
        panels = [
            image,
            transparent_overlay(image, gt, (0.0, 0.72, 0.22), alpha=0.42),
            transparent_overlay(image, pred, (0.95, 0.08, 0.06), alpha=0.42),
        ]
        titles = ["Original", "GT overlay", "Prediction overlay"]
        for col_idx, panel in enumerate(panels):
            ax = fig.add_subplot(grid[row_idx * 2, col_idx])
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(titles[col_idx], fontsize=9)

        text_ax = fig.add_subplot(grid[row_idx * 2 + 1, :])
        text_ax.axis("off")
        caption = short_text(row["text_prompt"], 110)
        text_ax.text(
            0.5,
            0.5,
            f"{kind} | Dice {row['dice']:.4f} | {caption}",
            ha="center",
            va="center",
            fontsize=8,
        )

    fig.suptitle(f"{run.label}: segmentation success/failure examples", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(run.out_dir / "seg_success_failure.png")
    plt.close(fig)


def check_inputs(run: RunPaths) -> None:
    missing = [path for path in [run.config, run.log, run.checkpoint, run.manifest] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{run.name} missing required files: {missing}")


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    device = device_from_arg(args.device)
    print(f"[info] device={device}")

    histories: dict[str, list[dict]] = {}
    summary: dict[str, dict] = {}

    for run_name, run_meta in RUNS.items():
        run = make_run_paths(run_name, run_meta, args.seed, root)
        check_inputs(run)
        run.out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[info] plotting training curves for {run.name}")
        history = read_training_history(run.log)
        if not history:
            raise RuntimeError(f"No epoch metrics parsed from {run.log}")
        histories[run_name] = history
        write_training_csv(history, run.out_dir / "training_curves.csv")
        plot_training_history(run, history)

        print(f"[info] scoring and visualizing segmentation examples for {run.name}")
        rows = read_manifest_scores(run)
        plot_segmentation_examples(run, rows, args.max_vis_per_group)

        print(f"[info] computing embedding space for {run.name}")
        image_embeds, text_embeds, _ = compute_embeddings(
            run,
            args.test_json,
            device,
            args.embedding_batch_size,
            args.max_embedding_samples,
            args.seed,
        )
        embedding_metrics = plot_embedding_space(run, image_embeds, text_embeds, args.seed, args.tsne_perplexity)
        summary[run.name] = {
            "checkpoint": str(run.checkpoint),
            "log": str(run.log),
            "manifest": str(run.manifest),
            "figures": {
                "training_curves": str(run.out_dir / "training_curves.png"),
                "embedding_space": str(run.out_dir / "embedding_space.png"),
                "seg_success_failure": str(run.out_dir / "seg_success_failure.png"),
            },
            "embedding_metrics": embedding_metrics,
            "segmentation_dice": {
                "samples": len(rows),
                "mean": float(np.mean([row["dice"] for row in rows])),
                "median": float(np.median([row["dice"] for row in rows])),
                "best": float(rows[0]["dice"]),
                "worst": float(rows[-1]["dice"]),
            },
            "loss_note": "Historical training logs contain total train loss, validation loss, and validation Dice only; component loss curves were not logged.",
        }

    plot_training_comparison(histories, root)
    with (root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[done] wrote figures and summary to {root}")


if __name__ == "__main__":
    main()
