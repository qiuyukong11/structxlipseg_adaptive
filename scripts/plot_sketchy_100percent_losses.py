#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


EPOCH_PATTERN = re.compile(r"EPOCH:\s*(?P<epoch>\d+)\s*\|\s*(?P<body>.*)")
METRIC_PATTERN = re.compile(r"([^|:]+):\s*([-+0-9.eE]+)")


LOSS_GROUPS = [
    (
        "total_main_losses",
        "Total / Main Training Losses",
        ["train_total", "train_seg", "train_weightedclip", "train_weightedstructxlip"],
    ),
    (
        "segmentation_losses",
        "Training Segmentation Loss Components",
        ["train_bce", "train_diceloss"],
    ),
    (
        "clip_losses",
        "CLIP Auxiliary Losses",
        ["train_clip"],
    ),
    (
        "structxlip_aux_losses",
        "StructXLIP Auxiliary Losses",
        ["train_loss_st", "train_loss_rs", "train_loss_chunk_align"],
    ),
]

STALE_PLOTS = ["validation_metric.png"]


def parse_log(log_path: Path) -> pd.DataFrame:
    rows = []
    for line in log_path.read_text(errors="ignore").splitlines():
        match = EPOCH_PATTERN.search(line)
        if not match:
            continue
        row = {"epoch": int(match.group("epoch"))}
        for key, value in METRIC_PATTERN.findall(match.group("body")):
            normalized_key = key.strip().lower().replace(" ", "_")
            row[normalized_key] = float(value)
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No epoch rows parsed from {log_path}")
    return pd.DataFrame(rows).sort_values("epoch")


def pretty_label(column: str) -> str:
    labels = {
        "train_total": "Train Total",
        "train_seg": "Train Seg",
        "train_bce": "Train BCE",
        "train_diceloss": "Train DiceLoss",
        "train_clip": "Train CLIP",
        "train_weightedclip": "Train WeightedCLIP",
        "train_loss_st": "Train loss_st",
        "train_loss_rs": "Train loss_rs",
        "train_loss_chunk_align": "Train loss_chunk_align",
        "train_weightedstructxlip": "Train WeightedStructXLIP",
        "val_total": "Val Total",
        "val_bce": "Val BCE",
        "val_diceloss": "Val DiceLoss",
        "val_dicemetric": "Val DiceMetric",
    }
    return labels.get(column, column)


def plot_columns(df: pd.DataFrame, columns: list[str], title: str, path: Path) -> bool:
    available = [col for col in columns if col in df.columns and df[col].notna().any()]
    if not available:
        return False
    fig, ax = plt.subplots(figsize=(9.5, 5.4), constrained_layout=True)
    x = df["epoch"]
    for col in available:
        ax.plot(x, df[col], marker="o", linewidth=1.9, markersize=4.5, label=pretty_label(col))
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=10)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def plot_summary(df: pd.DataFrame, run_name: str, output_dir: Path) -> Path:
    panels = [(title, cols) for _, title, cols in LOSS_GROUPS if any(col in df.columns for col in cols)]
    ncols = 2
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.6 * nrows), constrained_layout=True)
    axes = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]
    x = df["epoch"]

    for ax, (title, cols) in zip(axes, panels):
        for col in cols:
            if col in df.columns and df[col].notna().any():
                ax.plot(x, df[col], marker="o", linewidth=1.8, markersize=4, label=pretty_label(col))
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Value", fontsize=11)
        ax.grid(True, alpha=0.28)
        ax.legend(fontsize=8)

    for ax in axes[len(panels) :]:
        ax.axis("off")

    fig.suptitle(f"{run_name} Loss Curves", fontsize=16)
    path = output_dir / "losses_all.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_run(run_root: Path, seed: str) -> list[Path]:
    log_path = run_root / "trained_models" / seed / "log.txt"
    if not log_path.is_file():
        raise FileNotFoundError(log_path)

    out_dir = run_root / "loss_plots" / seed
    out_dir.mkdir(parents=True, exist_ok=True)
    df = parse_log(log_path)
    df = df[[col for col in df.columns if col == "epoch" or col.startswith("train_")]]
    if {"train_loss_st", "train_loss_rs", "train_loss_chunk_align"}.issubset(df.columns):
        df["train_weightedstructxlip"] = (
            0.25 * df["train_loss_st"]
            + 0.02 * df["train_loss_rs"]
            + 0.07 * df["train_loss_chunk_align"]
        )
    for stale_name in STALE_PLOTS:
        stale_path = out_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    csv_path = out_dir / "losses.csv"
    df.to_csv(csv_path, index=False)

    paths = [csv_path, plot_summary(df, run_root.name, out_dir)]
    for filename, title, columns in LOSS_GROUPS:
        path = out_dir / f"{filename}.png"
        if plot_columns(df, columns, title, path):
            paths.append(path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-roots",
        nargs="+",
        type=Path,
        default=[
            Path("output/sketchy_clipseg_100percent"),
            Path("output/sketchy_structxlipseg_100percent"),
        ],
    )
    parser.add_argument("--seed", default="seed42")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for run_root in args.run_roots:
        paths = plot_run(run_root, args.seed)
        print(f"{run_root}:")
        for path in paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
