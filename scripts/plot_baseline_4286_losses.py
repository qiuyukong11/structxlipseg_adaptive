#!/usr/bin/env python3
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "output" / "sketchy_structxlipseg_100percent"
LOG = RUN_ROOT / "trained_models" / "seed42" / "log.txt"
METRICS = RUN_ROOT / "seg_results" / "seed42" / "Seg_structxlip_ViT-B-16_latest" / "metrics_miou_ciou.csv"
OUT = ROOT / "analysis" / "baseline_4286_4392_losses"

FIELDS = {
    "train_loss": r"Train Total: ([0-9.eE+-]+)",
    "seg_loss": r"Train Seg: ([0-9.eE+-]+)",
    "bce_loss": r"Train BCE: ([0-9.eE+-]+)",
    "dice_loss": r"Train DiceLoss: ([0-9.eE+-]+)",
    "clip_loss": r"Train CLIP: ([0-9.eE+-]+)",
    "weighted_clip_loss": r"Train WeightedCLIP: ([0-9.eE+-]+)",
    "loss_st": r"Train loss_st: ([0-9.eE+-]+)",
    "loss_rs": r"Train loss_rs: ([0-9.eE+-]+)",
    "loss_chunk_align": r"Train loss_chunk_align: ([0-9.eE+-]+)",
    "val_loss": r"Val Total: ([0-9.eE+-]+)",
    "val_bce": r"Val BCE: ([0-9.eE+-]+)",
    "val_dice_loss": r"Val DiceLoss: ([0-9.eE+-]+)",
    "val_dice_metric": r"Val DiceMetric: ([0-9.eE+-]+)",
}


def parse_log():
    rows = []
    epoch_re = re.compile(r"EPOCH:\s*(\d+)")
    for line in LOG.read_text(errors="ignore").splitlines():
        if "EPOCH:" not in line:
            continue
        m = epoch_re.search(line)
        if not m:
            continue
        row = {"epoch": int(m.group(1))}
        for key, pattern in FIELDS.items():
            mm = re.search(pattern, line)
            row[key] = float(mm.group(1)) if mm else 0.0
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No epoch rows parsed from {LOG}")
    return pd.DataFrame(rows)


def plot(df):
    OUT.mkdir(parents=True, exist_ok=True)
    x = df["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes = axes.ravel()
    panels = [
        ("Total / Main Losses", ["train_loss", "seg_loss", "weighted_clip_loss", "val_loss"]),
        ("Segmentation Components", ["bce_loss", "dice_loss", "val_bce", "val_dice_loss"]),
        ("Raw StructXLIP Auxiliary Losses", ["loss_st", "loss_rs", "loss_chunk_align"]),
        ("Validation Dice", ["val_dice_metric"]),
    ]
    for ax, (title, cols) in zip(axes, panels):
        for col in cols:
            ax.plot(x, df[col], marker="o", linewidth=1.8, label=col)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Fixed StructXLIP 100% Sketchy latest: mIoU 42.86 / cIoU 43.92", fontsize=14)
    path = OUT / "baseline_4286_4392_losses.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for col in ["loss_st", "loss_rs", "loss_chunk_align"]:
        ax.plot(x, df[col], marker="o", linewidth=1.8, label=col)
    ax.set_title("Fixed StructXLIP Raw Auxiliary Losses")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()
    aux_path = OUT / "baseline_4286_4392_aux_losses.png"
    fig.savefig(aux_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for col in ["train_loss", "seg_loss", "weighted_clip_loss", "val_loss"]:
        ax.plot(x, df[col], marker="o", linewidth=1.8, label=col)
    ax.set_title("Fixed StructXLIP Total/Main Losses")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()
    main_path = OUT / "baseline_4286_4392_total_main_losses.png"
    fig.savefig(main_path, dpi=180)
    plt.close(fig)
    return path, aux_path, main_path


def main():
    df = parse_log()
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "baseline_4286_4392_losses.csv"
    df.to_csv(csv_path, index=False)
    paths = plot(df)
    print(f"Parsed epochs: {len(df)}")
    print(f"Saved CSV: {csv_path}")
    for p in paths:
        print(f"Saved plot: {p}")
    print(f"Metrics: {METRICS}")
    print(df.tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
