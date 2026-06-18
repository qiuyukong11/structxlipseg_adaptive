#!/usr/bin/env python3
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    {
        "version": "v1",
        "name": "adaptive_3loss_gradnorm_prior",
        "root": ROOT / "output_adaptive_100_full" / "sketchy_structxlipseg_100percent_st_0.25_rs_0.02_chunk_0.07",
    },
    {
        "version": "v2",
        "name": "pure_online_cosine_softmax_alpha",
        "root": ROOT / "output_adaptive_v2_100_full" / "sketchy_structxlipseg_100percent_st_0.25_rs_0.02_chunk_0.07",
    },
    {
        "version": "v3",
        "name": "bounded_alpha_sigmoid",
        "root": ROOT / "output_adaptive_v3_100_full" / "sketchy_structxlipseg_100percent_st_0.25_rs_0.02_chunk_0.07",
    },
    {
        "version": "v4",
        "name": "scheduled_alpha_cosine",
        "root": ROOT / "output_adaptive_v4_100_full" / "sketchy_structxlipseg_100percent_st_0.25_rs_0.02_chunk_0.07",
    },
    {
        "version": "v5",
        "name": "not_found_locally",
        "root": None,
    },
    {
        "version": "v6",
        "name": "validation_simplex_multiplicative",
        "root": ROOT / "output_adaptive_v6_sketchy_100" / "sketchy_structxlipseg_100percent_st_0.25_rs_0.02_chunk_0.07",
    },
]

OUT = ROOT / "analysis" / "adaptive_v1_v6_losses"


def read_metric(path: Path):
    if not path.exists():
        return None, None
    avg = None
    cum = None
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("Name") == "__average__":
                avg = float(row["IoU"]) * 100.0
            elif row.get("Name") == "__cumulative__":
                cum = float(row["IoU"]) * 100.0
    return avg, cum


def parse_val_dice(log_path: Path):
    if not log_path.exists():
        return None, None, None
    pattern = re.compile(r"EPOCH:\s*(\d+).*?Val DiceMetric:\s*([0-9.]+)")
    best_epoch = None
    best_val = None
    final_val = None
    for line in log_path.read_text(errors="ignore").splitlines():
        m = pattern.search(line)
        if not m:
            continue
        epoch = int(m.group(1))
        val = float(m.group(2))
        final_val = val
        if best_val is None or val > best_val:
            best_val = val
            best_epoch = epoch
    return best_val, best_epoch, final_val


def plot_one(run, df):
    version = run["version"]
    title = f"{version}: {run['name']}"
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes = axes.ravel()
    x = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)

    groups = [
        ("Total / Main", ["train_loss", "seg_loss", "weighted_clip_loss", "weighted_struct_total"]),
        ("Raw Auxiliary", ["loss_st", "loss_rs", "loss_chunk_align"]),
        ("Weighted Auxiliary", ["weighted_loss_st", "weighted_loss_rs", "weighted_loss_chunk_align", "weighted_struct_total"]),
        ("Adaptive Weights", ["lambda_st", "lambda_rs", "lambda_chunk"]),
    ]
    for ax, (name, cols) in zip(axes, groups):
        for col in cols:
            if col in df.columns:
                ax.plot(x, df[col], marker="o", linewidth=1.8, label=col)
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=14)
    path = OUT / "per_version" / f"{version}_losses.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_compare(frames):
    compare_specs = [
        ("train_loss", "all_versions_train_loss.png", "Train Total Loss"),
        ("seg_loss", "all_versions_seg_loss.png", "Segmentation Loss"),
        ("weighted_struct_total", "all_versions_weighted_struct_total.png", "Weighted Struct Total"),
        ("gamma_aux_over_seg", "all_versions_gamma_aux_over_seg.png", "Gamma Aux / Seg"),
    ]
    paths = []
    for col, filename, title in compare_specs:
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for version, df in frames.items():
            if col not in df.columns:
                continue
            x = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)
            ax.plot(x, df[col], marker="o", linewidth=1.8, label=version)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
        path = OUT / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    for cols, filename, title in [
        (["loss_st", "loss_rs", "loss_chunk_align"], "all_versions_raw_aux_losses.png", "Raw Auxiliary Losses"),
        (["weighted_loss_st", "weighted_loss_rs", "weighted_loss_chunk_align"], "all_versions_weighted_aux_losses.png", "Weighted Auxiliary Losses"),
        (["lambda_st", "lambda_rs", "lambda_chunk"], "all_versions_lambdas.png", "Adaptive Lambdas"),
    ]:
        fig, axes = plt.subplots(len(cols), 1, figsize=(10, 9), sharex=True, constrained_layout=True)
        for ax, col in zip(axes, cols):
            for version, df in frames.items():
                if col not in df.columns:
                    continue
                x = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)
                ax.plot(x, df[col], marker="o", linewidth=1.5, label=version)
            ax.set_title(col)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        axes[-1].set_xlabel("epoch")
        fig.suptitle(title, fontsize=14)
        path = OUT / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def main():
    (OUT / "per_version").mkdir(parents=True, exist_ok=True)
    frames = {}
    rows = []
    plot_paths = []

    for run in RUNS:
        version = run["version"]
        root = run["root"]
        row = {"version": version, "name": run["name"], "status": "missing"}
        if root is None or not root.exists():
            rows.append(row)
            continue
        diag_path = root / "trained_models" / "seed42" / "structxlip_train_diagnostics.csv"
        log_path = root / "trained_models" / "seed42" / "log.txt"
        row["diagnostics_csv"] = str(diag_path.relative_to(ROOT)) if diag_path.exists() else ""
        row["log_path"] = str(log_path.relative_to(ROOT)) if log_path.exists() else ""
        if diag_path.exists():
            df = pd.read_csv(diag_path)
            frames[version] = df
            plot_paths.append(plot_one(run, df))
            final = df.iloc[-1]
            row.update({
                "status": "ok",
                "epochs": int(final.get("epoch", len(df))),
                "final_train_loss": float(final.get("train_loss", 0.0)),
                "final_seg_loss": float(final.get("seg_loss", 0.0)),
                "final_clip_loss": float(final.get("clip_loss", 0.0)),
                "final_weighted_struct_total": float(final.get("weighted_struct_total", 0.0)),
                "final_struct_over_seg": float(final.get("struct_over_seg", 0.0)),
                "final_lambda_st": float(final.get("lambda_st", 0.0)),
                "final_lambda_rs": float(final.get("lambda_rs", 0.0)),
                "final_lambda_chunk": float(final.get("lambda_chunk", 0.0)),
            })
        else:
            row["status"] = "missing_diagnostics"

        best_val, best_epoch, final_val = parse_val_dice(log_path)
        row["best_val_dice"] = best_val if best_val is not None else ""
        row["best_val_epoch"] = best_epoch if best_epoch is not None else ""
        row["final_val_dice"] = final_val if final_val is not None else ""

        for ckpt in ["latest", "best_dice"]:
            metric_path = root / "seg_results" / "seed42" / f"Seg_structxlip_ViT-B-16_{ckpt}" / "metrics_miou_ciou.csv"
            miou, ciou = read_metric(metric_path)
            row[f"{ckpt}_mIoU"] = miou if miou is not None else ""
            row[f"{ckpt}_cIoU"] = ciou if ciou is not None else ""
            row[f"{ckpt}_metrics_csv"] = str(metric_path.relative_to(ROOT)) if metric_path.exists() else ""
        rows.append(row)

    plot_paths.extend(plot_compare(frames))
    summary = pd.DataFrame(rows)
    summary_path = OUT / "summary.csv"
    summary.to_csv(summary_path, index=False)
    with (OUT / "plot_files.txt").open("w") as f:
        for path in plot_paths:
            f.write(str(path.relative_to(ROOT)) + "\n")
    print(f"Saved summary: {summary_path}")
    print(f"Saved {len(plot_paths)} plot files under: {OUT}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
