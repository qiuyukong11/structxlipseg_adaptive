#!/usr/bin/env python
import argparse
import csv
import re
from pathlib import Path


EPOCH_PATTERN = re.compile(r"EPOCH:\s*(?P<epoch>\d+)\s*\|\s*(?P<body>.*)")
METRIC_PATTERN = re.compile(r"([^|:]+):\s*([-+0-9.eE]+)")
NAME_PATTERN = re.compile(r"_st_(?P<st>[^_]+)_rs_(?P<rs>[^_]+)_chunk_(?P<chunk>[^_]+)")


DIAGNOSTIC_COLUMNS = [
    "epoch",
    "train_loss",
    "seg_loss",
    "clip_loss",
    "weighted_clip_loss",
    "loss_st",
    "loss_rs",
    "loss_chunk_align",
    "weighted_loss_st",
    "weighted_loss_rs",
    "weighted_loss_chunk_align",
    "weighted_struct_total",
    "struct_over_seg",
    "weighted_st_over_seg",
    "weighted_rs_over_seg",
    "weighted_chunk_over_seg",
    "grad_norm_main",
    "grad_norm_st",
    "grad_norm_rs",
    "grad_norm_chunk",
    "cos_main_st",
    "cos_main_rs",
    "cos_main_chunk",
    "lambda_st",
    "lambda_rs",
    "lambda_chunk",
]


SUMMARY_COLUMNS = [
    "experiment",
    "seed",
    "lambda_st",
    "lambda_rs",
    "lambda_chunk",
    "epochs",
    "status",
    "best_val_dice",
    "best_val_dice_epoch",
    "final_val_dice",
    "final_val_loss",
    "final_train_loss",
    "final_train_seg",
    "final_train_clip",
    "final_weighted_clip",
    "final_loss_st",
    "final_loss_rs",
    "final_loss_chunk_align",
    "min_val_loss",
    "min_val_loss_epoch",
    "best_checkpoint",
    "latest_checkpoint",
    "prediction_rows",
    "diagnostics_csv",
    "diagnostics_generated",
    "diagnostics_epochs",
    "diagnostics_best_struct_over_seg",
    "diagnostics_final_struct_over_seg",
    "diagnostics_final_cos_main_st",
    "diagnostics_final_cos_main_rs",
    "diagnostics_final_cos_main_chunk",
    "log_path",
]


def safe_float(value, default=0.0):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_lambdas(experiment_name):
    match = NAME_PATTERN.search(experiment_name)
    if not match:
        return {"lambda_st": "", "lambda_rs": "", "lambda_chunk": ""}
    return {
        "lambda_st": match.group("st"),
        "lambda_rs": match.group("rs"),
        "lambda_chunk": match.group("chunk"),
    }


def numeric_lambdas(experiment_name):
    lambdas = parse_lambdas(experiment_name)
    return {
        "lambda_st": safe_float(lambdas["lambda_st"]),
        "lambda_rs": safe_float(lambdas["lambda_rs"]),
        "lambda_chunk": safe_float(lambdas["lambda_chunk"]),
    }


def parse_epoch_line(line):
    match = EPOCH_PATTERN.search(line)
    if not match:
        return None
    metrics = {"epoch": int(match.group("epoch"))}
    for key, value in METRIC_PATTERN.findall(match.group("body")):
        normalized_key = key.strip().lower().replace(" ", "_")
        metrics[normalized_key] = safe_float(value)
    return metrics


def parse_log(log_path):
    epochs = []
    if not log_path.exists():
        return epochs
    for line in log_path.read_text(errors="ignore").splitlines():
        parsed = parse_epoch_line(line)
        if parsed:
            epochs.append(parsed)
    return epochs


def count_manifest_rows(manifest_path):
    if manifest_path is None or not manifest_path.is_file():
        return 0
    with manifest_path.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        row_count = sum(1 for _ in reader)
    return max(0, row_count - 1)


def diagnostics_row_from_epoch(epoch_row, lambdas):
    seg_loss = safe_float(epoch_row.get("train_seg"))
    loss_st = safe_float(epoch_row.get("train_loss_st"))
    loss_rs = safe_float(epoch_row.get("train_loss_rs"))
    loss_chunk = safe_float(epoch_row.get("train_loss_chunk_align"))

    weighted_st = lambdas["lambda_st"] * loss_st
    weighted_rs = lambdas["lambda_rs"] * loss_rs
    weighted_chunk = lambdas["lambda_chunk"] * loss_chunk
    weighted_struct_total = weighted_st + weighted_rs + weighted_chunk

    def over_seg(value):
        return value / seg_loss if abs(seg_loss) > 1e-12 else 0.0

    return {
        "epoch": epoch_row.get("epoch", 0),
        "train_loss": safe_float(epoch_row.get("train_total")),
        "seg_loss": seg_loss,
        "clip_loss": safe_float(epoch_row.get("train_clip")),
        "weighted_clip_loss": safe_float(epoch_row.get("train_weightedclip")),
        "loss_st": loss_st,
        "loss_rs": loss_rs,
        "loss_chunk_align": loss_chunk,
        "weighted_loss_st": weighted_st,
        "weighted_loss_rs": weighted_rs,
        "weighted_loss_chunk_align": weighted_chunk,
        "weighted_struct_total": weighted_struct_total,
        "struct_over_seg": over_seg(weighted_struct_total),
        "weighted_st_over_seg": over_seg(weighted_st),
        "weighted_rs_over_seg": over_seg(weighted_rs),
        "weighted_chunk_over_seg": over_seg(weighted_chunk),
        "grad_norm_main": 0.0,
        "grad_norm_st": 0.0,
        "grad_norm_rs": 0.0,
        "grad_norm_chunk": 0.0,
        "cos_main_st": 0.0,
        "cos_main_rs": 0.0,
        "cos_main_chunk": 0.0,
        "lambda_st": lambdas["lambda_st"],
        "lambda_rs": lambdas["lambda_rs"],
        "lambda_chunk": lambdas["lambda_chunk"],
    }


def ensure_diagnostics_csv(experiment_name, diagnostics_path, epochs, overwrite=False):
    if diagnostics_path.exists() and not overwrite:
        return False

    lambdas = numeric_lambdas(experiment_name)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    with diagnostics_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=DIAGNOSTIC_COLUMNS)
        writer.writeheader()
        for epoch_row in epochs:
            row = diagnostics_row_from_epoch(epoch_row, lambdas)
            writer.writerow({column: row.get(column, 0.0) for column in DIAGNOSTIC_COLUMNS})
    return True


def read_diagnostics(diagnostics_path):
    if not diagnostics_path.exists():
        return {}
    with diagnostics_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        return {"diagnostics_epochs": 0}

    final = rows[-1]
    struct_values = [safe_float(row.get("struct_over_seg")) for row in rows]
    return {
        "diagnostics_epochs": len(rows),
        "diagnostics_best_struct_over_seg": max(struct_values) if struct_values else 0.0,
        "diagnostics_final_struct_over_seg": safe_float(final.get("struct_over_seg")),
        "diagnostics_final_cos_main_st": safe_float(final.get("cos_main_st")),
        "diagnostics_final_cos_main_rs": safe_float(final.get("cos_main_rs")),
        "diagnostics_final_cos_main_chunk": safe_float(final.get("cos_main_chunk")),
    }


def summarize_seed(experiment_dir, seed_dir, overwrite_diagnostics=False):
    experiment = experiment_dir.name
    seed = seed_dir.name.replace("seed", "")
    log_path = seed_dir / "log.txt"
    epochs = parse_log(log_path)
    final_epoch = epochs[-1] if epochs else {}

    val_dice_epochs = [row for row in epochs if "val_dicemetric" in row]
    val_loss_epochs = [row for row in epochs if "val_total" in row]
    best_dice_epoch = max(val_dice_epochs, key=lambda row: row["val_dicemetric"], default={})
    min_loss_epoch = min(val_loss_epochs, key=lambda row: row["val_total"], default={})

    best_checkpoint = next(seed_dir.glob("*best_dice.pth"), None)
    latest_checkpoint = next(seed_dir.glob("*latest.pth"), None)
    diagnostics_path = seed_dir / "structxlip_train_diagnostics.csv"
    diagnostics_generated = ensure_diagnostics_csv(
        experiment,
        diagnostics_path,
        epochs,
        overwrite=overwrite_diagnostics,
    )

    manifest_candidates = sorted((experiment_dir / "seg_results" / f"seed{seed}").glob("*/manifest.csv"))
    manifest_path = manifest_candidates[-1] if manifest_candidates else None

    row = {
        "experiment": experiment,
        "seed": seed,
        **parse_lambdas(experiment),
        "epochs": len(epochs),
        "status": "complete" if latest_checkpoint else ("log_only" if epochs else "missing_log"),
        "best_val_dice": best_dice_epoch.get("val_dicemetric", 0.0),
        "best_val_dice_epoch": best_dice_epoch.get("epoch", 0),
        "final_val_dice": final_epoch.get("val_dicemetric", 0.0),
        "final_val_loss": final_epoch.get("val_total", 0.0),
        "final_train_loss": final_epoch.get("train_total", 0.0),
        "final_train_seg": final_epoch.get("train_seg", 0.0),
        "final_train_clip": final_epoch.get("train_clip", 0.0),
        "final_weighted_clip": final_epoch.get("train_weightedclip", 0.0),
        "final_loss_st": final_epoch.get("train_loss_st", 0.0),
        "final_loss_rs": final_epoch.get("train_loss_rs", 0.0),
        "final_loss_chunk_align": final_epoch.get("train_loss_chunk_align", 0.0),
        "min_val_loss": min_loss_epoch.get("val_total", 0.0),
        "min_val_loss_epoch": min_loss_epoch.get("epoch", 0),
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else "",
        "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint else "",
        "prediction_rows": count_manifest_rows(manifest_path),
        "diagnostics_csv": str(diagnostics_path) if diagnostics_path.exists() else "",
        "diagnostics_generated": diagnostics_generated,
        "log_path": str(log_path) if log_path.exists() else "",
    }
    row.update(read_diagnostics(diagnostics_path))
    return row


def collect_rows(root, overwrite_diagnostics=False):
    rows = []
    for experiment_dir in sorted(root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        trained_models_dir = experiment_dir / "trained_models"
        seed_dirs = sorted(path for path in trained_models_dir.glob("seed*") if path.is_dir())
        if not seed_dirs:
            rows.append({
                "experiment": experiment_dir.name,
                **parse_lambdas(experiment_dir.name),
                "status": "missing_seed",
            })
            continue
        for seed_dir in seed_dirs:
            rows.append(summarize_seed(
                experiment_dir,
                seed_dir,
                overwrite_diagnostics=overwrite_diagnostics,
            ))
    return rows


def write_summary(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


def print_table(rows):
    print("experiment,seed,status,epochs,best_val_dice,best_epoch,final_val_dice,prediction_rows,diagnostics_generated")
    for row in rows:
        print(
            f"{row.get('experiment', '')},"
            f"{row.get('seed', '')},"
            f"{row.get('status', '')},"
            f"{row.get('epochs', '')},"
            f"{safe_float(row.get('best_val_dice')):.4f},"
            f"{row.get('best_val_dice_epoch', '')},"
            f"{safe_float(row.get('final_val_dice')):.4f},"
            f"{row.get('prediction_rows', '')},"
            f"{row.get('diagnostics_generated', '')}"
        )


def main():
    parser = argparse.ArgumentParser(description="Summarize StructXLIP ablation experiment directories.")
    parser.add_argument("--root", default="abalation", help="Ablation output root directory.")
    parser.add_argument("--output", default=None, help="Output CSV path. Default: <root>/summary.csv")
    parser.add_argument(
        "--overwrite-diagnostics",
        action="store_true",
        help="Regenerate structxlip_train_diagnostics.csv even if it already exists.",
    )
    parser.add_argument("--quiet", action="store_true", help="Do not print the compact table.")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Ablation root not found: {root}")

    output_path = Path(args.output) if args.output else root / "summary.csv"
    rows = collect_rows(root, overwrite_diagnostics=args.overwrite_diagnostics)
    write_summary(rows, output_path)

    if not args.quiet:
        print_table(rows)
    print(f"\nWrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
