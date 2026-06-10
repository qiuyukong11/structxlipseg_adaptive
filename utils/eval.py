import argparse
import csv
import os
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from main_utils import load_cfg_from_cfg_file


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, type=str, help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--data_percentage", type=int, default=100, help="Percentage of data to use")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--pred-dir", type=str, default="", help="Optional prediction directory override")
    parser.add_argument("--gt-dir", type=str, default="", help="Optional resized GT directory override")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER, help="modify config options")
    args = parser.parse_args()

    cfg = load_cfg_from_cfg_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.update({k: v for k, v in vars(args).items()})
    return cfg


def dataset_name_with_percentage(cfg):
    name = cfg.DATASET.NAME
    return name + f"_{cfg.data_percentage}" if cfg.data_percentage != 100 else name


def result_name(cfg):
    backbone_name = cfg.MODEL.BACKBONE.replace("/", "-")
    return f"Seg_{cfg.MODEL.CLIP_MODEL}_{backbone_name}"


def read_binary_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask >= 127


def binary_iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum(dtype=np.float64)
    union = np.logical_or(pred, gt).sum(dtype=np.float64)
    if union == 0:
        return 1.0, intersection, union
    return float(intersection / union), intersection, union


def read_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def matching_pairs_from_dirs(pred_dir, gt_dir):
    valid_exts = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}
    pred_files = {
        os.path.splitext(name)[0]: name
        for name in os.listdir(pred_dir)
        if os.path.splitext(name)[1].lower() in valid_exts
    }
    gt_files = {
        os.path.splitext(name)[0]: name
        for name in os.listdir(gt_dir)
        if os.path.splitext(name)[1].lower() in valid_exts
    }
    names = sorted(set(pred_files) & set(gt_files))
    return [
        {
            "name": gt_files[name],
            "prediction_path": os.path.join(pred_dir, pred_files[name]),
            "gt_path": os.path.join(gt_dir, gt_files[name]),
        }
        for name in names
    ]


def main():
    cfg = get_arguments()
    cfg.DATASET.NAME = dataset_name_with_percentage(cfg)

    checkpoint_type = "latest" if cfg.TEST.USE_LATEST else "best_dice"
    run_name = f"{result_name(cfg)}_{checkpoint_type}"
    pred_dir = cfg.pred_dir or os.path.join(cfg.output_dir, cfg.DATASET.NAME, "seg_results", f"seed{cfg.seed}", run_name)
    gt_dir = cfg.gt_dir or os.path.join(cfg.output_dir, cfg.DATASET.NAME, "gt_results", f"seed{cfg.seed}", run_name)
    manifest_path = os.path.join(pred_dir, "manifest.csv")

    if os.path.isfile(manifest_path):
        rows = read_manifest(manifest_path)
    else:
        rows = matching_pairs_from_dirs(pred_dir, gt_dir)

    if not rows:
        raise ValueError(f"No prediction/GT pairs found. pred_dir={pred_dir}, gt_dir={gt_dir}")

    save_path = os.path.join(pred_dir, "metrics_miou_ciou.csv")
    metrics = OrderedDict(Name=[], IoU=[])
    total_intersection = 0.0
    total_union = 0.0

    with tqdm(rows, desc="Evaluate") as pbar:
        for row in pbar:
            pred_path = row["prediction_path"]
            gt_path = row["gt_path"]
            pred = read_binary_mask(pred_path)
            gt = read_binary_mask(gt_path)
            if pred.shape != gt.shape:
                pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

            iou, intersection, union = binary_iou(pred, gt)
            total_intersection += intersection
            total_union += union
            metrics["Name"].append(row.get("name") or os.path.basename(gt_path))
            metrics["IoU"].append(round(iou, 6))

            miou = float(np.mean(metrics["IoU"]))
            ciou = 1.0 if total_union == 0 else float(total_intersection / total_union)
            pbar.set_postfix({"mIoU": f"{miou:.4f}", "cIoU": f"{ciou:.4f}"})

    dataframe = pd.DataFrame(metrics)
    miou = float(dataframe["IoU"].mean())
    ciou = 1.0 if total_union == 0 else float(total_intersection / total_union)
    dataframe.loc[len(dataframe)] = ["__average__", round(miou, 6)]
    dataframe.loc[len(dataframe)] = ["__cumulative__", round(ciou, 6)]
    dataframe.to_csv(save_path, index=False)

    print(20 * ">")
    print(f"mIoU for {os.path.basename(pred_dir)} {cfg.DATASET.NAME}: {miou * 100:.2f}%")
    print(f"cIoU for {os.path.basename(pred_dir)} {cfg.DATASET.NAME}: {ciou * 100:.2f}%")
    print(f"Saved metrics to: {save_path}")
    print(20 * "<")


if __name__ == "__main__":
    main()
