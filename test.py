import argparse
import csv
import logging
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.json_refseg_dataset import JsonRefSegDataset
from trainers import *
from utils.main_utils import load_cfg_from_cfg_file


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="configs/sketchy.yaml", type=str, help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--data_percentage", type=int, default=100, help="Percentage of data to use")
    parser.add_argument("--source_dataset", type=str, default="sketchy", help="Dataset name used when loading checkpoint")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--weight-method", choices=("equal", "uncert", "dwa", "autol"), default=None)
    parser.add_argument("--grad-method", choices=("none", "graddrop", "pcgrad", "cagrad"), default=None)
    parser.add_argument("--all-loss-adaptive", action="store_true", help="Use all-loss adaptive output suffix.")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER, help="modify config options")
    args = parser.parse_args()
    # args.config_file = "/mnt/data/zruan/kqy/pami/segmentation/structxlip_seg_v2/configs/sketchy_structxlipseg_100percent.yaml"

    cfg = load_cfg_from_cfg_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.update({k: v for k, v in vars(args).items()})
    return cfg


def logger_config(log_path):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(level=logging.INFO)

    handler = logging.FileHandler(log_path, encoding="UTF-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)
    return logger


def dataset_name_with_percentage(cfg):
    name = cfg.DATASET.NAME
    return name + f"_{cfg.data_percentage}" if cfg.data_percentage != 100 else name


def result_name(cfg):
    backbone_name = cfg.MODEL.BACKBONE.replace("/", "-")
    return f"Seg_{cfg.MODEL.CLIP_MODEL}_{backbone_name}"


def cfg_get(node, name, default=None):
    try:
        return getattr(node, name)
    except AttributeError:
        return default


def build_json_refseg_test_dataset(cfg):
    struct_cfg = cfg_get(cfg, "STRUCTXLIP", None)
    common_kwargs = {
        "data_root": cfg_get(cfg.DATASET, "DATA_ROOT", ""),
        "image_size": int(cfg.DATASET.SIZE),
        "hflip_prob": 0.0,
        "min_similarity": cfg_get(cfg.DATASET, "MIN_SIMILARITY", None),
        "use_original_caption_prefix": bool(cfg_get(cfg.DATASET, "USE_ORIGINAL_CAPTION_PREFIX", False)),
        "structure_image_field": cfg_get(struct_cfg, "STRUCTURE_IMAGE_FIELD", "filename_canny"),
        "chunk_top_k": int(cfg_get(struct_cfg, "CHUNK_TOP_K", 3)),
    }
    test_json = cfg_get(cfg.DATASET, "TEST_JSON", "")
    val_json = cfg_get(cfg.DATASET, "VAL_JSON", "")
    train_json = cfg_get(cfg.DATASET, "TRAIN_JSON", "")

    if test_json:
        return JsonRefSegDataset(test_json, train=False, **common_kwargs)
    if val_json:
        return JsonRefSegDataset(val_json, train=False, **common_kwargs)
    if not train_json:
        raise ValueError("Set DATASET.TEST_JSON, DATASET.VAL_JSON, or DATASET.TRAIN_JSON for JsonRefSegDataset testing")

    full_dataset = JsonRefSegDataset(train_json, train=False, **common_kwargs)
    val_ratio = float(cfg_get(cfg.DATASET, "AUTO_VAL_RATIO", 0.0) or 0.0)
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("DATASET.VAL_JSON/TEST_JSON are empty, so DATASET.AUTO_VAL_RATIO must be in (0, 1)")

    indices = list(range(len(full_dataset.samples)))
    split_seed = int(cfg_get(cfg.DATASET, "VAL_SPLIT_SEED", cfg.seed))
    random.Random(split_seed).shuffle(indices)
    val_count = max(1, int(round(len(indices) * val_ratio)))
    val_indices = set(indices[:val_count])
    val_samples = [sample for i, sample in enumerate(full_dataset.samples) if i in val_indices]
    return JsonRefSegDataset(train_json, train=False, samples=val_samples, **common_kwargs)


def build_model(cfg):
    if cfg.MODEL.CLIP_MODEL == "structxlip":
        return build_structxlip(cfg)
    if cfg.MODEL.CLIP_MODEL == "clip":
        return build_clip(cfg)
    raise ValueError(f"Unsupported MODEL.CLIP_MODEL: {cfg.MODEL.CLIP_MODEL}")


def to_python_int(value):
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


def main():
    cfg = get_arguments()
    cfg.DATASET.NAME = dataset_name_with_percentage(cfg)
    if cfg.MODEL.CLIP_MODEL == "structxlip":
        cfg.DATASET.NAME = cfg.DATASET.NAME + f"_st_{getattr(cfg.STRUCTXLIP, 'LAMBDA_STRUCTURE_TEXT', 0.0)}_rs_{getattr(cfg.STRUCTXLIP, 'LAMBDA_RGB_STRUCTURE_CONSISTENCY', 0.0)}_chunk_{getattr(cfg.STRUCTXLIP, 'LAMBDA_CHUNK_ALIGN', 0.0)}"
        tau_cfg = cfg_get(cfg_get(cfg, "STRUCTXLIP", None), "LEARNABLE_TAU_LOSS", None)
        gradbudget_cfg = cfg_get(cfg_get(cfg, "STRUCTXLIP", None), "ADAPTIVE_GRADBUDGET_ALIGN", None)
        if bool(cfg_get(tau_cfg, "ENABLED", False)):
            cfg.DATASET.NAME = cfg.DATASET.NAME + f"_learnable_tau_w{float(cfg_get(tau_cfg, 'OVERALL_WEIGHT', 1.0)):g}"
        norm_balanced_cfg = cfg_get(cfg_get(cfg, "STRUCTXLIP", None), "ADAPTIVE_NORM_BALANCED", None)
        gradnorm_cfg = cfg_get(cfg_get(cfg, "STRUCTXLIP", None), "ADAPTIVE_GRADNORM", None)
        if bool(cfg_get(norm_balanced_cfg, "ENABLED", False)):
            cfg.DATASET.NAME = cfg.DATASET.NAME + "_norm_balanced"
        elif bool(cfg_get(gradnorm_cfg, "ENABLED", False)):
            cfg.DATASET.NAME = cfg.DATASET.NAME + "_gradnorm"
        elif bool(cfg_get(gradbudget_cfg, "ENABLED", False)):
            cfg.DATASET.NAME = cfg.DATASET.NAME + "_gradbudget_align"
        elif cfg_get(cfg, "weight_method", None) is not None and cfg_get(cfg, "grad_method", None) is not None:
            adaptive_prefix = "_autolambda_allloss" if bool(cfg_get(cfg, "all_loss_adaptive", False)) else "_autolambda"
            cfg.DATASET.NAME = cfg.DATASET.NAME + f"{adaptive_prefix}_{cfg.weight_method}_{cfg.grad_method}"

    if cfg.seed >= 0:
        print(f"Setting fixed seed: {cfg.seed}")
        set_random_seed(cfg.seed)

    checkpoint_type = "latest" if cfg.TEST.USE_LATEST else "best_dice"
    results_name = result_name(cfg)
    checkpoint_dataset = cfg.source_dataset or cfg.DATASET.NAME
    if cfg.data_percentage != 100:
        checkpoint_dataset = cfg.DATASET.NAME
    checkpoint_path = os.path.join(
        cfg.output_dir,
        cfg.DATASET.NAME,
        "trained_models",
        f"seed{cfg.seed}",
        f"{cfg.DATASET.NAME}_{results_name}_{checkpoint_type}.pth",
    )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    run_name = f"{results_name}_{checkpoint_type}"
    results_root = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "seg_results", f"seed{cfg.seed}", run_name)
    gt_root = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "gt_results", f"seed{cfg.seed}", run_name)
    os.makedirs(results_root, exist_ok=True)
    os.makedirs(gt_root, exist_ok=True)

    logger = logger_config(os.path.join(results_root, "log.txt"))
    logger.info("************")
    logger.info("** Config **")
    logger.info("************")
    logger.info(cfg)
    logger.info(f"Checkpoint: {checkpoint_path}")

    model = build_model(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=cfg.MODEL.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval().to(cfg.MODEL.DEVICE)

    test_dataset = build_json_refseg_test_dataset(cfg)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=int(cfg_get(cfg.TEST, "BATCH_SIZE", 32)),
        shuffle=False,
        num_workers=int(cfg_get(cfg.TEST, "WORKERS", cfg_get(cfg.TRAIN, "WORKERS", 8))),
        pin_memory=True,
    )
    logger.info(f"Test samples: {len(test_dataset)}")

    manifest_path = os.path.join(results_root, "manifest.csv")
    num_samples = int(cfg_get(cfg.TEST, "NUM_SAMPLES", 1))
    threshold = float(cfg_get(cfg.TEST, "THRESHOLD", 0.5))

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "prediction_path",
                "gt_path",
                "image_path",
                "mask_path",
                "item_idx",
                "seg_idx",
                "text_prompt",
            ],
        )
        writer.writeheader()

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Test"):
                images = batch["image"].to(cfg.MODEL.DEVICE)
                seg_samples = model(image=images, text=batch["text_prompt"], num_samples=num_samples)
                seg_probs = torch.sigmoid(seg_samples).mean(dim=0)
                mask_preds = (seg_probs > threshold).cpu().numpy().astype(np.uint8)
                gt_masks = batch["ground_truth_mask"].cpu().numpy().astype(np.uint8)

                for i in range(mask_preds.shape[0]):
                    item_idx = to_python_int(batch["item_idx"][i])
                    seg_idx = to_python_int(batch["seg_idx"][i])
                    name = f"item{item_idx:06d}_seg{seg_idx:04d}.png"
                    pred_path = os.path.join(results_root, name)
                    gt_path = os.path.join(gt_root, name)

                    cv2.imwrite(pred_path, mask_preds[i] * 255)
                    cv2.imwrite(gt_path, (gt_masks[i] > 0).astype(np.uint8) * 255)

                    writer.writerow({
                        "name": name,
                        "prediction_path": pred_path,
                        "gt_path": gt_path,
                        "image_path": batch["image_name"][i],
                        "mask_path": batch["mask_name"][i],
                        "item_idx": item_idx,
                        "seg_idx": seg_idx,
                        "text_prompt": batch["text_prompt"][i],
                    })

    logger.info(f"Saved predictions to: {results_root}")
    logger.info(f"Saved resized GT masks to: {gt_root}")
    logger.info(f"Saved manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
