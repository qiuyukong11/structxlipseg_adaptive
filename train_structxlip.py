import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import DataLoader
from datasets.json_refseg_dataset import JsonRefSegDataset
from trainers import *
import os
import argparse
import random
import numpy as np
from torch.nn.modules.loss import BCEWithLogitsLoss
import logging
from utils.main_utils import load_cfg_from_cfg_file

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
    "--config-file",
    # required=True,
    default="configs/sketchy.yaml",
    type=str,
    help="Path to config file",
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help="Whether to resume training"
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    parser.add_argument(
        "--data_percentage", 
        type=int, 
        default=100, 
        help="Percentage of data to use.")
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="output", 
        help="output directory")
    
    parser.add_argument(
            "opts",
            default=[],
            nargs=argparse.REMAINDER,
            help="modify config options using the command-line",
        )

    args = parser.parse_args()

    cfg = load_cfg_from_cfg_file(args.config_file)

    cfg.merge_from_list(args.opts)

    cfg.update({k: v for k, v in vars(args).items()})    

    return cfg


def print_args(args, cfg):
    logging.info("***************")
    logging.info("** Arguments **")
    logging.info("***************")
    logging.info("************")
    logging.info("** Config **")
    logging.info("************")
    logging.info(cfg)

def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr

def _as_bchw(tensor):
    if tensor.ndim == 3:
        return tensor.unsqueeze(1)
    return tensor


def calc_loss(low_res_logits, low_res_label_batch, ce_loss, dice_loss, cfg):
    logits = _as_bchw(low_res_logits)
    labels = _as_bchw(low_res_label_batch).float()
    loss_ce = ce_loss(logits, labels)
    loss_dice = dice_loss(logits, labels)
    loss = cfg.TRAIN.DICE_WEIGHT * loss_dice + cfg.TRAIN.CE_WEIGHT * loss_ce
    if not torch.isfinite(loss):
        raise FloatingPointError(
            f"Non-finite segmentation loss: total={loss.item()}, "
            f"bce={loss_ce.item()}, dice={loss_dice.item()}, "
            f"logits_finite={torch.isfinite(logits).all().item()}, "
            f"labels_finite={torch.isfinite(labels).all().item()}"
        )
    return loss


def describe_batch(batch):
    def first_value(value):
        if isinstance(value, (list, tuple)):
            return value[0] if value else ""
        if torch.is_tensor(value):
            return value[0].item() if value.numel() else ""
        return value

    return {
        "image_name": first_value(batch.get("image_name", "")),
        "mask_name": first_value(batch.get("mask_name", "")),
        "text_prompt": first_value(batch.get("text_prompt", "")),
        "item_idx": first_value(batch.get("item_idx", "")),
        "seg_idx": first_value(batch.get("seg_idx", "")),
    }

def build_json_refseg_datasets(cfg):
    struct_cfg = getattr(cfg, "STRUCTXLIP", None)
    common_kwargs = {
        "data_root": getattr(cfg.DATASET, "DATA_ROOT", ""),
        "image_size": int(cfg.DATASET.SIZE),
        "hflip_prob": float(getattr(cfg.DATASET, "HFLIP_PROB", 0.0)),
        "min_similarity": getattr(cfg.DATASET, "MIN_SIMILARITY", None),
        "use_original_caption_prefix": bool(getattr(cfg.DATASET, "USE_ORIGINAL_CAPTION_PREFIX", False)),
        "structure_image_field": getattr(struct_cfg, "STRUCTURE_IMAGE_FIELD", "filename_canny"),
        "chunk_top_k": int(getattr(struct_cfg, "CHUNK_TOP_K", 3)),
    }
    train_json = getattr(cfg.DATASET, "TRAIN_JSON", "")
    val_json = getattr(cfg.DATASET, "VAL_JSON", "")
    if not train_json:
        raise ValueError("DATASET.TRAIN_JSON must be set when using JsonRefSegDataset")

    if val_json:
        train_dataset = JsonRefSegDataset(train_json, train=True, **common_kwargs)
        val_dataset = JsonRefSegDataset(val_json, train=False, **common_kwargs)
        return train_dataset, val_dataset

    full_dataset = JsonRefSegDataset(train_json, train=True, **common_kwargs)
    val_ratio = float(getattr(cfg.DATASET, "AUTO_VAL_RATIO", 0.0) or 0.0)
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("DATASET.VAL_JSON is empty, so DATASET.AUTO_VAL_RATIO must be in (0, 1)")

    indices = list(range(len(full_dataset.samples)))
    split_seed = int(getattr(cfg.DATASET, "VAL_SPLIT_SEED", cfg.seed))
    random.Random(split_seed).shuffle(indices)
    val_count = max(1, int(round(len(indices) * val_ratio)))
    val_indices = set(indices[:val_count])
    train_samples = [sample for i, sample in enumerate(full_dataset.samples) if i not in val_indices]
    val_samples = [sample for i, sample in enumerate(full_dataset.samples) if i in val_indices]

    train_dataset = JsonRefSegDataset(train_json, train=True, samples=train_samples, **common_kwargs)
    val_dataset = JsonRefSegDataset(train_json, train=False, samples=val_samples, **common_kwargs)
    return train_dataset, val_dataset

# Validation function
def evaluate_validation_loss(model, val_dataloader, device, ce_loss, dice_loss, cfg):
    model.eval()
    val_losses = []
    dice_scores = []

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            images = batch["image"].to(device)
            masks = batch["ground_truth_mask"].to(device)
            text = batch["text_prompt"]

            logits = model(images, text=text, num_samples=1)[0]
            loss = calc_loss(logits, masks, ce_loss, dice_loss, cfg)
            val_losses.append(loss.item())

            # Compute Dice score manually
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # Add channel dimension if missing
            if preds.ndim == 3:
                preds = preds.unsqueeze(1)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
            dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
            dice_scores.extend(dice.cpu().numpy())

    avg_loss = mean(val_losses)
    avg_dice = mean(dice_scores)
    model.train()
    return avg_loss, avg_dice

def main():
    cfg = get_arguments()
    cfg.DATASET.NAME = cfg.DATASET.NAME+f"_{cfg.data_percentage}" if cfg.data_percentage != 100 else cfg.DATASET.NAME
    os.makedirs(os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}"),exist_ok = True)

    logger = logger_config(os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}", "log.txt"))
    logger.info("************")
    logger.info("** Config **")
    logger.info("************")
    logger.info(cfg)
    if cfg.seed >= 0:
        logger.info("Setting fixed seed: {}".format(cfg.seed))
        set_random_seed(cfg.seed)

    # loss functions
    ce_loss = BCEWithLogitsLoss()
    dice_loss = monai.losses.DiceLoss(
        include_background=True,
        sigmoid=True,
        reduction="mean"
    )

    # data loaders
    train_dataset, val_dataset = build_json_refseg_datasets(cfg)
    logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    def worker_init_fn(worker_id):
        seed = cfg.seed + worker_id
        random.seed(seed)
        np.random.seed(seed)
    
    train_dataloader = DataLoader(train_dataset,
                                batch_size=cfg.TRAIN.BATCH_SIZE,
                                shuffle=True,
                                worker_init_fn=worker_init_fn,
                                num_workers=int(getattr(cfg.TRAIN, "WORKERS", 8)),
                                pin_memory=True,)

    val_dataloader = DataLoader(val_dataset,
                            batch_size=cfg.TRAIN.BATCH_SIZE,
                            shuffle=False,
                            worker_init_fn=worker_init_fn,
                            num_workers=int(getattr(cfg.TRAIN, "WORKERS", 8)),
                            pin_memory=True)

    if(cfg.MODEL.CLIP_MODEL == "structxlip"):
        model = build_structxlip(cfg)
    elif(cfg.MODEL.CLIP_MODEL == "clip"):
        model = build_clip(cfg)
    else:
        raise ValueError(f"Unsupported MODEL.CLIP_MODEL: {cfg.MODEL.CLIP_MODEL}")

    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)

    logger.info(f"Parameters to be updated: {enabled}")
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Initialize optimizer and Loss
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.TRAIN.LEARNING_RATE)
    num_epochs = cfg.TRAIN.NUM_EPOCHS
    use_clip_loss = bool(getattr(cfg.TRAIN, "USE_CLIP_LOSS", True))
    clip_loss_weight = float(getattr(cfg.TRAIN, "CLIP_WEIGHT", 0.0))
    logger.info(f"CLIP auxiliary loss enabled: {use_clip_loss}, weight: {clip_loss_weight}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,     # decay over all epochs
        eta_min=1e-4
    )

    backbone_name = cfg.MODEL.BACKBONE.replace("/", "-")

    results_name = (
        f"CLIPSeg_"
        f"{cfg.MODEL.CLIP_MODEL}_"
        f"{backbone_name}"
    )

    # Resume functionality
    resume_path = os.path.join(
                cfg.output_dir,
                cfg.DATASET.NAME,
                "trained_models",
                f"seed{cfg.seed}",
                f"{results_name}_latest.pth")

    start_epoch = 0
    best_loss = float("inf")
    best_dice = -1.0

    if cfg.resume and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=cfg.MODEL.DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint.get("scheduler", {}))
        resume_lr = float(cfg.TRAIN.LEARNING_RATE)
        for group in optimizer.param_groups:
            group["lr"] = resume_lr
        scheduler.base_lrs = [resume_lr for _ in scheduler.base_lrs]
        if hasattr(scheduler, "_last_lr"):
            scheduler._last_lr = [resume_lr for _ in scheduler._last_lr]
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("best_loss", best_loss)
        best_dice = checkpoint.get("best_dice", best_dice)
        logger.info(
            f"Loaded checkpoint from epoch {start_epoch}, "
            f"best loss: {best_loss:.4f}, best dice: {best_dice:.4f}"
        )

    # Set model to train and into the device
    model.train()
    model.to(cfg.MODEL.DEVICE)

    total_loss = []

    for epoch in range(start_epoch, num_epochs):
        epoch_losses = []

        for i, batch in enumerate(tqdm(train_dataloader)):

            model_kwargs = {
                "image": batch["image"].to(cfg.MODEL.DEVICE),
                "text": batch["text_prompt"],
                "return_clip_loss": use_clip_loss and clip_loss_weight != 0,
            }
            if cfg.MODEL.CLIP_MODEL == "structxlip":
                structure_image = batch["original_structure_image"] if "original_structure_image" in batch else batch["structure_image"]
                has_structure = batch["has_original_structure"] if "has_original_structure" in batch else batch["has_structure"]
                original_text = batch["original_text"] if "original_text" in batch else batch["text_prompt"]
                model_kwargs.update({
                    "structure_image": structure_image.to(cfg.MODEL.DEVICE),
                    "edge_images": batch["edge_images"].to(cfg.MODEL.DEVICE),
                    "has_structure": has_structure.to(cfg.MODEL.DEVICE),
                    "edge_valid_mask": batch["edge_valid_mask"].to(cfg.MODEL.DEVICE),
                    "original_text": original_text,
                })
            seg_logits, clip_loss = model(**model_kwargs)
            if not torch.isfinite(seg_logits).all():
                raise FloatingPointError(f"Non-finite logits at epoch={epoch + 1}, iter={i}, batch={describe_batch(batch)}")

            total_loss = 0
            loss = calc_loss(seg_logits, batch['ground_truth_mask'].to(cfg.MODEL.DEVICE), ce_loss, dice_loss, cfg)
            if use_clip_loss and clip_loss_weight != 0:
                loss += clip_loss_weight * clip_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite total loss at epoch={epoch + 1}, iter={i}, batch={describe_batch(batch)}")

            optimizer.zero_grad()
            loss.backward()
            grad_clip_norm = float(getattr(cfg.TRAIN, "GRAD_CLIP_NORM", 1.0))
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), grad_clip_norm)
            # Optimize
            optimizer.step()
            epoch_losses.append(loss.item())

        # Scheduler step at the end of the epoch
        scheduler.step()

        # End of epoch operations
        mean_epoch_loss = mean(epoch_losses)
        # Validation phase
        mean_val_loss, mean_val_dice = evaluate_validation_loss(model, val_dataloader, cfg.MODEL.DEVICE, ce_loss, dice_loss, cfg)
        logger.info(f'EPOCH: {epoch+1} | Training Loss: {mean_epoch_loss:.4f} | Validation Loss: {mean_val_loss:.4f}')

        # Save the best model based on validation loss
        if mean_val_dice > best_dice:
            logger.info(f"New best Dice: {best_dice:.4f} → {mean_val_dice:.4f}")
            best_dice = mean_val_dice
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_dice": best_dice,
            }, os.path.join(
                cfg.output_dir,
                cfg.DATASET.NAME,
                "trained_models",
                f"seed{cfg.seed}",
                f"{results_name}_best_dice.pth"
            ))
        else:
            logger.info(f"Dice: {mean_val_dice:.4f}")

        best_loss = min(best_loss, mean_val_loss)

        # Save the latest model
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_loss": best_loss,
            "best_dice": best_dice,
        }, 
        os.path.join(
        cfg.output_dir,
        cfg.DATASET.NAME,
        "trained_models",
        f"seed{cfg.seed}",
        f"{results_name}_latest.pth")
        )
        
if __name__ == "__main__":
    main()