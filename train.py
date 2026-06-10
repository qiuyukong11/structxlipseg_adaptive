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
import torch.nn.functional as F
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

def move_optimizer_state_to_device(optimizer, device):
    device = torch.device(device)
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

def get_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
    "--config-file",
    # required=True,
    default="configs/sketchy_structxlipseg_5percent_debug.yaml",
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

def calc_loss(low_res_logits, low_res_label_batch, ce_loss, dice_loss, cfg, return_components=False):
    logits = _as_bchw(low_res_logits)
    labels = _as_bchw(low_res_label_batch).float()

    loss_ce = ce_loss(logits, labels)
    loss_dice = dice_loss(logits, labels)
    loss = cfg.TRAIN.DICE_WEIGHT * loss_dice + cfg.TRAIN.CE_WEIGHT * loss_ce
    if return_components:
        return loss, {
            "bce": loss_ce.detach(),
            "dice_loss": loss_dice.detach(),
            "seg_loss": loss.detach(),
        }
    return loss

# def calc_loss(low_res_logits, low_res_label_batch, ce_loss, dice_loss, cfg):
#     logits = _as_bchw(low_res_logits)
#     labels = _as_bchw(low_res_label_batch).float()

#     positives = labels.sum()
#     negatives = labels.numel() - positives
#     max_pos_weight = float(getattr(cfg.TRAIN, "MAX_POS_WEIGHT", 20.0))
#     if positives > 0:
#         pos_weight = torch.clamp(negatives / positives.clamp_min(1.0), min=1.0, max=max_pos_weight)
#     else:
#         pos_weight = logits.new_tensor(1.0)
#     pos_weight = pos_weight.reshape(1).to(dtype=logits.dtype, device=logits.device)

#     loss_ce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
#     loss_dice = dice_loss(logits, labels)
#     loss = cfg.TRAIN.DICE_WEIGHT * loss_dice + cfg.TRAIN.CE_WEIGHT * loss_ce
#     if not torch.isfinite(loss):
#         raise FloatingPointError(
#             f"Non-finite segmentation loss: total={loss.item()}, "
#             f"bce={loss_ce.item()}, dice={loss_dice.item()}, pos_weight={pos_weight.item():.4g}, "
#             f"logits_finite={torch.isfinite(logits).all().item()}, "
#             f"labels_finite={torch.isfinite(labels).all().item()}, "
#             f"logits_range=({torch.nan_to_num(logits.detach()).min().item():.4g}, "
#             f"{torch.nan_to_num(logits.detach()).max().item():.4g}), "
#             f"label_values={torch.unique(labels.detach()).cpu().tolist()[:10]}"
#         )
#     return loss


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

def build_datasets_from_json(cfg):
    struct_cfg = getattr(cfg, "STRUCTXLIP", None)
    common_kwargs = {
        "data_root": getattr(cfg.DATASET, "DATA_ROOT", ""),
        "image_size": int(cfg.DATASET.SIZE),
        "hflip_prob": float(getattr(cfg.DATASET, "HFLIP_PROB", 0.0)),
        "min_similarity": getattr(cfg.DATASET, "MIN_SIMILARITY", None),
        "use_original_caption_prefix": bool(getattr(cfg.DATASET, "USE_ORIGINAL_CAPTION_PREFIX", False)),
        "structure_image_field": getattr(struct_cfg, "STRUCTURE_IMAGE_FIELD", "filename_canny"),
        "chunk_top_k": int(getattr(struct_cfg, "CHUNK_TOP_K", 3)),
        "load_aux_images": (
            struct_cfg is not None
            and getattr(cfg.MODEL, "CLIP_MODEL", "") == "structxlip"
            and any(
                float(getattr(struct_cfg, key, 0.0)) != 0.0
                for key in (
                    "LAMBDA_STRUCTURE_TEXT",
                    "LAMBDA_RGB_STRUCTURE_CONSISTENCY",
                    "LAMBDA_CHUNK_ALIGN",
                )
            )
        ),
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
    val_bce_losses = []
    val_dice_losses = []
    dice_scores = []

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            images = batch["image"].to(device)
            masks = batch["ground_truth_mask"].to(device)
            text = batch["text_prompt"]

            logits = model(images, text=text, num_samples=1)[0]
            loss, loss_parts = calc_loss(logits, masks, ce_loss, dice_loss, cfg, return_components=True)
            val_losses.append(loss.item())
            val_bce_losses.append(loss_parts["bce"].item())
            val_dice_losses.append(loss_parts["dice_loss"].item())

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
    avg_bce_loss = mean(val_bce_losses)
    avg_dice_loss = mean(val_dice_losses)
    avg_dice = mean(dice_scores)
    model.train()
    return avg_loss, avg_dice, {
        "bce": avg_bce_loss,
        "dice_loss": avg_dice_loss,
    }

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
        include_background=False,  # 只对前景类别算 Dice loss
        sigmoid=True,
        reduction="mean"
    )

    # data loaders
    train_dataset, val_dataset = build_datasets_from_json(cfg)
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
        f"{cfg.DATASET.NAME}_Seg_"
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

    # debug
    cfg.resume = False
    # end debug
    if cfg.resume and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=cfg.MODEL.DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        move_optimizer_state_to_device(optimizer, cfg.MODEL.DEVICE)
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

    for epoch in range(start_epoch, num_epochs):
        epoch_losses = []
        epoch_seg_losses = []
        epoch_bce_losses = []
        epoch_dice_losses = []
        epoch_clip_losses = []
        epoch_weighted_clip_losses = []
        epoch_loss_st = []
        epoch_loss_rs = []
        epoch_loss_chunk_align = []

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
            model_outputs = model(**model_kwargs)
            structxlip_loss = None
            if cfg.MODEL.CLIP_MODEL == "structxlip":
                if len(model_outputs) == 3:
                    seg_logits, clip_loss, structxlip_loss = model_outputs
                else:
                    seg_logits, clip_loss = model_outputs
            else:
                seg_logits, clip_loss = model_outputs
            if not torch.isfinite(seg_logits).all():
                raise FloatingPointError(f"Non-finite logits at epoch={epoch + 1}, iter={i}, batch={describe_batch(batch)}")

            seg_loss, loss_parts = calc_loss(
                seg_logits,
                batch['ground_truth_mask'].to(cfg.MODEL.DEVICE),
                ce_loss,
                dice_loss,
                cfg,
                return_components=True,
            )
            loss = seg_loss
            weighted_clip_loss = seg_loss.new_zeros(())
            if use_clip_loss and clip_loss_weight != 0:
                weighted_clip_loss = clip_loss_weight * clip_loss
                loss = loss + weighted_clip_loss
            if structxlip_loss is not None:
                loss = loss + structxlip_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite total loss at epoch={epoch + 1}, iter={i}, batch={describe_batch(batch)}")

            optimizer.zero_grad()
            loss.backward()
            grad_clip_norm = float(getattr(cfg.TRAIN, "GRAD_CLIP_NORM", 1.0))
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    grad_clip_norm,
                    error_if_nonfinite=True,
                )
            # Optimize
            optimizer.step()
            epoch_losses.append(loss.item())
            epoch_seg_losses.append(loss_parts["seg_loss"].item())
            epoch_bce_losses.append(loss_parts["bce"].item())
            epoch_dice_losses.append(loss_parts["dice_loss"].item())
            epoch_clip_losses.append(clip_loss.detach().item() if torch.is_tensor(clip_loss) else float(clip_loss))
            epoch_weighted_clip_losses.append(weighted_clip_loss.detach().item())
            if cfg.MODEL.CLIP_MODEL == "structxlip":
                struct_loss_parts = getattr(model, "last_structxlip_losses", {})
                epoch_loss_st.append(struct_loss_parts.get("loss_st", seg_loss.new_zeros(())).detach().item())
                epoch_loss_rs.append(struct_loss_parts.get("loss_rs", seg_loss.new_zeros(())).detach().item())
                epoch_loss_chunk_align.append(struct_loss_parts.get("loss_chunk_align", seg_loss.new_zeros(())).detach().item())

        # Scheduler step at the end of the epoch
        scheduler.step()

        # End of epoch operations
        mean_epoch_loss = mean(epoch_losses)
        mean_epoch_seg_loss = mean(epoch_seg_losses)
        mean_epoch_bce_loss = mean(epoch_bce_losses)
        mean_epoch_dice_loss = mean(epoch_dice_losses)
        mean_epoch_clip_loss = mean(epoch_clip_losses)
        mean_epoch_weighted_clip_loss = mean(epoch_weighted_clip_losses)
        mean_epoch_loss_st = mean(epoch_loss_st) if epoch_loss_st else 0.0
        mean_epoch_loss_rs = mean(epoch_loss_rs) if epoch_loss_rs else 0.0
        mean_epoch_loss_chunk_align = mean(epoch_loss_chunk_align) if epoch_loss_chunk_align else 0.0
        # Validation phase
        mean_val_loss, mean_val_dice, val_loss_parts = evaluate_validation_loss(model, val_dataloader, cfg.MODEL.DEVICE, ce_loss, dice_loss, cfg)
        log_msg = (
            f"EPOCH: {epoch+1} | "
            f"Train Total: {mean_epoch_loss:.4f} | "
            f"Train Seg: {mean_epoch_seg_loss:.4f} | "
            f"Train BCE: {mean_epoch_bce_loss:.4f} | "
            f"Train DiceLoss: {mean_epoch_dice_loss:.4f} | "
            f"Train CLIP: {mean_epoch_clip_loss:.4f} | "
            f"Train WeightedCLIP: {mean_epoch_weighted_clip_loss:.4f} | "
        )
        if cfg.MODEL.CLIP_MODEL == "structxlip":
            log_msg += (
                f"Train loss_st: {mean_epoch_loss_st:.4f} | "
                f"Train loss_rs: {mean_epoch_loss_rs:.4f} | "
                f"Train loss_chunk_align: {mean_epoch_loss_chunk_align:.4f} | "
            )
        log_msg += (
            f"Val Total: {mean_val_loss:.4f} | "
            f"Val BCE: {val_loss_parts['bce']:.4f} | "
            f"Val DiceLoss: {val_loss_parts['dice_loss']:.4f} | "
            f"Val DiceMetric: {mean_val_dice:.4f}"
        )
        logger.info(log_msg)

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