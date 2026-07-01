import argparse
import csv
import logging
import os
import random
from statistics import mean

import monai
import numpy as np
import torch
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.json_refseg_dataset import JsonRefSegDataset
from trainers import build_clip, build_structxlip
from utils.main_utils import CfgNode, load_cfg_from_cfg_file


GRADNORM_TASKS = ("st", "rs", "chunk")
GRADNORM_LOSS_KEYS = ("loss_st", "loss_rs", "loss_chunk_align")

GRADNORM_DIAGNOSTIC_COLUMNS = [
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
    "gradnorm_loss",
    "grad_norm_st",
    "grad_norm_rs",
    "grad_norm_chunk",
    "target_grad_norm_st",
    "target_grad_norm_rs",
    "target_grad_norm_chunk",
    "loss_ratio_st",
    "loss_ratio_rs",
    "loss_ratio_chunk",
    "inverse_rate_st",
    "inverse_rate_rs",
    "inverse_rate_chunk",
    "w_st",
    "w_rs",
    "w_chunk",
    "lambda_st",
    "lambda_rs",
    "lambda_chunk",
    "gamma",
]


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


def configure_gradnorm_second_order_attention():
    if not torch.cuda.is_available():
        return
    cuda_backend = getattr(torch.backends, "cuda", None)
    if cuda_backend is None:
        return
    for name, value in (
        ("enable_flash_sdp", False),
        ("enable_mem_efficient_sdp", False),
        ("enable_math_sdp", True),
        ("enable_cudnn_sdp", False),
    ):
        fn = getattr(cuda_backend, name, None)
        if callable(fn):
            fn(value)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        default="configs/sketchy_structxlipseg_5percent_debug.yaml",
        type=str,
        help="Path to config file",
    )
    parser.add_argument("--resume", action="store_true", help="Whether to resume training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--data_percentage", type=int, default=100, help="Percentage of data to use.")
    parser.add_argument("--output-dir", type=str, default="output", help="output directory")
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
    force_gradnorm_only(cfg)
    return cfg


def force_gradnorm_only(cfg):
    if getattr(getattr(cfg, "MODEL", None), "CLIP_MODEL", "") != "structxlip":
        return
    struct_cfg = getattr(cfg, "STRUCTXLIP", None)
    if struct_cfg is None:
        cfg.STRUCTXLIP = CfgNode()
        struct_cfg = cfg.STRUCTXLIP

    disabled_sections = (
        "ADAPTIVE_3LOSS",
        "ADAPTIVE_V2",
        "ADAPTIVE_V3",
        "ADAPTIVE_V4",
        "ADAPTIVE_V6",
        "ADAPTIVE_V7",
        "ADAPTIVE_GRADBUDGET_ALIGN",
        "ADAPTIVE_NORM_BALANCED",
        "LEARNABLE_TAU_LOSS",
    )
    for section in disabled_sections:
        if section in struct_cfg and isinstance(struct_cfg[section], dict):
            struct_cfg[section]["ENABLED"] = False

    if "ADAPTIVE_GRADNORM" not in struct_cfg or not isinstance(struct_cfg["ADAPTIVE_GRADNORM"], dict):
        struct_cfg["ADAPTIVE_GRADNORM"] = CfgNode()
    gradnorm_cfg = struct_cfg["ADAPTIVE_GRADNORM"]
    gradnorm_cfg["ENABLED"] = True
    gradnorm_cfg.setdefault("ALPHA", 1.5)
    gradnorm_cfg.setdefault("GAMMA", 0.1)
    gradnorm_cfg.setdefault("WEIGHT_LR", getattr(cfg.TRAIN, "LEARNING_RATE", 1e-4))
    gradnorm_cfg.setdefault("EPS", 1e-8)
    gradnorm_cfg.setdefault("CLAMP_AUX_LOSS_NONNEG", True)
    gradnorm_cfg.setdefault("MIN_WEIGHT", 0.0)
    gradnorm_cfg.setdefault("MAX_WEIGHT", 10.0)


def logger_config(log_path):
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    logger.handlers.clear()

    handler = logging.FileHandler(log_path, encoding="UTF-8")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


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


def safe_float(value):
    if value is None:
        return 0.0
    try:
        if torch.is_tensor(value):
            detached = value.detach()
            if detached.numel() == 0:
                return 0.0
            if detached.numel() > 1:
                detached = detached.float().mean()
            result = float(detached.item())
        else:
            result = float(value)
    except (TypeError, ValueError, RuntimeError):
        return 0.0
    return result if np.isfinite(result) else 0.0


def safe_div(numerator, denominator, eps=1e-12):
    denominator = safe_float(denominator)
    if abs(denominator) <= eps:
        return 0.0
    return safe_float(numerator) / denominator


def mean_or_zero(values):
    return mean(values) if values else 0.0


def get_gradnorm_config(cfg):
    gradnorm_cfg = getattr(getattr(cfg, "STRUCTXLIP", None), "ADAPTIVE_GRADNORM", None)
    eps = float(getattr(gradnorm_cfg, "EPS", 1e-8))
    min_weight = float(getattr(gradnorm_cfg, "MIN_WEIGHT", 0.0))
    max_weight = float(getattr(gradnorm_cfg, "MAX_WEIGHT", 10.0))
    if max_weight <= min_weight:
        max_weight = 10.0
    return {
        "enabled": bool(getattr(gradnorm_cfg, "ENABLED", False)),
        "alpha": float(getattr(gradnorm_cfg, "ALPHA", 1.5)),
        "gamma": float(getattr(gradnorm_cfg, "GAMMA", 0.1)),
        "weight_lr": float(getattr(gradnorm_cfg, "WEIGHT_LR", getattr(cfg.TRAIN, "LEARNING_RATE", 1e-4))),
        "eps": eps if eps > 0.0 else 1e-8,
        "clamp_aux_loss_nonneg": bool(getattr(gradnorm_cfg, "CLAMP_AUX_LOSS_NONNEG", True)),
        "min_weight": max(0.0, min_weight),
        "max_weight": max_weight,
    }


def is_gradnorm_enabled(cfg):
    return (
        getattr(getattr(cfg, "MODEL", None), "CLIP_MODEL", "") == "structxlip"
        and get_gradnorm_config(cfg)["enabled"]
    )


def select_gradnorm_parameters(model):
    adapters = getattr(model, "pvl_adapters", None)
    if adapters is not None and len(adapters) > 0:
        last_adapter_params = [param for param in adapters[-1].parameters() if param.requires_grad]
        if last_adapter_params:
            return last_adapter_params
    return [param for param in model.parameters() if param.requires_grad]


def grad_norm_for_loss(loss, parameters):
    grads = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        create_graph=True,
        allow_unused=True,
    )
    flat_grads = []
    for grad in grads:
        if grad is not None:
            flat_grads.append(grad.reshape(-1).float())
    if not flat_grads:
        return loss.new_zeros(())
    return torch.linalg.vector_norm(torch.cat(flat_grads))


class GradNormController:
    def __init__(self, cfg, device):
        gradnorm_cfg = get_gradnorm_config(cfg)
        self.alpha = max(0.0, float(gradnorm_cfg["alpha"]))
        self.gamma = max(0.0, float(gradnorm_cfg["gamma"]))
        self.eps = float(gradnorm_cfg["eps"])
        self.clamp_aux_loss_nonneg = bool(gradnorm_cfg["clamp_aux_loss_nonneg"])
        self.min_weight = float(gradnorm_cfg["min_weight"])
        self.max_weight = float(gradnorm_cfg["max_weight"])
        self.weights = torch.nn.Parameter(torch.ones(len(GRADNORM_TASKS), device=device))
        self.initial_losses = None
        self.total_weight = float(len(GRADNORM_TASKS))

    def state_dict(self):
        return {
            "weights": self.weights.detach().cpu(),
            "initial_losses": None if self.initial_losses is None else self.initial_losses.detach().cpu(),
        }

    def load_state_dict(self, state):
        if not state:
            return
        if "weights" in state and torch.is_tensor(state["weights"]):
            self.weights.data.copy_(state["weights"].to(self.weights.device, dtype=self.weights.dtype))
            self.renormalize_()
        if "initial_losses" in state and torch.is_tensor(state["initial_losses"]):
            self.initial_losses = state["initial_losses"].to(self.weights.device, dtype=self.weights.dtype)

    def renormalize_(self):
        with torch.no_grad():
            self.weights.data.clamp_(min=self.min_weight, max=self.max_weight)
            weight_sum = self.weights.data.sum().clamp_min(self.eps)
            self.weights.data.mul_(self.total_weight / weight_sum)
            self.weights.data.clamp_(min=self.min_weight, max=self.max_weight)
            weight_sum = self.weights.data.sum().clamp_min(self.eps)
            self.weights.data.mul_(self.total_weight / weight_sum)

    def compute(self, raw_losses, parameters):
        params = list(parameters)
        losses = []
        for key in GRADNORM_LOSS_KEYS:
            loss = raw_losses.get(key)
            if not torch.is_tensor(loss):
                raise RuntimeError(f"GradNorm expected tensor for {key}")
            if self.clamp_aux_loss_nonneg:
                loss = torch.clamp(loss, min=0.0)
            losses.append(loss)

        loss_vec = torch.stack(losses)
        if self.initial_losses is None:
            self.initial_losses = loss_vec.detach().clamp_min(self.eps)

        scaled_weights = self.gamma * self.weights
        weighted_losses = scaled_weights * loss_vec
        weighted_struct_total = weighted_losses.sum()

        grad_norms = torch.stack([
            grad_norm_for_loss(weighted_losses[idx], params)
            for idx in range(len(GRADNORM_TASKS))
        ])

        loss_ratio = loss_vec.detach().clamp_min(self.eps) / self.initial_losses.clamp_min(self.eps)
        inverse_rate = loss_ratio / loss_ratio.mean().clamp_min(self.eps)
        mean_grad_norm = grad_norms.detach().mean()
        target_grad_norms = (mean_grad_norm * inverse_rate.pow(self.alpha)).detach()
        gradnorm_loss = torch.abs(grad_norms - target_grad_norms).sum()

        diagnostics = {
            "weighted_loss_st": weighted_losses[0],
            "weighted_loss_rs": weighted_losses[1],
            "weighted_loss_chunk_align": weighted_losses[2],
            "weighted_struct_total": weighted_struct_total,
            "gradnorm_loss": gradnorm_loss,
            "grad_norm_st": grad_norms[0],
            "grad_norm_rs": grad_norms[1],
            "grad_norm_chunk": grad_norms[2],
            "target_grad_norm_st": target_grad_norms[0],
            "target_grad_norm_rs": target_grad_norms[1],
            "target_grad_norm_chunk": target_grad_norms[2],
            "loss_ratio_st": loss_ratio[0],
            "loss_ratio_rs": loss_ratio[1],
            "loss_ratio_chunk": loss_ratio[2],
            "inverse_rate_st": inverse_rate[0],
            "inverse_rate_rs": inverse_rate[1],
            "inverse_rate_chunk": inverse_rate[2],
            "w_st": self.weights[0],
            "w_rs": self.weights[1],
            "w_chunk": self.weights[2],
            "lambda_st": scaled_weights[0],
            "lambda_rs": scaled_weights[1],
            "lambda_chunk": scaled_weights[2],
            "gamma": self.gamma,
        }
        return weighted_struct_total, gradnorm_loss, diagnostics


def fixed_struct_loss(model, seg_loss):
    raw_losses = getattr(model, "last_structxlip_loss_tensors", {})
    zero = seg_loss.new_zeros(())
    loss_st = raw_losses.get("loss_st", zero)
    loss_rs = raw_losses.get("loss_rs", zero)
    loss_chunk = raw_losses.get("loss_chunk_align", zero)
    lambda_st = safe_float(getattr(model, "lambda_scribble_text", 0.0))
    lambda_rs = safe_float(getattr(model, "lambda_rgb_scribble", 0.0))
    lambda_chunk = safe_float(getattr(model, "lambda_chunk", 0.0))
    weighted_loss_st = lambda_st * loss_st if torch.is_tensor(loss_st) else zero
    weighted_loss_rs = lambda_rs * loss_rs if torch.is_tensor(loss_rs) else zero
    weighted_loss_chunk = lambda_chunk * loss_chunk if torch.is_tensor(loss_chunk) else zero
    return weighted_loss_st + weighted_loss_rs + weighted_loss_chunk, {
        "weighted_loss_st": weighted_loss_st,
        "weighted_loss_rs": weighted_loss_rs,
        "weighted_loss_chunk_align": weighted_loss_chunk,
        "weighted_struct_total": weighted_loss_st + weighted_loss_rs + weighted_loss_chunk,
        "lambda_st": lambda_st,
        "lambda_rs": lambda_rs,
        "lambda_chunk": lambda_chunk,
        "w_st": 1.0,
        "w_rs": 1.0,
        "w_chunk": 1.0,
        "gamma": 1.0,
    }


def append_epoch_diagnostics_csv(path, row):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=GRADNORM_DIAGNOSTIC_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, 0.0) for column in GRADNORM_DIAGNOSTIC_COLUMNS})


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
    load_aux_images = (
        struct_cfg is not None
        and getattr(cfg.MODEL, "CLIP_MODEL", "") == "structxlip"
        and (
            is_gradnorm_enabled(cfg)
            or any(
                float(getattr(struct_cfg, key, 0.0)) != 0.0
                for key in (
                    "LAMBDA_STRUCTURE_TEXT",
                    "LAMBDA_RGB_STRUCTURE_CONSISTENCY",
                    "LAMBDA_CHUNK_ALIGN",
                )
            )
        )
    )
    common_kwargs = {
        "data_root": getattr(cfg.DATASET, "DATA_ROOT", ""),
        "image_size": int(cfg.DATASET.SIZE),
        "hflip_prob": float(getattr(cfg.DATASET, "HFLIP_PROB", 0.0)),
        "min_similarity": getattr(cfg.DATASET, "MIN_SIMILARITY", None),
        "use_original_caption_prefix": bool(getattr(cfg.DATASET, "USE_ORIGINAL_CAPTION_PREFIX", False)),
        "structure_image_field": getattr(struct_cfg, "STRUCTURE_IMAGE_FIELD", "filename_canny"),
        "chunk_top_k": int(getattr(struct_cfg, "CHUNK_TOP_K", 3)),
        "load_aux_images": load_aux_images,
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

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            if preds.ndim == 3:
                preds = preds.unsqueeze(1)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
            dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
            dice_scores.extend(dice.cpu().numpy())

    model.train()
    return mean(val_losses), mean(dice_scores), {
        "bce": mean(val_bce_losses),
        "dice_loss": mean(val_dice_losses),
    }


def main():
    cfg = get_arguments()
    cfg.DATASET.NAME = cfg.DATASET.NAME + f"_{cfg.data_percentage}" if cfg.data_percentage != 100 else cfg.DATASET.NAME
    if cfg.MODEL.CLIP_MODEL == "structxlip":
        cfg.DATASET.NAME = (
            cfg.DATASET.NAME
            + f"_st_{getattr(cfg.STRUCTXLIP, 'LAMBDA_STRUCTURE_TEXT', 0.0)}"
            + f"_rs_{getattr(cfg.STRUCTXLIP, 'LAMBDA_RGB_STRUCTURE_CONSISTENCY', 0.0)}"
            + f"_chunk_{getattr(cfg.STRUCTXLIP, 'LAMBDA_CHUNK_ALIGN', 0.0)}"
            + "_gradnorm"
        )

    run_output_dir = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}")
    os.makedirs(run_output_dir, exist_ok=True)
    diagnostics_path = os.path.join(run_output_dir, "structxlip_gradnorm_diagnostics.csv")

    logger = logger_config(os.path.join(run_output_dir, "log.txt"))
    logger.info("************")
    logger.info("** Config **")
    logger.info("************")
    logger.info(cfg)
    if cfg.seed >= 0:
        logger.info(f"Setting fixed seed: {cfg.seed}")
        set_random_seed(cfg.seed)
    if is_gradnorm_enabled(cfg):
        configure_gradnorm_second_order_attention()
        logger.info("GradNorm second-order gradients use math SDP attention kernels")

    ce_loss = BCEWithLogitsLoss()
    dice_loss = monai.losses.DiceLoss(include_background=False, sigmoid=True, reduction="mean")

    train_dataset, val_dataset = build_datasets_from_json(cfg)
    logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    def worker_init_fn(worker_id):
        seed = cfg.seed + worker_id
        random.seed(seed)
        np.random.seed(seed)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        num_workers=int(getattr(cfg.TRAIN, "WORKERS", 8)),
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        worker_init_fn=worker_init_fn,
        num_workers=int(getattr(cfg.TRAIN, "WORKERS", 8)),
        pin_memory=True,
    )

    if cfg.MODEL.CLIP_MODEL == "structxlip":
        model = build_structxlip(cfg)
    elif cfg.MODEL.CLIP_MODEL == "clip":
        model = build_clip(cfg)
    else:
        raise ValueError(f"Unsupported MODEL.CLIP_MODEL: {cfg.MODEL.CLIP_MODEL}")

    model.to(cfg.MODEL.DEVICE)
    model.train()
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    logger.info(f"Parameters to be updated: {trainable_names}")
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.TRAIN.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.TRAIN.NUM_EPOCHS,
        eta_min=1e-4,
    )

    gradnorm_cfg = get_gradnorm_config(cfg)
    gradnorm_enabled = is_gradnorm_enabled(cfg)
    gradnorm_controller = None
    gradnorm_optimizer = None
    gradnorm_parameters = []
    if gradnorm_enabled:
        gradnorm_controller = GradNormController(cfg, cfg.MODEL.DEVICE)
        gradnorm_optimizer = torch.optim.Adam([gradnorm_controller.weights], lr=gradnorm_cfg["weight_lr"])
        gradnorm_parameters = select_gradnorm_parameters(model)
        logger.info(
            "StructXLIP GradNorm enabled: "
            f"alpha={gradnorm_cfg['alpha']}, gamma={gradnorm_cfg['gamma']}, "
            f"weight_lr={gradnorm_cfg['weight_lr']}, eps={gradnorm_cfg['eps']}, "
            f"min/max weight={gradnorm_cfg['min_weight']}/{gradnorm_cfg['max_weight']}"
        )
    else:
        logger.info("StructXLIP GradNorm enabled: False")

    use_clip_loss = bool(getattr(cfg.TRAIN, "USE_CLIP_LOSS", True))
    clip_loss_weight = float(getattr(cfg.TRAIN, "CLIP_WEIGHT", 0.0))
    logger.info(f"CLIP auxiliary loss enabled: {use_clip_loss}, weight: {clip_loss_weight}")

    backbone_name = cfg.MODEL.BACKBONE.replace("/", "-")
    results_name = f"{cfg.DATASET.NAME}_Seg_{cfg.MODEL.CLIP_MODEL}_{backbone_name}"
    resume_path = os.path.join(run_output_dir, f"{results_name}_latest.pth")

    start_epoch = 0
    best_loss = float("inf")
    best_dice = -1.0
    if cfg.resume and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=cfg.MODEL.DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        move_optimizer_state_to_device(optimizer, cfg.MODEL.DEVICE)
        scheduler.load_state_dict(checkpoint.get("scheduler", {}))
        if gradnorm_controller is not None:
            gradnorm_controller.load_state_dict(checkpoint.get("gradnorm_controller", {}))
        if gradnorm_optimizer is not None and checkpoint.get("gradnorm_optimizer") is not None:
            gradnorm_optimizer.load_state_dict(checkpoint["gradnorm_optimizer"])
            move_optimizer_state_to_device(gradnorm_optimizer, cfg.MODEL.DEVICE)
        resume_lr = float(cfg.TRAIN.LEARNING_RATE)
        for group in optimizer.param_groups:
            group["lr"] = resume_lr
        scheduler.base_lrs = [resume_lr for _ in scheduler.base_lrs]
        if hasattr(scheduler, "_last_lr"):
            scheduler._last_lr = [resume_lr for _ in scheduler._last_lr]
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("best_loss", best_loss)
        best_dice = checkpoint.get("best_dice", best_dice)
        logger.info(f"Loaded checkpoint from epoch {start_epoch}, best loss: {best_loss:.4f}, best dice: {best_dice:.4f}")

    if cfg.MODEL.CLIP_MODEL == "structxlip" and start_epoch == 0:
        with open(diagnostics_path, "w", newline="") as csv_file:
            csv.DictWriter(csv_file, fieldnames=GRADNORM_DIAGNOSTIC_COLUMNS).writeheader()
        logger.info(f"StructXLIP GradNorm diagnostics CSV: {diagnostics_path}")

    for epoch in range(start_epoch, cfg.TRAIN.NUM_EPOCHS):
        epoch_losses = []
        epoch_seg_losses = []
        epoch_bce_losses = []
        epoch_dice_losses = []
        epoch_clip_losses = []
        epoch_weighted_clip_losses = []
        epoch_loss_st = []
        epoch_loss_rs = []
        epoch_loss_chunk_align = []
        epoch_struct_diagnostics = {
            column: [] for column in GRADNORM_DIAGNOSTIC_COLUMNS if column != "epoch"
        }

        for i, batch in enumerate(tqdm(train_dataloader)):
            model_kwargs = {
                "image": batch["image"].to(cfg.MODEL.DEVICE),
                "text": batch["text_prompt"],
                "return_clip_loss": (
                    gradnorm_enabled
                    or (use_clip_loss and clip_loss_weight != 0.0)
                    or cfg.MODEL.CLIP_MODEL == "structxlip"
                ),
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
            if cfg.MODEL.CLIP_MODEL == "structxlip" and len(model_outputs) == 3:
                seg_logits, clip_loss, _ = model_outputs
            else:
                seg_logits, clip_loss = model_outputs
            if not torch.isfinite(seg_logits).all():
                raise FloatingPointError(f"Non-finite logits at epoch={epoch + 1}, iter={i}, batch={describe_batch(batch)}")

            seg_loss, loss_parts = calc_loss(
                seg_logits,
                batch["ground_truth_mask"].to(cfg.MODEL.DEVICE),
                ce_loss,
                dice_loss,
                cfg,
                return_components=True,
            )
            weighted_clip_loss = seg_loss.new_zeros(())
            if use_clip_loss and clip_loss_weight != 0.0:
                weighted_clip_loss = clip_loss_weight * clip_loss
            main_loss = seg_loss + weighted_clip_loss
            loss = main_loss
            gradnorm_loss = None
            gradnorm_weight_grad = None
            struct_diag_tensors = {}

            if cfg.MODEL.CLIP_MODEL == "structxlip":
                raw_struct_losses = getattr(model, "last_structxlip_loss_tensors", {})
                if gradnorm_enabled:
                    weighted_struct_loss, gradnorm_loss, struct_diag_tensors = gradnorm_controller.compute(
                        raw_struct_losses,
                        gradnorm_parameters,
                    )
                    loss = loss + weighted_struct_loss
                    if gradnorm_loss.requires_grad:
                        gradnorm_weight_grad = torch.autograd.grad(
                            gradnorm_loss,
                            [gradnorm_controller.weights],
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                else:
                    weighted_struct_loss, struct_diag_tensors = fixed_struct_loss(model, seg_loss)
                    loss = loss + weighted_struct_loss

                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite total loss at epoch={epoch + 1}, iter={i}, batch={describe_batch(batch)}")

                zero = seg_loss.new_zeros(())
                loss_st_tensor = raw_struct_losses.get("loss_st", zero)
                loss_rs_tensor = raw_struct_losses.get("loss_rs", zero)
                loss_chunk_tensor = raw_struct_losses.get("loss_chunk_align", zero)
                struct_diag_values = {
                    "train_loss": safe_float(loss),
                    "seg_loss": safe_float(seg_loss),
                    "clip_loss": safe_float(clip_loss),
                    "weighted_clip_loss": safe_float(weighted_clip_loss),
                    "loss_st": safe_float(loss_st_tensor),
                    "loss_rs": safe_float(loss_rs_tensor),
                    "loss_chunk_align": safe_float(loss_chunk_tensor),
                    "struct_over_seg": safe_div(struct_diag_tensors.get("weighted_struct_total"), seg_loss),
                    "weighted_st_over_seg": safe_div(struct_diag_tensors.get("weighted_loss_st"), seg_loss),
                    "weighted_rs_over_seg": safe_div(struct_diag_tensors.get("weighted_loss_rs"), seg_loss),
                    "weighted_chunk_over_seg": safe_div(struct_diag_tensors.get("weighted_loss_chunk_align"), seg_loss),
                }
                struct_diag_values.update({key: safe_float(value) for key, value in struct_diag_tensors.items()})
                for key, value in struct_diag_values.items():
                    if key in epoch_struct_diagnostics:
                        epoch_struct_diagnostics[key].append(value)

            optimizer.zero_grad()
            if gradnorm_optimizer is not None:
                gradnorm_optimizer.zero_grad()
            loss.backward()
            if gradnorm_optimizer is not None:
                if gradnorm_weight_grad is not None:
                    gradnorm_controller.weights.grad = gradnorm_weight_grad.detach()
                else:
                    gradnorm_controller.weights.grad = None

            grad_clip_norm = float(getattr(cfg.TRAIN, "GRAD_CLIP_NORM", 1.0))
            if grad_clip_norm > 0:
                clip_params = list(filter(lambda p: p.requires_grad, model.parameters()))
                torch.nn.utils.clip_grad_norm_(clip_params, grad_clip_norm, error_if_nonfinite=True)
            optimizer.step()
            if gradnorm_optimizer is not None:
                gradnorm_optimizer.step()
                gradnorm_controller.renormalize_()

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

        scheduler.step()

        mean_epoch_loss = mean(epoch_losses)
        mean_epoch_seg_loss = mean(epoch_seg_losses)
        mean_epoch_bce_loss = mean(epoch_bce_losses)
        mean_epoch_dice_loss = mean(epoch_dice_losses)
        mean_epoch_clip_loss = mean(epoch_clip_losses)
        mean_epoch_weighted_clip_loss = mean(epoch_weighted_clip_losses)
        mean_epoch_loss_st = mean_or_zero(epoch_loss_st)
        mean_epoch_loss_rs = mean_or_zero(epoch_loss_rs)
        mean_epoch_loss_chunk_align = mean_or_zero(epoch_loss_chunk_align)
        struct_diag_means = {
            column: mean_or_zero(values)
            for column, values in epoch_struct_diagnostics.items()
        }

        mean_val_loss, mean_val_dice, val_loss_parts = evaluate_validation_loss(
            model,
            val_dataloader,
            cfg.MODEL.DEVICE,
            ce_loss,
            dice_loss,
            cfg,
        )

        log_msg = (
            f"EPOCH: {epoch + 1} | "
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
                f"Weighted st/rs/chunk: "
                f"{struct_diag_means.get('weighted_loss_st', 0.0):.4f}/"
                f"{struct_diag_means.get('weighted_loss_rs', 0.0):.4f}/"
                f"{struct_diag_means.get('weighted_loss_chunk_align', 0.0):.4f} | "
                f"GradNormLoss: {struct_diag_means.get('gradnorm_loss', 0.0):.4f} | "
                f"GradNorm st/rs/chunk: "
                f"{struct_diag_means.get('grad_norm_st', 0.0):.4f}/"
                f"{struct_diag_means.get('grad_norm_rs', 0.0):.4f}/"
                f"{struct_diag_means.get('grad_norm_chunk', 0.0):.4f} | "
                f"W st/rs/chunk: "
                f"{struct_diag_means.get('w_st', 0.0):.4f}/"
                f"{struct_diag_means.get('w_rs', 0.0):.4f}/"
                f"{struct_diag_means.get('w_chunk', 0.0):.4f} | "
                f"Lambda st/rs/chunk: "
                f"{struct_diag_means.get('lambda_st', 0.0):.4f}/"
                f"{struct_diag_means.get('lambda_rs', 0.0):.4f}/"
                f"{struct_diag_means.get('lambda_chunk', 0.0):.4f} | "
            )
        log_msg += (
            f"Val Total: {mean_val_loss:.4f} | "
            f"Val BCE: {val_loss_parts['bce']:.4f} | "
            f"Val DiceLoss: {val_loss_parts['dice_loss']:.4f} | "
            f"Val DiceMetric: {mean_val_dice:.4f}"
        )
        logger.info(log_msg)

        if cfg.MODEL.CLIP_MODEL == "structxlip":
            diag_row = {column: 0.0 for column in GRADNORM_DIAGNOSTIC_COLUMNS}
            diag_row.update(struct_diag_means)
            diag_row.update({
                "epoch": epoch + 1,
                "train_loss": mean_epoch_loss,
                "seg_loss": mean_epoch_seg_loss,
                "clip_loss": mean_epoch_clip_loss,
                "weighted_clip_loss": mean_epoch_weighted_clip_loss,
                "loss_st": mean_epoch_loss_st,
                "loss_rs": mean_epoch_loss_rs,
                "loss_chunk_align": mean_epoch_loss_chunk_align,
            })
            append_epoch_diagnostics_csv(diagnostics_path, diag_row)

        if mean_val_dice > best_dice:
            logger.info(f"New best Dice: {best_dice:.4f} -> {mean_val_dice:.4f}")
            best_dice = mean_val_dice
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "gradnorm_controller": gradnorm_controller.state_dict() if gradnorm_controller is not None else None,
                "gradnorm_optimizer": gradnorm_optimizer.state_dict() if gradnorm_optimizer is not None else None,
                "best_dice": best_dice,
            }, os.path.join(run_output_dir, f"{results_name}_best_dice.pth"))
        else:
            logger.info(f"Dice: {mean_val_dice:.4f}")

        best_loss = min(best_loss, mean_val_loss)
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "gradnorm_controller": gradnorm_controller.state_dict() if gradnorm_controller is not None else None,
            "gradnorm_optimizer": gradnorm_optimizer.state_dict() if gradnorm_optimizer is not None else None,
            "best_loss": best_loss,
            "best_dice": best_dice,
        }, os.path.join(run_output_dir, f"{results_name}_latest.pth"))


if __name__ == "__main__":
    main()
