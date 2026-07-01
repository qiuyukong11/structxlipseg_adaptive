
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
from utils.main_utils import load_cfg_from_cfg_file


LOSS_TASKS = ("seg", "clip", "st", "rs", "chunk")
LOSS_KEYS = ("loss_seg", "loss_clip", "loss_st", "loss_rs", "loss_chunk_align")
STRUCT_LOSS_KEYS = ("loss_st", "loss_rs", "loss_chunk_align")

AUTOLAMBDA_DIAGNOSTIC_COLUMNS = [
    "epoch",
    "train_loss",
    "seg_loss",
    "clip_loss",
    "weighted_clip_loss",
    "loss_seg",
    "loss_clip",
    "weighted_loss_seg",
    "weighted_loss_clip",
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
    "weight_method",
    "grad_method",
    "weight_seg",
    "weight_clip",
    "weight_st",
    "weight_rs",
    "weight_chunk",
    "log_sigma_seg",
    "log_sigma_clip",
    "log_sigma_st",
    "log_sigma_rs",
    "log_sigma_chunk",
    "autol_meta_grad_seg",
    "autol_meta_grad_clip",
    "autol_meta_grad_st",
    "autol_meta_grad_rs",
    "autol_meta_grad_chunk",
    "dwa_ratio_seg",
    "dwa_ratio_clip",
    "dwa_ratio_st",
    "dwa_ratio_rs",
    "dwa_ratio_chunk",
]


WEIGHT_METHOD_TO_ID = {
    "equal": 0.0,
    "uncert": 1.0,
    "dwa": 2.0,
    "autol": 3.0,
}

GRAD_METHOD_TO_ID = {
    "none": 0.0,
    "graddrop": 1.0,
    "pcgrad": 2.0,
    "cagrad": 3.0,
}


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
        default="configs/sketchy_structxlipseg_5percent_debug.yaml",
        type=str,
        help="Path to config file",
    )
    parser.add_argument("--resume", action="store_true", help="Whether to resume training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--data_percentage", type=int, default=100, help="Percentage of data to use.")
    parser.add_argument("--output-dir", type=str, default="output", help="output directory")
    parser.add_argument(
        "--weight-method",
        choices=("equal", "uncert", "dwa", "autol"),
        default="autol",
        help="Weighting method from Auto-Lambda repo.",
    )
    parser.add_argument(
        "--grad-method",
        choices=("none", "graddrop", "pcgrad", "cagrad"),
        default="none",
        help="Gradient manipulation method from Auto-Lambda repo.",
    )
    parser.add_argument("--autol-init", type=float, default=0.1, help="Initial Auto-Lambda meta weight.")
    parser.add_argument("--autol-lr", type=float, default=1e-4, help="Auto-Lambda meta-weight learning rate.")
    parser.add_argument("--autol-val-batches", type=int, default=4, help="Validation batches averaged for each Auto-Lambda meta step.")
    parser.add_argument("--dwa-temperature", type=float, default=2.0, help="Dynamic Weight Average temperature.")
    parser.add_argument("--uncert-init", type=float, default=-0.7, help="Initial log sigma for uncertainty weighting.")
    parser.add_argument("--cagrad-alpha", type=float, default=0.4, help="CAGrad alpha.")
    parser.add_argument("--save-every-checkpoint", action="store_true", help="Save a checkpoint for every epoch.")
    parser.add_argument("--clamp-aux-loss-nonneg", action="store_true", default=True)
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
    disable_other_adaptive_strategies(cfg)
    return cfg


def disable_other_adaptive_strategies(cfg):
    if getattr(getattr(cfg, "MODEL", None), "CLIP_MODEL", "") != "structxlip":
        return
    struct_cfg = getattr(cfg, "STRUCTXLIP", None)
    if struct_cfg is None:
        return
    for section in (
        "ADAPTIVE_3LOSS",
        "ADAPTIVE_V2",
        "ADAPTIVE_V3",
        "ADAPTIVE_V4",
        "ADAPTIVE_V6",
        "ADAPTIVE_V7",
        "ADAPTIVE_GRADNORM",
        "ADAPTIVE_GRADBUDGET_ALIGN",
        "ADAPTIVE_NORM_BALANCED",
        "LEARNABLE_TAU_LOSS",
    ):
        if section in struct_cfg and isinstance(struct_cfg[section], dict):
            struct_cfg[section]["ENABLED"] = False


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


def get_trainable_model_parameters(model):
    return [param for param in model.parameters() if param.requires_grad]


def select_meta_gradient_parameters(model):
    adapters = getattr(model, "pvl_adapters", None)
    if adapters is not None and len(adapters) > 0:
        params = [param for param in adapters[-1].parameters() if param.requires_grad]
        if params:
            return params
    return get_trainable_model_parameters(model)


class StructWeightController:
    def __init__(self, cfg, device):
        self.method = str(getattr(cfg, "weight_method", "autol")).lower()
        self.num_tasks = len(LOSS_TASKS)
        self.device = torch.device(device)
        self.dwa_temperature = float(getattr(cfg, "dwa_temperature", 2.0))
        self.clamp_aux_loss_nonneg = bool(getattr(cfg, "clamp_aux_loss_nonneg", True))
        self.epoch_weights = torch.ones(self.num_tasks, device=self.device)
        self.previous_epoch_losses = []
        self.last_meta_grad = torch.zeros(self.num_tasks, device=self.device)
        self.last_dwa_ratio = torch.ones(self.num_tasks, device=self.device)

        self.logsigma = None
        self.meta_weights = None
        if self.method == "uncert":
            init = float(getattr(cfg, "uncert_init", -0.7))
            self.logsigma = torch.nn.Parameter(torch.full((self.num_tasks,), init, device=self.device))
        elif self.method == "autol":
            init = float(getattr(cfg, "autol_init", 0.1))
            self.meta_weights = torch.nn.Parameter(torch.full((self.num_tasks,), init, device=self.device))

    def parameters(self):
        if self.logsigma is not None:
            return [self.logsigma]
        if self.meta_weights is not None:
            return [self.meta_weights]
        return []

    def state_dict(self):
        return {
            "method": self.method,
            "epoch_weights": self.epoch_weights.detach().cpu(),
            "previous_epoch_losses": [losses.detach().cpu() for losses in self.previous_epoch_losses],
            "last_meta_grad": self.last_meta_grad.detach().cpu(),
            "last_dwa_ratio": self.last_dwa_ratio.detach().cpu(),
            "logsigma": None if self.logsigma is None else self.logsigma.detach().cpu(),
            "meta_weights": None if self.meta_weights is None else self.meta_weights.detach().cpu(),
        }

    def load_state_dict(self, state):
        if not state:
            return
        if torch.is_tensor(state.get("epoch_weights")):
            self.epoch_weights = state["epoch_weights"].to(self.device)
        if torch.is_tensor(state.get("last_meta_grad")):
            self.last_meta_grad = state["last_meta_grad"].to(self.device)
        if torch.is_tensor(state.get("last_dwa_ratio")):
            self.last_dwa_ratio = state["last_dwa_ratio"].to(self.device)
        previous = []
        for losses in state.get("previous_epoch_losses", []):
            if torch.is_tensor(losses):
                previous.append(losses.to(self.device))
        self.previous_epoch_losses = previous
        if self.logsigma is not None and torch.is_tensor(state.get("logsigma")):
            self.logsigma.data.copy_(state["logsigma"].to(self.device, dtype=self.logsigma.dtype))
        if self.meta_weights is not None and torch.is_tensor(state.get("meta_weights")):
            self.meta_weights.data.copy_(state["meta_weights"].to(self.device, dtype=self.meta_weights.dtype))

    def start_epoch(self):
        if self.method != "dwa":
            return
        if len(self.previous_epoch_losses) < 2:
            self.epoch_weights = torch.ones(self.num_tasks, device=self.device)
            self.last_dwa_ratio = torch.ones(self.num_tasks, device=self.device)
            return
        eps = 1e-8
        ratio = self.previous_epoch_losses[-1] / self.previous_epoch_losses[-2].clamp_min(eps)
        ratio = torch.where(torch.isfinite(ratio), ratio, torch.ones_like(ratio))
        temperature = max(float(self.dwa_temperature), eps)
        weights = self.num_tasks * torch.softmax(ratio / temperature, dim=0)
        self.epoch_weights = weights.to(self.device)
        self.last_dwa_ratio = ratio.to(self.device)

    def finish_epoch(self, raw_loss_means):
        losses = torch.as_tensor(raw_loss_means, device=self.device, dtype=torch.float32).clamp_min(1e-8)
        self.previous_epoch_losses.append(losses.detach())
        if len(self.previous_epoch_losses) > 3:
            self.previous_epoch_losses = self.previous_epoch_losses[-3:]

    def _loss_vector(self, raw_losses):
        losses = []
        ref = next((raw_losses.get(key) for key in LOSS_KEYS if torch.is_tensor(raw_losses.get(key))), None)
        if ref is None:
            raise RuntimeError("StructWeightController expected all-loss tensors")
        zero = ref.new_zeros(())
        for key in LOSS_KEYS:
            loss = raw_losses.get(key, zero)
            if not torch.is_tensor(loss):
                loss = zero
            if self.clamp_aux_loss_nonneg:
                loss = torch.clamp(loss, min=0.0)
            losses.append(loss)
        return torch.stack(losses)

    def compute(self, raw_losses):
        loss_vec = self._loss_vector(raw_losses)
        diagnostics = {}
        if self.method in ("equal", "dwa"):
            weights = self.epoch_weights.to(loss_vec.device, dtype=loss_vec.dtype)
            weighted_losses = weights * loss_vec
            effective_weights = weights
            log_sigma = torch.zeros_like(loss_vec)
        elif self.method == "uncert":
            log_sigma = self.logsigma.to(loss_vec.device, dtype=loss_vec.dtype)
            effective_weights = 1.0 / (2.0 * torch.exp(log_sigma))
            weighted_losses = effective_weights * loss_vec + 0.5 * log_sigma
        elif self.method == "autol":
            effective_weights = self.meta_weights.to(loss_vec.device, dtype=loss_vec.dtype)
            weighted_losses = effective_weights * loss_vec
            log_sigma = torch.zeros_like(loss_vec)
        else:
            raise ValueError(f"Unsupported weight method: {self.method}")

        weighted_struct_total = weighted_losses.sum()
        diagnostics.update({
            "weighted_loss_seg": weighted_losses[0],
            "weighted_loss_clip": weighted_losses[1],
            "weighted_loss_st": weighted_losses[2],
            "weighted_loss_rs": weighted_losses[3],
            "weighted_loss_chunk_align": weighted_losses[4],
            "weighted_struct_total": weighted_struct_total,
            "weight_method": WEIGHT_METHOD_TO_ID.get(self.method, -1.0),
            "weight_seg": effective_weights[0],
            "weight_clip": effective_weights[1],
            "weight_st": effective_weights[2],
            "weight_rs": effective_weights[3],
            "weight_chunk": effective_weights[4],
            "log_sigma_seg": log_sigma[0],
            "log_sigma_clip": log_sigma[1],
            "log_sigma_st": log_sigma[2],
            "log_sigma_rs": log_sigma[3],
            "log_sigma_chunk": log_sigma[4],
            "autol_meta_grad_seg": self.last_meta_grad[0],
            "autol_meta_grad_clip": self.last_meta_grad[1],
            "autol_meta_grad_st": self.last_meta_grad[2],
            "autol_meta_grad_rs": self.last_meta_grad[3],
            "autol_meta_grad_chunk": self.last_meta_grad[4],
            "dwa_ratio_seg": self.last_dwa_ratio[0],
            "dwa_ratio_clip": self.last_dwa_ratio[1],
            "dwa_ratio_st": self.last_dwa_ratio[2],
            "dwa_ratio_rs": self.last_dwa_ratio[3],
            "dwa_ratio_chunk": self.last_dwa_ratio[4],
        })
        return weighted_struct_total, [weighted_losses[i] for i in range(self.num_tasks)], diagnostics


def flatten_grads(grads, params):
    flat = []
    for grad, param in zip(grads, params):
        if grad is None:
            flat.append(torch.zeros_like(param, dtype=torch.float32).reshape(-1))
        else:
            flat.append(grad.detach().reshape(-1).float())
    if not flat:
        return torch.empty(0)
    return torch.cat(flat)


def grad_vector(loss, params, retain_graph=True):
    if not torch.is_tensor(loss) or not loss.requires_grad:
        return torch.zeros(sum(param.numel() for param in params), device=params[0].device if params else "cpu")
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    return flatten_grads(grads, params)


def graddrop(grads):
    probability = 0.5 * (1.0 + grads.sum(1) / (grads.abs().sum(1) + 1e-8))
    uniform = torch.rand_like(grads[:, 0])
    mask = probability.gt(uniform).view(-1, 1) * grads.gt(0) + probability.lt(uniform).view(-1, 1) * grads.lt(0)
    return (grads * mask.float()).mean(1)


def pcgrad(grads, rng, num_tasks):
    grad_vec = grads.t()
    shuffled_task_indices = np.zeros((num_tasks, num_tasks - 1), dtype=int)
    for i in range(num_tasks):
        task_indices = np.arange(num_tasks)
        task_indices[i] = task_indices[-1]
        shuffled_task_indices[i] = task_indices[:-1]
        rng.shuffle(shuffled_task_indices[i])
    shuffled_task_indices = shuffled_task_indices.T

    normalized_grad_vec = grad_vec / (grad_vec.norm(dim=1, keepdim=True) + 1e-8)
    modified_grad_vec = grad_vec.clone()
    for task_indices in shuffled_task_indices:
        normalized_shuffled_grad = normalized_grad_vec[task_indices]
        dot = (modified_grad_vec * normalized_shuffled_grad).sum(dim=1, keepdim=True)
        modified_grad_vec -= torch.clamp_max(dot, 0) * normalized_shuffled_grad
    return modified_grad_vec.mean(dim=0)


def cagrad(grads, num_tasks, alpha=0.5, rescale=1):
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise ImportError("CAGrad requires scipy. Please install scipy or use another grad method.") from exc
    gram = grads.t().mm(grads).cpu()
    g0_norm = (gram.mean() + 1e-8).sqrt()
    x_start = np.ones(num_tasks) / num_tasks
    bounds = tuple((0, 1) for _ in x_start)
    cons = ({"type": "eq", "fun": lambda x: 1 - sum(x)})
    matrix = gram.numpy()
    base = x_start.copy()
    c_value = (alpha * g0_norm + 1e-8).item()

    def objfn(x):
        x = x.reshape(1, num_tasks)
        return (x.dot(matrix).dot(base.reshape(num_tasks, 1)) + c_value * np.sqrt(x.dot(matrix).dot(x.T) + 1e-8)).sum()

    result = minimize(objfn, x_start, bounds=bounds, constraints=cons)
    weights = torch.as_tensor(result.x, device=grads.device, dtype=grads.dtype)
    gw = (grads * weights.view(1, -1)).sum(1)
    lmbda = c_value / (gw.norm() + 1e-8)
    combined = grads.mean(1) + lmbda * gw
    if rescale == 0:
        return combined
    if rescale == 1:
        return combined / (1 + alpha ** 2)
    return combined / (1 + alpha)


def set_flat_grads(params, flat_grad, scale=1.0):
    offset = 0
    for param in params:
        numel = param.numel()
        grad = flat_grad[offset:offset + numel].view_as(param).to(param.device, dtype=param.dtype)
        param.grad = (scale * grad).clone()
        offset += numel


def apply_gradient_method(loss_terms, model_params, grad_method, rng, cagrad_alpha):
    num_tasks = len(loss_terms)
    if grad_method == "none" or num_tasks <= 1:
        sum(loss_terms).backward()
        return
    grad_columns = [grad_vector(loss, model_params, retain_graph=True) for loss in loss_terms]
    grads = torch.stack(grad_columns, dim=1)
    if grad_method == "graddrop":
        combined = graddrop(grads)
    elif grad_method == "pcgrad":
        combined = pcgrad(grads, rng, num_tasks)
    elif grad_method == "cagrad":
        combined = cagrad(grads, num_tasks, alpha=cagrad_alpha, rescale=1)
    else:
        raise ValueError(f"Unsupported grad method: {grad_method}")
    set_flat_grads(model_params, combined, scale=float(num_tasks))


def set_controller_grads(controller, loss):
    params = controller.parameters()
    if not params:
        return
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    for param, grad in zip(params, grads):
        param.grad = None if grad is None else grad.detach().clone()


def append_epoch_diagnostics_csv(path, row):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=AUTOLAMBDA_DIAGNOSTIC_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, 0.0) for column in AUTOLAMBDA_DIAGNOSTIC_COLUMNS})


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
    load_aux_images = struct_cfg is not None and getattr(cfg.MODEL, "CLIP_MODEL", "") == "structxlip"
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
        return (
            JsonRefSegDataset(train_json, train=True, **common_kwargs),
            JsonRefSegDataset(val_json, train=False, **common_kwargs),
        )
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
    return (
        JsonRefSegDataset(train_json, train=True, samples=train_samples, **common_kwargs),
        JsonRefSegDataset(train_json, train=False, samples=val_samples, **common_kwargs),
    )


def make_model_kwargs(batch, cfg, return_clip_loss=True):
    kwargs = {
        "image": batch["image"].to(cfg.MODEL.DEVICE),
        "text": batch["text_prompt"],
        "return_clip_loss": return_clip_loss,
    }
    if cfg.MODEL.CLIP_MODEL == "structxlip":
        structure_image = batch["original_structure_image"] if "original_structure_image" in batch else batch["structure_image"]
        has_structure = batch["has_original_structure"] if "has_original_structure" in batch else batch["has_structure"]
        original_text = batch["original_text"] if "original_text" in batch else batch["text_prompt"]
        kwargs.update({
            "structure_image": structure_image.to(cfg.MODEL.DEVICE),
            "edge_images": batch["edge_images"].to(cfg.MODEL.DEVICE),
            "has_structure": has_structure.to(cfg.MODEL.DEVICE),
            "edge_valid_mask": batch["edge_valid_mask"].to(cfg.MODEL.DEVICE),
            "original_text": original_text,
        })
    return kwargs


def forward_losses(model, batch, cfg, ce_loss, dice_loss, use_clip_loss, clip_loss_weight, return_clip_loss=True):
    model_outputs = model(**make_model_kwargs(batch, cfg, return_clip_loss=return_clip_loss))
    if cfg.MODEL.CLIP_MODEL == "structxlip" and len(model_outputs) == 3:
        seg_logits, clip_loss, _ = model_outputs
    else:
        seg_logits, clip_loss = model_outputs
    if not torch.isfinite(seg_logits).all():
        raise FloatingPointError(f"Non-finite logits for batch={describe_batch(batch)}")
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
    return {
        "seg_logits": seg_logits,
        "clip_loss": clip_loss,
        "seg_loss": seg_loss,
        "loss_parts": loss_parts,
        "weighted_clip_loss": weighted_clip_loss,
        "main_loss": main_loss,
        "raw_struct_losses": getattr(model, "last_structxlip_loss_tensors", {}),
    }


def build_all_loss_terms(data, cfg):
    raw_struct_losses = data["raw_struct_losses"]
    zero = data["seg_loss"].new_zeros(())
    struct_cfg = getattr(cfg, "STRUCTXLIP", None)
    lambda_st = float(getattr(struct_cfg, "LAMBDA_STRUCTURE_TEXT", 0.0)) if struct_cfg is not None else 0.0
    lambda_rs = float(getattr(struct_cfg, "LAMBDA_RGB_STRUCTURE_CONSISTENCY", 0.0)) if struct_cfg is not None else 0.0
    lambda_chunk = float(getattr(struct_cfg, "LAMBDA_CHUNK_ALIGN", 0.0)) if struct_cfg is not None else 0.0
    loss_st = raw_struct_losses.get("loss_st", zero)
    loss_rs = raw_struct_losses.get("loss_rs", zero)
    loss_chunk = raw_struct_losses.get("loss_chunk_align", zero)
    return {
        "loss_seg": data["seg_loss"],
        "loss_clip": data["weighted_clip_loss"],
        "loss_st": lambda_st * (loss_st if torch.is_tensor(loss_st) else zero),
        "loss_rs": lambda_rs * (loss_rs if torch.is_tensor(loss_rs) else zero),
        "loss_chunk_align": lambda_chunk * (loss_chunk if torch.is_tensor(loss_chunk) else zero),
    }


def next_val_batch(val_iter, val_dataloader):
    try:
        return next(val_iter), val_iter
    except StopIteration:
        val_iter = iter(val_dataloader)
        return next(val_iter), val_iter


def _train_objective_for_autolambda(model, controller, batch, cfg, ce_loss, dice_loss, use_clip_loss, clip_loss_weight):
    data = forward_losses(model, batch, cfg, ce_loss, dice_loss, use_clip_loss, clip_loss_weight, return_clip_loss=True)
    loss_terms = build_all_loss_terms(data, cfg)
    losses = controller._loss_vector(loss_terms)
    weights = controller.meta_weights.to(losses.device, dtype=losses.dtype)
    return (weights * losses).sum()


def _grad_or_zero(loss, params, retain_graph=False):
    return torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)


def _zero_meta_grad_like(controller):
    return torch.zeros_like(controller.meta_weights)


def _meta_weight_grad(model, controller, train_batch, cfg, ce_loss, dice_loss, use_clip_loss, clip_loss_weight):
    loss = _train_objective_for_autolambda(
        model,
        controller,
        train_batch,
        cfg,
        ce_loss,
        dice_loss,
        use_clip_loss,
        clip_loss_weight,
    )
    grad = torch.autograd.grad(loss, controller.meta_weights, allow_unused=True)[0]
    if grad is None:
        return _zero_meta_grad_like(controller)
    return grad


def _compute_autolambda_hessian(model, controller, train_batch, cfg, ce_loss, dice_loss, use_clip_loss, clip_loss_weight, params, d_model):
    valid_pairs = [(param, grad) for param, grad in zip(params, d_model) if grad is not None]
    if not valid_pairs:
        return _zero_meta_grad_like(controller)
    norm = torch.cat([grad.detach().reshape(-1).float() for _, grad in valid_pairs]).norm()
    if not torch.isfinite(norm) or norm.item() <= 0:
        return _zero_meta_grad_like(controller)
    eps = 0.01 / norm
    originals = [param.detach().clone() for param, _ in valid_pairs]
    try:
        with torch.no_grad():
            for param, grad in valid_pairs:
                param.add_(eps.to(param.device, dtype=param.dtype) * grad.to(param.device, dtype=param.dtype))
        d_weight_p = _meta_weight_grad(
            model,
            controller,
            train_batch,
            cfg,
            ce_loss,
            dice_loss,
            use_clip_loss,
            clip_loss_weight,
        )

        with torch.no_grad():
            for param, grad in valid_pairs:
                param.add_(-2.0 * eps.to(param.device, dtype=param.dtype) * grad.to(param.device, dtype=param.dtype))
        d_weight_n = _meta_weight_grad(
            model,
            controller,
            train_batch,
            cfg,
            ce_loss,
            dice_loss,
            use_clip_loss,
            clip_loss_weight,
        )
    finally:
        with torch.no_grad():
            for param, original in zip((pair[0] for pair in valid_pairs), originals):
                param.copy_(original)
    return (d_weight_p - d_weight_n) / (2.0 * eps.to(d_weight_p.device, dtype=d_weight_p.dtype))


def auto_lambda_meta_step(model, controller, meta_optimizer, model_optimizer, train_batch, val_batches, cfg, ce_loss, dice_loss, use_clip_loss, clip_loss_weight, model_params):
    if controller.method != "autol" or controller.meta_weights is None:
        return
    lr = float(model_optimizer.param_groups[0].get("lr", getattr(cfg.TRAIN, "LEARNING_RATE", 1e-4)))
    params = list(model_params)
    if not params:
        return

    train_loss = _train_objective_for_autolambda(
        model,
        controller,
        train_batch,
        cfg,
        ce_loss,
        dice_loss,
        use_clip_loss,
        clip_loss_weight,
    )
    train_grads = _grad_or_zero(train_loss, params)
    originals = [param.detach().clone() for param in params]
    weight_decay = float(model_optimizer.param_groups[0].get("weight_decay", 0.0))
    try:
        with torch.no_grad():
            for param, grad in zip(params, train_grads):
                if grad is None:
                    continue
                update = grad
                if weight_decay != 0.0:
                    update = update + weight_decay * param
                param.copy_(param - lr * update)

        val_losses = []
        for val_batch in val_batches:
            val_data = forward_losses(model, val_batch, cfg, ce_loss, dice_loss, False, 0.0, return_clip_loss=False)
            val_losses.append(val_data["seg_loss"])
        if not val_losses:
            return
        val_loss = torch.stack(val_losses).mean()
        d_model = _grad_or_zero(val_loss, params)
    finally:
        with torch.no_grad():
            for param, original in zip(params, originals):
                param.copy_(original)

    hessian = _compute_autolambda_hessian(
        model,
        controller,
        train_batch,
        cfg,
        ce_loss,
        dice_loss,
        use_clip_loss,
        clip_loss_weight,
        params,
        d_model,
    )
    meta_grad = (-lr * hessian).detach().to(controller.meta_weights.device, dtype=controller.meta_weights.dtype)
    controller.last_meta_grad = meta_grad.detach()
    meta_optimizer.zero_grad()
    controller.meta_weights.grad = meta_grad
    meta_optimizer.step()


def evaluate_validation_loss(model, val_dataloader, device, ce_loss, dice_loss, cfg):
    model.eval()
    val_losses = []
    val_bce_losses = []
    val_dice_losses = []
    iou_scores = []
    total_intersection = 0.0
    total_union = 0.0
    eps = 1e-7
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            images = batch["image"].to(device)
            masks = batch["ground_truth_mask"].to(device)
            logits = model(images, text=batch["text_prompt"], num_samples=1)[0]
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
            masks = (masks > 0).float()
            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) - intersection
            iou = torch.where(union > 0, intersection / (union + eps), torch.ones_like(union))
            iou_scores.extend(iou.cpu().numpy())
            total_intersection += float(intersection.sum().item())
            total_union += float(union.sum().item())
    model.train()
    ciou = 1.0 if total_union <= 0.0 else total_intersection / total_union
    return mean(val_losses), mean(iou_scores), {
        "bce": mean(val_bce_losses),
        "dice_loss": mean(val_dice_losses),
        "ciou": ciou,
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
            + f"_autolambda_allloss_{cfg.weight_method}_{cfg.grad_method}"
        )

    run_output_dir = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}")
    os.makedirs(run_output_dir, exist_ok=True)
    diagnostics_path = os.path.join(run_output_dir, "structxlip_autolambda_diagnostics.csv")
    logger = logger_config(os.path.join(run_output_dir, "log.txt"))
    logger.info("************")
    logger.info("** Config **")
    logger.info("************")
    logger.info(cfg)
    if cfg.seed >= 0:
        logger.info(f"Setting fixed seed: {cfg.seed}")
        set_random_seed(cfg.seed)

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

    controller = StructWeightController(cfg, cfg.MODEL.DEVICE)
    model_params = get_trainable_model_parameters(model)
    optimizer = torch.optim.Adam(model_params, lr=cfg.TRAIN.LEARNING_RATE)
    weight_optimizer = None
    if controller.method == "uncert" and controller.parameters():
        weight_optimizer = torch.optim.Adam(controller.parameters(), lr=cfg.TRAIN.LEARNING_RATE)
    meta_optimizer = None
    if controller.method == "autol" and controller.parameters():
        meta_optimizer = torch.optim.Adam(controller.parameters(), lr=float(getattr(cfg, "autol_lr", 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAIN.NUM_EPOCHS, eta_min=1e-4)

    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    logger.info(f"Parameters to be updated: {trainable_names}")
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logger.info(f"StructXLIP all-loss weight method: {controller.method}; grad method: {cfg.grad_method}")
    logger.info(f"Auto-Lambda init/lr/val_batches: {cfg.autol_init}/{cfg.autol_lr}/{cfg.autol_val_batches}; DWA T: {cfg.dwa_temperature}; CAGrad alpha: {cfg.cagrad_alpha}")

    use_clip_loss = bool(getattr(cfg.TRAIN, "USE_CLIP_LOSS", True))
    clip_loss_weight = float(getattr(cfg.TRAIN, "CLIP_WEIGHT", 0.0))
    logger.info(f"CLIP auxiliary loss enabled: {use_clip_loss}, weight: {clip_loss_weight}")

    backbone_name = cfg.MODEL.BACKBONE.replace("/", "-")
    results_name = f"{cfg.DATASET.NAME}_Seg_{cfg.MODEL.CLIP_MODEL}_{backbone_name}"
    resume_path = os.path.join(run_output_dir, f"{results_name}_latest.pth")
    start_epoch = 0
    best_loss = float("inf")
    best_miou = -1.0
    if cfg.resume and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=cfg.MODEL.DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        move_optimizer_state_to_device(optimizer, cfg.MODEL.DEVICE)
        scheduler.load_state_dict(checkpoint.get("scheduler", {}))
        controller.load_state_dict(checkpoint.get("struct_weight_controller", {}))
        if weight_optimizer is not None and checkpoint.get("weight_optimizer") is not None:
            weight_optimizer.load_state_dict(checkpoint["weight_optimizer"])
            move_optimizer_state_to_device(weight_optimizer, cfg.MODEL.DEVICE)
        if meta_optimizer is not None and checkpoint.get("meta_optimizer") is not None:
            meta_optimizer.load_state_dict(checkpoint["meta_optimizer"])
            move_optimizer_state_to_device(meta_optimizer, cfg.MODEL.DEVICE)
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("best_loss", best_loss)
        best_miou = checkpoint.get("best_miou", -1.0)
        logger.info(f"Loaded checkpoint from epoch {start_epoch}, best loss: {best_loss:.4f}, best mIoU: {best_miou:.4f}")
        if "best_miou" not in checkpoint and "best_dice" in checkpoint:
            logger.info(f"Previous checkpoint best Dice {checkpoint['best_dice']:.4f} is not reused for mIoU-based selection")

    if cfg.MODEL.CLIP_MODEL == "structxlip" and start_epoch == 0:
        with open(diagnostics_path, "w", newline="") as csv_file:
            csv.DictWriter(csv_file, fieldnames=AUTOLAMBDA_DIAGNOSTIC_COLUMNS).writeheader()
        logger.info(f"StructXLIP all-loss adaptive diagnostics CSV: {diagnostics_path}")

    rng = np.random.default_rng(int(cfg.seed) if int(cfg.seed) >= 0 else None)
    val_iter = iter(val_dataloader)

    for epoch in range(start_epoch, cfg.TRAIN.NUM_EPOCHS):
        controller.start_epoch()
        epoch_losses = []
        epoch_seg_losses = []
        epoch_bce_losses = []
        epoch_dice_losses = []
        epoch_clip_losses = []
        epoch_weighted_clip_losses = []
        epoch_loss_st = []
        epoch_loss_rs = []
        epoch_loss_chunk_align = []
        epoch_struct_diagnostics = {column: [] for column in AUTOLAMBDA_DIAGNOSTIC_COLUMNS if column != "epoch"}

        for i, batch in enumerate(tqdm(train_dataloader)):
            if controller.method == "autol" and meta_optimizer is not None:
                val_batches = []
                for _ in range(max(1, int(getattr(cfg, "autol_val_batches", 1)))):
                    val_batch, val_iter = next_val_batch(val_iter, val_dataloader)
                    val_batches.append(val_batch)
                auto_lambda_meta_step(
                    model,
                    controller,
                    meta_optimizer,
                    optimizer,
                    batch,
                    val_batches,
                    cfg,
                    ce_loss,
                    dice_loss,
                    use_clip_loss,
                    clip_loss_weight,
                    model_params,
                )

            data = forward_losses(model, batch, cfg, ce_loss, dice_loss, use_clip_loss, clip_loss_weight, return_clip_loss=True)
            raw_struct_losses = data["raw_struct_losses"]
            all_loss_terms = build_all_loss_terms(data, cfg)
            weighted_struct_loss, struct_loss_terms, struct_diag_tensors = controller.compute(all_loss_terms)
            loss = weighted_struct_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite total loss at epoch={epoch + 1}, iter={i}, batch={describe_batch(batch)}")

            zero = data["seg_loss"].new_zeros(())
            loss_st_tensor = raw_struct_losses.get("loss_st", zero)
            loss_rs_tensor = raw_struct_losses.get("loss_rs", zero)
            loss_chunk_tensor = raw_struct_losses.get("loss_chunk_align", zero)
            struct_diag_values = {
                "train_loss": safe_float(loss),
                "seg_loss": safe_float(data["seg_loss"]),
                "clip_loss": safe_float(data["clip_loss"]),
                "weighted_clip_loss": safe_float(data["weighted_clip_loss"]),
                "loss_seg": safe_float(all_loss_terms.get("loss_seg")),
                "loss_clip": safe_float(all_loss_terms.get("loss_clip")),
                "loss_st": safe_float(loss_st_tensor),
                "loss_rs": safe_float(loss_rs_tensor),
                "loss_chunk_align": safe_float(loss_chunk_tensor),
                "struct_over_seg": safe_div(struct_diag_tensors.get("weighted_struct_total"), data["seg_loss"]),
                "weighted_st_over_seg": safe_div(struct_diag_tensors.get("weighted_loss_st"), data["seg_loss"]),
                "weighted_rs_over_seg": safe_div(struct_diag_tensors.get("weighted_loss_rs"), data["seg_loss"]),
                "weighted_chunk_over_seg": safe_div(struct_diag_tensors.get("weighted_loss_chunk_align"), data["seg_loss"]),
                "grad_method": GRAD_METHOD_TO_ID.get(cfg.grad_method, -1.0),
            }
            struct_diag_values.update({key: safe_float(value) for key, value in struct_diag_tensors.items()})
            for key, value in struct_diag_values.items():
                if key in epoch_struct_diagnostics:
                    epoch_struct_diagnostics[key].append(value)

            optimizer.zero_grad()
            if weight_optimizer is not None:
                weight_optimizer.zero_grad()
            if meta_optimizer is not None:
                meta_optimizer.zero_grad()

            grad_loss_terms = struct_loss_terms
            if cfg.grad_method == "none":
                loss.backward()
            else:
                apply_gradient_method(grad_loss_terms, model_params, cfg.grad_method, rng, float(cfg.cagrad_alpha))
                set_controller_grads(controller, loss)

            grad_clip_norm = float(getattr(cfg.TRAIN, "GRAD_CLIP_NORM", 1.0))
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model_params, grad_clip_norm, error_if_nonfinite=True)
            optimizer.step()
            if weight_optimizer is not None:
                weight_optimizer.step()

            epoch_losses.append(loss.item())
            epoch_seg_losses.append(data["loss_parts"]["seg_loss"].item())
            epoch_bce_losses.append(data["loss_parts"]["bce"].item())
            epoch_dice_losses.append(data["loss_parts"]["dice_loss"].item())
            epoch_clip_losses.append(data["clip_loss"].detach().item() if torch.is_tensor(data["clip_loss"]) else float(data["clip_loss"]))
            epoch_weighted_clip_losses.append(data["weighted_clip_loss"].detach().item())
            epoch_loss_st.append(safe_float(loss_st_tensor))
            epoch_loss_rs.append(safe_float(loss_rs_tensor))
            epoch_loss_chunk_align.append(safe_float(loss_chunk_tensor))

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
        struct_cfg = getattr(cfg, "STRUCTXLIP", None)
        base_st = float(getattr(struct_cfg, "LAMBDA_STRUCTURE_TEXT", 0.0)) if struct_cfg is not None else 0.0
        base_rs = float(getattr(struct_cfg, "LAMBDA_RGB_STRUCTURE_CONSISTENCY", 0.0)) if struct_cfg is not None else 0.0
        base_chunk = float(getattr(struct_cfg, "LAMBDA_CHUNK_ALIGN", 0.0)) if struct_cfg is not None else 0.0
        controller.finish_epoch([
            mean_epoch_seg_loss,
            mean_epoch_weighted_clip_loss,
            base_st * mean_epoch_loss_st,
            base_rs * mean_epoch_loss_rs,
            base_chunk * mean_epoch_loss_chunk_align,
        ])
        struct_diag_means = {column: mean_or_zero(values) for column, values in epoch_struct_diagnostics.items()}

        mean_val_loss, mean_val_miou, val_loss_parts = evaluate_validation_loss(model, val_dataloader, cfg.MODEL.DEVICE, ce_loss, dice_loss, cfg)
        log_msg = (
            f"EPOCH: {epoch + 1} | Train Total: {mean_epoch_loss:.4f} | Train Seg: {mean_epoch_seg_loss:.4f} | "
            f"Train BCE: {mean_epoch_bce_loss:.4f} | Train DiceLoss: {mean_epoch_dice_loss:.4f} | "
            f"Train CLIP: {mean_epoch_clip_loss:.4f} | Train WeightedCLIP: {mean_epoch_weighted_clip_loss:.4f} | "
        )
        if cfg.MODEL.CLIP_MODEL == "structxlip":
            log_msg += (
                f"Train loss_st/rs/chunk: {mean_epoch_loss_st:.4f}/{mean_epoch_loss_rs:.4f}/{mean_epoch_loss_chunk_align:.4f} | "
                f"Weighted st/rs/chunk: {struct_diag_means.get('weighted_loss_st', 0.0):.4f}/"
                f"{struct_diag_means.get('weighted_loss_rs', 0.0):.4f}/"
                f"{struct_diag_means.get('weighted_loss_chunk_align', 0.0):.4f} | "
                f"Weights seg/clip/st/rs/chunk: {struct_diag_means.get('weight_seg', 0.0):.4f}/"
                f"{struct_diag_means.get('weight_clip', 0.0):.4f}/"
                f"{struct_diag_means.get('weight_st', 0.0):.4f}/"
                f"{struct_diag_means.get('weight_rs', 0.0):.4f}/"
                f"{struct_diag_means.get('weight_chunk', 0.0):.4f} | "
            )
        log_msg += (
            f"Val Total: {mean_val_loss:.4f} | Val BCE: {val_loss_parts['bce']:.4f} | "
            f"Val DiceLoss: {val_loss_parts['dice_loss']:.4f} | Val mIoU: {mean_val_miou:.4f} | "
            f"Val cIoU: {val_loss_parts['ciou']:.4f}"
        )
        logger.info(log_msg)

        if cfg.MODEL.CLIP_MODEL == "structxlip":
            diag_row = {column: 0.0 for column in AUTOLAMBDA_DIAGNOSTIC_COLUMNS}
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

        checkpoint_common = {
            "model": model.state_dict(),
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "struct_weight_controller": controller.state_dict(),
            "weight_optimizer": weight_optimizer.state_dict() if weight_optimizer is not None else None,
            "meta_optimizer": meta_optimizer.state_dict() if meta_optimizer is not None else None,
            "best_loss": min(best_loss, mean_val_loss),
            "best_miou": max(best_miou, mean_val_miou),
            "best_dice": max(best_miou, mean_val_miou),
        }
        if mean_val_miou > best_miou:
            logger.info(f"New best mIoU: {best_miou:.4f} -> {mean_val_miou:.4f}")
            best_miou = mean_val_miou
            torch.save(checkpoint_common, os.path.join(run_output_dir, f"{results_name}_best_dice.pth"))
        else:
            logger.info(f"mIoU: {mean_val_miou:.4f}")
        best_loss = min(best_loss, mean_val_loss)
        if getattr(cfg, "save_every_checkpoint", False):
            torch.save(checkpoint_common, os.path.join(run_output_dir, f"{results_name}_epoch{epoch + 1:03d}.pth"))
        torch.save(checkpoint_common, os.path.join(run_output_dir, f"{results_name}_latest.pth"))


if __name__ == "__main__":
    main()
