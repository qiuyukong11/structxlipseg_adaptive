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
import math
import numpy as np
from torch.nn.modules.loss import BCEWithLogitsLoss
import torch.nn.functional as F
import logging
import csv
from utils.main_utils import load_cfg_from_cfg_file
from losses import SegWithStructXLIPLoss

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

STRUCTXLIP_DIAGNOSTIC_COLUMNS = [
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
    "tau_reg_loss",
    "struct_objective_total",
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
    "s_st",
    "s_rs",
    "s_chunk",
    "z_st",
    "z_rs",
    "z_chunk",
    "alpha",
    "p_st",
    "p_rs",
    "p_chunk",
    "lambda_st",
    "lambda_rs",
    "lambda_chunk",
    "ema_st",
    "ema_rs",
    "ema_chunk",
    "w_st",
    "w_rs",
    "w_chunk",
    "a_st",
    "a_rs",
    "a_chunk",
    "reward_st_obs",
    "reward_rs_obs",
    "reward_chunk_obs",
    "reward_st",
    "reward_rs",
    "reward_chunk",
    "reward_tilde_st",
    "reward_tilde_rs",
    "reward_tilde_chunk",
    "norm_loss_st",
    "norm_loss_rs",
    "norm_loss_chunk",
    "aux_loss_mean",
    "gamma",
    "gamma_aux_over_seg",
]


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


def gradient_vector(loss, parameters):
    if not torch.is_tensor(loss) or not loss.requires_grad:
        return None
    params = list(parameters)
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    flat_grads = []
    has_any_grad = False
    for param, grad in zip(params, grads):
        if grad is None:
            flat_grads.append(torch.zeros_like(param, dtype=torch.float32).reshape(-1))
        else:
            has_any_grad = True
            flat_grads.append(grad.detach().reshape(-1).float())
    if not flat_grads or not has_any_grad:
        return None
    return torch.cat(flat_grads)


def safe_norm(vector):
    if vector is None or vector.numel() == 0:
        return 0.0
    norm = torch.linalg.vector_norm(vector)
    return safe_float(norm)


def safe_cosine(vector_a, vector_b, eps=1e-12):
    norm_a = safe_norm(vector_a)
    norm_b = safe_norm(vector_b)
    if norm_a <= eps or norm_b <= eps:
        return 0.0
    cosine = torch.dot(vector_a, vector_b) / (norm_a * norm_b)
    return safe_float(cosine.clamp(-1.0, 1.0))


def get_adaptive_v2_config(cfg):
    struct_cfg = getattr(cfg, "STRUCTXLIP", None)
    adaptive_cfg = getattr(struct_cfg, "ADAPTIVE_V2", None)
    return {
        "enabled": bool(getattr(adaptive_cfg, "ENABLED", False)),
        "eps": float(getattr(adaptive_cfg, "EPS", 1e-8)),
    }


def is_adaptive_v2_enabled(cfg):
    return (
        getattr(getattr(cfg, "MODEL", None), "CLIP_MODEL", "") == "structxlip"
        and get_adaptive_v2_config(cfg)["enabled"]
    )


def get_adaptive_v3_config(cfg):
    struct_cfg = getattr(cfg, "STRUCTXLIP", None)
    adaptive_cfg = getattr(struct_cfg, "ADAPTIVE_V3", None)
    return {
        "enabled": bool(getattr(adaptive_cfg, "ENABLED", False)),
        "eps": float(getattr(adaptive_cfg, "EPS", 1e-8)),
        "alpha_min": float(getattr(adaptive_cfg, "ALPHA_MIN", 0.05)),
        "alpha_max": float(getattr(adaptive_cfg, "ALPHA_MAX", 0.30)),
        "tau": float(getattr(adaptive_cfg, "TAU", 10.0)),
    }


def is_adaptive_v3_enabled(cfg):
    return (
        getattr(getattr(cfg, "MODEL", None), "CLIP_MODEL", "") == "structxlip"
        and get_adaptive_v3_config(cfg)["enabled"]
    )


def get_adaptive_v4_config(cfg):
    struct_cfg = getattr(cfg, "STRUCTXLIP", None)
    adaptive_cfg = getattr(struct_cfg, "ADAPTIVE_V4", None)
    return {
        "enabled": bool(getattr(adaptive_cfg, "ENABLED", False)),
        "eps": float(getattr(adaptive_cfg, "EPS", 1e-8)),
        "alpha_min": float(getattr(adaptive_cfg, "ALPHA_MIN", 0.08)),
        "alpha_max": float(getattr(adaptive_cfg, "ALPHA_MAX", 0.35)),
    }


def is_adaptive_v4_enabled(cfg):
    return (
        getattr(getattr(cfg, "MODEL", None), "CLIP_MODEL", "") == "structxlip"
        and get_adaptive_v4_config(cfg)["enabled"]
    )


def get_adaptive_v6_config(cfg):
    struct_cfg = getattr(cfg, "STRUCTXLIP", None)
    adaptive_cfg = getattr(struct_cfg, "ADAPTIVE_V6", None)
    return {
        "enabled": bool(getattr(adaptive_cfg, "ENABLED", False)),
        "gamma": float(getattr(adaptive_cfg, "GAMMA", 0.1)),
        "beta": float(getattr(adaptive_cfg, "BETA", 0.9)),
        "eta": float(getattr(adaptive_cfg, "ETA", 0.5)),
        "eps": float(getattr(adaptive_cfg, "EPS", 1e-8)),
        "val_grad_batches": int(getattr(adaptive_cfg, "VAL_GRAD_BATCHES", 1)),
    }


def is_adaptive_v6_enabled(cfg):
    return (
        getattr(getattr(cfg, "MODEL", None), "CLIP_MODEL", "") == "structxlip"
        and get_adaptive_v6_config(cfg)["enabled"]
    )


def get_adaptive_v7_config(cfg):
    struct_cfg = getattr(cfg, "STRUCTXLIP", None)
    adaptive_cfg = getattr(struct_cfg, "ADAPTIVE_V7", None)
    return {
        "enabled": bool(getattr(adaptive_cfg, "ENABLED", False)),
        "gamma": float(getattr(adaptive_cfg, "GAMMA", 0.1)),
        "beta": float(getattr(adaptive_cfg, "BETA", 0.9)),
        "eta": float(getattr(adaptive_cfg, "ETA", 0.05)),
        "reward_ema": float(getattr(adaptive_cfg, "REWARD_EMA", 0.8)),
        "logit_clip": float(getattr(adaptive_cfg, "LOGIT_CLIP", 1.0)),
        "lambda_min": float(getattr(adaptive_cfg, "LAMBDA_MIN", 0.15)),
        "eps": float(getattr(adaptive_cfg, "EPS", 1e-8)),
        "val_grad_batches": int(getattr(adaptive_cfg, "VAL_GRAD_BATCHES", 1)),
    }


def is_adaptive_v7_enabled(cfg):
    return (
        getattr(getattr(cfg, "MODEL", None), "CLIP_MODEL", "") == "structxlip"
        and get_adaptive_v7_config(cfg)["enabled"]
    )


def get_learnable_tau_loss_config(cfg):
    struct_cfg = getattr(cfg, "STRUCTXLIP", None)
    tau_cfg = getattr(struct_cfg, "LEARNABLE_TAU_LOSS", None)
    return {
        "enabled": bool(getattr(tau_cfg, "ENABLED", False)),
        "max_tau_scale": float(getattr(tau_cfg, "MAX_TAU_SCALE", 100.0)),
        "init_temperature": float(getattr(tau_cfg, "INIT_TEMPERATURE", 0.07)),
        "min_temperature": float(getattr(tau_cfg, "MIN_TEMPERATURE", 0.01)),
        "eps": float(getattr(tau_cfg, "EPS", 1e-6)),
        "ignore_index": int(getattr(tau_cfg, "IGNORE_INDEX", -100)),
        "lr_mult": float(getattr(tau_cfg, "LR_MULT", 5.0)),
        "overall_weight": float(getattr(tau_cfg, "OVERALL_WEIGHT", 1.0)),
        "tau_regularizer_weight": float(getattr(tau_cfg, "TAU_REG_WEIGHT", 0.0)),
    }


def is_learnable_tau_loss_enabled(cfg):
    return (
        getattr(getattr(cfg, "MODEL", None), "CLIP_MODEL", "") == "structxlip"
        and get_learnable_tau_loss_config(cfg)["enabled"]
    )


def make_cross_entropy_seg_inputs(seg_logits, masks):
    logits = _as_bchw(seg_logits).float()
    labels = _as_bchw(masks).squeeze(1)
    labels = (labels > 0.5).long()
    if logits.shape[1] == 1:
        logits = torch.cat([-logits, logits], dim=1)
    return logits, labels


def select_structxlip_gradient_parameters(model):
    # Change this keyword list to widen/narrow the adaptive comparison scope.
    preferred_keywords = ("pvl_adapters",)
    preferred = [
        param for name, param in model.named_parameters()
        if param.requires_grad and any(keyword in name for keyword in preferred_keywords)
    ]
    if preferred:
        return preferred
    return [param for param in model.parameters() if param.requires_grad]


def fixed_structxlip_lambdas(model):
    return {
        "lambda_st": safe_float(getattr(model, "lambda_scribble_text", 0.0)),
        "lambda_rs": safe_float(getattr(model, "lambda_rgb_scribble", 0.0)),
        "lambda_chunk": safe_float(getattr(model, "lambda_chunk", 0.0)),
    }


def zero_adaptive_v2_result():
    lambdas = {"lambda_st": 0.0, "lambda_rs": 0.0, "lambda_chunk": 0.0}
    diagnostics = {
        "grad_norm_main": 0.0,
        "grad_norm_st": 0.0,
        "grad_norm_rs": 0.0,
        "grad_norm_chunk": 0.0,
        "cos_main_st": 0.0,
        "cos_main_rs": 0.0,
        "cos_main_chunk": 0.0,
        "s_st": 0.0,
        "s_rs": 0.0,
        "s_chunk": 0.0,
        "alpha": 0.0,
        "p_st": 0.0,
        "p_rs": 0.0,
        "p_chunk": 0.0,
    }
    return lambdas, diagnostics


def compute_adaptive_v2_lambdas(main_loss, raw_losses, parameters, adaptive_cfg):
    params = list(parameters)
    loss_st = raw_losses.get("loss_st")
    loss_rs = raw_losses.get("loss_rs")
    loss_chunk = raw_losses.get("loss_chunk_align")
    if not params or not all(torch.is_tensor(loss_value) for loss_value in (loss_st, loss_rs, loss_chunk)):
        return zero_adaptive_v2_result()

    try:
        grad_main = gradient_vector(main_loss, params)
        grad_st = gradient_vector(loss_st, params)
        grad_rs = gradient_vector(loss_rs, params)
        grad_chunk = gradient_vector(loss_chunk, params)
    except RuntimeError:
        return zero_adaptive_v2_result()

    if any(vector is None or vector.numel() == 0 for vector in (grad_main, grad_st, grad_rs, grad_chunk)):
        return zero_adaptive_v2_result()

    eps = float(adaptive_cfg["eps"])
    norm_main = safe_norm(grad_main)
    norm_st = safe_norm(grad_st)
    norm_rs = safe_norm(grad_rs)
    norm_chunk = safe_norm(grad_chunk)
    s_st = safe_cosine(grad_main, grad_st, eps=eps)
    s_rs = safe_cosine(grad_main, grad_rs, eps=eps)
    s_chunk = safe_cosine(grad_main, grad_chunk, eps=eps)

    scores = np.array([s_st, s_rs, s_chunk], dtype=np.float64)
    mu = float(scores.mean())
    sigma = float(scores.std() + eps)
    if not np.isfinite(sigma) or sigma <= 0.0:
        return zero_adaptive_v2_result()

    z = (scores - mu) / sigma
    z = z - np.max(z)
    exp_z = np.exp(z)
    p = exp_z / (exp_z.sum() + eps)
    alpha = max(0.0, float(scores.max()))

    lambda_st = safe_float(alpha * p[0])
    lambda_rs = safe_float(alpha * p[1])
    lambda_chunk = safe_float(alpha * p[2])
    lambdas = {
        "lambda_st": lambda_st,
        "lambda_rs": lambda_rs,
        "lambda_chunk": lambda_chunk,
    }
    diagnostics = {
        "grad_norm_main": norm_main,
        "grad_norm_st": lambda_st * norm_st,
        "grad_norm_rs": lambda_rs * norm_rs,
        "grad_norm_chunk": lambda_chunk * norm_chunk,
        "cos_main_st": s_st,
        "cos_main_rs": s_rs,
        "cos_main_chunk": s_chunk,
        "s_st": s_st,
        "s_rs": s_rs,
        "s_chunk": s_chunk,
        "alpha": safe_float(alpha),
        "p_st": safe_float(p[0]),
        "p_rs": safe_float(p[1]),
        "p_chunk": safe_float(p[2]),
    }
    if not all(np.isfinite(value) for value in list(lambdas.values()) + list(diagnostics.values())):
        return zero_adaptive_v2_result()
    return lambdas, diagnostics


def adaptive_v3_alpha(scores, adaptive_cfg):
    alpha_min = float(adaptive_cfg["alpha_min"])
    alpha_max = float(adaptive_cfg["alpha_max"])
    if not np.isfinite(alpha_min):
        alpha_min = 0.05
    if not np.isfinite(alpha_max):
        alpha_max = 0.30
    if alpha_max < alpha_min:
        alpha_min, alpha_max = alpha_max, alpha_min
    tau = float(adaptive_cfg["tau"])
    if not np.isfinite(tau):
        tau = 10.0
    s_max = float(np.max(scores))
    sigmoid_arg = np.clip(tau * s_max, -60.0, 60.0)
    strength = 1.0 / (1.0 + np.exp(-sigmoid_arg))
    return safe_float(alpha_min + (alpha_max - alpha_min) * strength)


def adaptive_v3_from_scores(scores, adaptive_cfg):
    eps = float(adaptive_cfg["eps"])
    scores = np.array(scores, dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        scores = np.zeros(3, dtype=np.float64)
    alpha = adaptive_v3_alpha(scores, adaptive_cfg)
    mu = float(scores.mean())
    sigma = float(scores.std() + eps)
    if not np.isfinite(sigma) or sigma <= 0.0:
        z = np.zeros(3, dtype=np.float64)
    else:
        z = (scores - mu) / sigma
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    denom = exp_z.sum()
    if not np.isfinite(denom) or denom <= eps:
        p = np.full(3, 1.0 / 3.0, dtype=np.float64)
    else:
        p = exp_z / denom
    lambdas = {
        "lambda_st": safe_float(alpha * p[0]),
        "lambda_rs": safe_float(alpha * p[1]),
        "lambda_chunk": safe_float(alpha * p[2]),
    }
    diagnostics = {
        "s_st": safe_float(scores[0]),
        "s_rs": safe_float(scores[1]),
        "s_chunk": safe_float(scores[2]),
        "z_st": safe_float(z[0]),
        "z_rs": safe_float(z[1]),
        "z_chunk": safe_float(z[2]),
        "alpha": safe_float(alpha),
        "p_st": safe_float(p[0]),
        "p_rs": safe_float(p[1]),
        "p_chunk": safe_float(p[2]),
    }
    if not all(np.isfinite(value) for value in list(lambdas.values()) + list(diagnostics.values())):
        alpha = adaptive_v3_alpha(np.zeros(3, dtype=np.float64), adaptive_cfg)
        lambdas = {"lambda_st": alpha / 3.0, "lambda_rs": alpha / 3.0, "lambda_chunk": alpha / 3.0}
        diagnostics.update({"z_st": 0.0, "z_rs": 0.0, "z_chunk": 0.0, "p_st": 1.0 / 3.0, "p_rs": 1.0 / 3.0, "p_chunk": 1.0 / 3.0, "alpha": alpha})
    return lambdas, diagnostics


def fallback_adaptive_v3_result(adaptive_cfg):
    lambdas, diagnostics = adaptive_v3_from_scores([0.0, 0.0, 0.0], adaptive_cfg)
    diagnostics.update({
        "grad_norm_main": 0.0,
        "grad_norm_st": 0.0,
        "grad_norm_rs": 0.0,
        "grad_norm_chunk": 0.0,
        "cos_main_st": 0.0,
        "cos_main_rs": 0.0,
        "cos_main_chunk": 0.0,
    })
    return lambdas, diagnostics


def compute_adaptive_v3_lambdas(main_loss, raw_losses, parameters, adaptive_cfg):
    params = list(parameters)
    loss_st = raw_losses.get("loss_st")
    loss_rs = raw_losses.get("loss_rs")
    loss_chunk = raw_losses.get("loss_chunk_align")
    if not params or not all(torch.is_tensor(loss_value) for loss_value in (loss_st, loss_rs, loss_chunk)):
        return fallback_adaptive_v3_result(adaptive_cfg)

    try:
        grad_main = gradient_vector(main_loss, params)
        grad_st = gradient_vector(loss_st, params)
        grad_rs = gradient_vector(loss_rs, params)
        grad_chunk = gradient_vector(loss_chunk, params)
    except RuntimeError:
        return fallback_adaptive_v3_result(adaptive_cfg)

    if any(vector is None or vector.numel() == 0 for vector in (grad_main, grad_st, grad_rs, grad_chunk)):
        return fallback_adaptive_v3_result(adaptive_cfg)

    eps = float(adaptive_cfg["eps"])
    norm_main = safe_norm(grad_main)
    norm_st = safe_norm(grad_st)
    norm_rs = safe_norm(grad_rs)
    norm_chunk = safe_norm(grad_chunk)
    s_st = safe_cosine(grad_main, grad_st, eps=eps)
    s_rs = safe_cosine(grad_main, grad_rs, eps=eps)
    s_chunk = safe_cosine(grad_main, grad_chunk, eps=eps)
    lambdas, diagnostics = adaptive_v3_from_scores([s_st, s_rs, s_chunk], adaptive_cfg)
    diagnostics.update({
        "grad_norm_main": norm_main,
        "grad_norm_st": lambdas["lambda_st"] * norm_st,
        "grad_norm_rs": lambdas["lambda_rs"] * norm_rs,
        "grad_norm_chunk": lambdas["lambda_chunk"] * norm_chunk,
        "cos_main_st": s_st,
        "cos_main_rs": s_rs,
        "cos_main_chunk": s_chunk,
    })
    if not all(np.isfinite(value) for value in list(lambdas.values()) + list(diagnostics.values())):
        return fallback_adaptive_v3_result(adaptive_cfg)
    return lambdas, diagnostics


def adaptive_v4_alpha(epoch_idx, num_epochs, adaptive_cfg):
    alpha_min = float(adaptive_cfg["alpha_min"])
    alpha_max = float(adaptive_cfg["alpha_max"])
    if not np.isfinite(alpha_min):
        alpha_min = 0.08
    if not np.isfinite(alpha_max):
        alpha_max = 0.35
    if alpha_max < alpha_min:
        alpha_min, alpha_max = alpha_max, alpha_min
    progress = (float(epoch_idx) - 1.0) / max(float(num_epochs) - 1.0, 1.0)
    progress = min(max(progress, 0.0), 1.0)
    strength = (1.0 + math.cos(math.pi * progress)) / 2.0
    return safe_float(alpha_min + (alpha_max - alpha_min) * strength)


def adaptive_v4_from_scores(scores, adaptive_cfg, epoch_idx, num_epochs, force_uniform=False):
    eps = float(adaptive_cfg["eps"])
    scores = np.array(scores, dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        scores = np.zeros(3, dtype=np.float64)
    alpha = adaptive_v4_alpha(epoch_idx, num_epochs, adaptive_cfg)
    if force_uniform:
        z = np.zeros(3, dtype=np.float64)
        p = np.full(3, 1.0 / 3.0, dtype=np.float64)
    else:
        mu = float(scores.mean())
        sigma = float(scores.std() + eps)
        if not np.isfinite(sigma) or sigma <= eps:
            z = np.zeros(3, dtype=np.float64)
        else:
            z = (scores - mu) / sigma
        z_shifted = z - np.max(z)
        exp_z = np.exp(z_shifted)
        denom = exp_z.sum()
        if not np.isfinite(denom) or denom <= eps:
            p = np.full(3, 1.0 / 3.0, dtype=np.float64)
        else:
            p = exp_z / denom
    lambdas = {
        "lambda_st": safe_float(alpha * p[0]),
        "lambda_rs": safe_float(alpha * p[1]),
        "lambda_chunk": safe_float(alpha * p[2]),
    }
    diagnostics = {
        "s_st": safe_float(scores[0]),
        "s_rs": safe_float(scores[1]),
        "s_chunk": safe_float(scores[2]),
        "z_st": safe_float(z[0]),
        "z_rs": safe_float(z[1]),
        "z_chunk": safe_float(z[2]),
        "alpha": safe_float(alpha),
        "p_st": safe_float(p[0]),
        "p_rs": safe_float(p[1]),
        "p_chunk": safe_float(p[2]),
    }
    if not all(np.isfinite(value) for value in list(lambdas.values()) + list(diagnostics.values())):
        alpha = adaptive_v4_alpha(epoch_idx, num_epochs, adaptive_cfg)
        lambdas = {"lambda_st": alpha / 3.0, "lambda_rs": alpha / 3.0, "lambda_chunk": alpha / 3.0}
        diagnostics.update({"z_st": 0.0, "z_rs": 0.0, "z_chunk": 0.0, "p_st": 1.0 / 3.0, "p_rs": 1.0 / 3.0, "p_chunk": 1.0 / 3.0, "alpha": alpha})
    return lambdas, diagnostics


def fallback_adaptive_v4_result(adaptive_cfg, epoch_idx, num_epochs):
    lambdas, diagnostics = adaptive_v4_from_scores(
        [0.0, 0.0, 0.0], adaptive_cfg, epoch_idx, num_epochs, force_uniform=True
    )
    diagnostics.update({
        "grad_norm_main": 0.0,
        "grad_norm_st": 0.0,
        "grad_norm_rs": 0.0,
        "grad_norm_chunk": 0.0,
        "cos_main_st": 0.0,
        "cos_main_rs": 0.0,
        "cos_main_chunk": 0.0,
    })
    return lambdas, diagnostics


def compute_adaptive_v4_lambdas(main_loss, raw_losses, parameters, adaptive_cfg, epoch_idx, num_epochs):
    params = list(parameters)
    loss_st = raw_losses.get("loss_st")
    loss_rs = raw_losses.get("loss_rs")
    loss_chunk = raw_losses.get("loss_chunk_align")
    if not params or not all(torch.is_tensor(loss_value) for loss_value in (loss_st, loss_rs, loss_chunk)):
        return fallback_adaptive_v4_result(adaptive_cfg, epoch_idx, num_epochs)

    try:
        grad_main = gradient_vector(main_loss, params)
        grad_st = gradient_vector(loss_st, params)
        grad_rs = gradient_vector(loss_rs, params)
        grad_chunk = gradient_vector(loss_chunk, params)
    except RuntimeError:
        return fallback_adaptive_v4_result(adaptive_cfg, epoch_idx, num_epochs)

    if any(vector is None or vector.numel() == 0 for vector in (grad_main, grad_st, grad_rs, grad_chunk)):
        return fallback_adaptive_v4_result(adaptive_cfg, epoch_idx, num_epochs)

    eps = float(adaptive_cfg["eps"])
    norm_main = safe_norm(grad_main)
    norm_st = safe_norm(grad_st)
    norm_rs = safe_norm(grad_rs)
    norm_chunk = safe_norm(grad_chunk)
    s_st = safe_cosine(grad_main, grad_st, eps=eps)
    s_rs = safe_cosine(grad_main, grad_rs, eps=eps)
    s_chunk = safe_cosine(grad_main, grad_chunk, eps=eps)
    lambdas, diagnostics = adaptive_v4_from_scores(
        [s_st, s_rs, s_chunk], adaptive_cfg, epoch_idx, num_epochs
    )
    diagnostics.update({
        "grad_norm_main": norm_main,
        "grad_norm_st": lambdas["lambda_st"] * norm_st,
        "grad_norm_rs": lambdas["lambda_rs"] * norm_rs,
        "grad_norm_chunk": lambdas["lambda_chunk"] * norm_chunk,
        "cos_main_st": s_st,
        "cos_main_rs": s_rs,
        "cos_main_chunk": s_chunk,
    })
    if not all(np.isfinite(value) for value in list(lambdas.values()) + list(diagnostics.values())):
        return fallback_adaptive_v4_result(adaptive_cfg, epoch_idx, num_epochs)
    return lambdas, diagnostics


def normalize_v6_weights(weights):
    weights = np.array(weights, dtype=np.float64)
    if weights.shape != (3,) or not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        weights = np.ones(3, dtype=np.float64)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        weights = np.ones(3, dtype=np.float64)
        total = 3.0
    return weights / total


def v6_lambdas_from_weights(weights):
    weights = normalize_v6_weights(weights)
    return {
        "lambda_st": safe_float(weights[0]),
        "lambda_rs": safe_float(weights[1]),
        "lambda_chunk": safe_float(weights[2]),
    }


def v7_lambdas_from_logits(logits, adaptive_cfg):
    eps = float(adaptive_cfg["eps"])
    logits = np.array(logits, dtype=np.float64)
    if logits.shape != (3,) or not np.all(np.isfinite(logits)):
        logits = np.zeros(3, dtype=np.float64)
    logit_clip = abs(float(adaptive_cfg["logit_clip"]))
    if np.isfinite(logit_clip) and logit_clip > 0.0:
        logits = np.clip(logits, -logit_clip, logit_clip)
    shifted = logits - np.max(logits)
    exp_logits = np.exp(shifted)
    denom = float(exp_logits.sum())
    if not np.isfinite(denom) or denom <= eps:
        weights = np.full(3, 1.0 / 3.0, dtype=np.float64)
    else:
        weights = exp_logits / denom
    lambda_min = float(adaptive_cfg["lambda_min"])
    if not np.isfinite(lambda_min):
        lambda_min = 0.0
    lambda_min = min(max(lambda_min, 0.0), (1.0 / 3.0) - eps)
    if lambda_min > 0.0:
        weights = lambda_min + (1.0 - 3.0 * lambda_min) * weights
    total = float(weights.sum())
    if np.isfinite(total) and total > eps:
        weights = weights / total
    else:
        weights = np.full(3, 1.0 / 3.0, dtype=np.float64)
    return weights


def v7_lambdas_dict(logits, adaptive_cfg):
    weights = v7_lambdas_from_logits(logits, adaptive_cfg)
    return {
        "lambda_st": safe_float(weights[0]),
        "lambda_rs": safe_float(weights[1]),
        "lambda_chunk": safe_float(weights[2]),
    }


def update_v7_logits(logits, smoothed_rewards, adaptive_cfg):
    logits = np.array(logits, dtype=np.float64)
    rewards = np.array(smoothed_rewards, dtype=np.float64)
    if logits.shape != (3,) or not np.all(np.isfinite(logits)):
        logits = np.zeros(3, dtype=np.float64)
    if rewards.shape != (3,) or not np.all(np.isfinite(rewards)):
        rewards = np.zeros(3, dtype=np.float64)
    reward_tilde = rewards - float(rewards.mean())
    eta = float(adaptive_cfg["eta"])
    if not np.isfinite(eta):
        eta = 0.05
    logits = logits + eta * reward_tilde
    logit_clip = abs(float(adaptive_cfg["logit_clip"]))
    if np.isfinite(logit_clip) and logit_clip > 0.0:
        logits = np.clip(logits, -logit_clip, logit_clip)
    logits = np.where(np.isfinite(logits), logits, 0.0)
    return logits, reward_tilde


def accumulate_gradient_vector(accum, count, loss, parameters):
    try:
        vector = gradient_vector(loss, parameters)
    except RuntimeError:
        return accum, count
    if vector is None or vector.numel() == 0:
        return accum, count
    vector = vector.detach()
    accum = vector if accum is None else accum + vector
    return accum, count + 1


def average_gradient_vector(accum, count):
    if accum is None or count <= 0:
        return None
    return accum / float(count)


def compute_v6_validation_main_gradient(model, val_dataloader, device, ce_loss, dice_loss, cfg, parameters, max_batches):
    params = list(parameters)
    if not params or max_batches <= 0:
        return None
    was_training = model.training
    model.eval()
    accum = None
    count = 0
    try:
        for batch_idx, batch in enumerate(val_dataloader):
            if batch_idx >= max_batches:
                break
            images = batch["image"].to(device)
            masks = batch["ground_truth_mask"].to(device)
            logits = model(images, text=batch["text_prompt"], num_samples=1)[0]
            val_main_loss = calc_loss(logits, masks, ce_loss, dice_loss, cfg)
            accum, count = accumulate_gradient_vector(accum, count, val_main_loss, params)
    except RuntimeError:
        accum = None
        count = 0
    finally:
        if was_training:
            model.train()
    return average_gradient_vector(accum, count)


def update_v6_simplex_weights(weights, rewards, eta):
    weights = normalize_v6_weights(weights)
    updated = weights.copy()
    for idx, reward in enumerate(rewards):
        if reward is None or not np.isfinite(reward):
            continue
        updated[idx] = updated[idx] * np.exp(np.clip(float(eta) * float(reward), -20.0, 20.0))
    return normalize_v6_weights(updated)


def compute_gradient_diagnostics(main_loss, weighted_loss_st, weighted_loss_rs, weighted_loss_chunk, parameters):
    params = list(parameters)
    if not params:
        return {
            "grad_norm_main": 0.0,
            "grad_norm_st": 0.0,
            "grad_norm_rs": 0.0,
            "grad_norm_chunk": 0.0,
            "cos_main_st": 0.0,
            "cos_main_rs": 0.0,
            "cos_main_chunk": 0.0,
        }

    try:
        grad_main = gradient_vector(main_loss, params)
        grad_st = gradient_vector(weighted_loss_st, params)
        grad_rs = gradient_vector(weighted_loss_rs, params)
        grad_chunk = gradient_vector(weighted_loss_chunk, params)
    except RuntimeError:
        return {
            "grad_norm_main": 0.0,
            "grad_norm_st": 0.0,
            "grad_norm_rs": 0.0,
            "grad_norm_chunk": 0.0,
            "cos_main_st": 0.0,
            "cos_main_rs": 0.0,
            "cos_main_chunk": 0.0,
        }

    return {
        "grad_norm_main": safe_norm(grad_main),
        "grad_norm_st": safe_norm(grad_st),
        "grad_norm_rs": safe_norm(grad_rs),
        "grad_norm_chunk": safe_norm(grad_chunk),
        "cos_main_st": safe_cosine(grad_main, grad_st),
        "cos_main_rs": safe_cosine(grad_main, grad_rs),
        "cos_main_chunk": safe_cosine(grad_main, grad_chunk),
    }


def append_epoch_diagnostics_csv(path, row):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=STRUCTXLIP_DIAGNOSTIC_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, 0.0) for column in STRUCTXLIP_DIAGNOSTIC_COLUMNS})


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
    adaptive_v2_enabled = is_adaptive_v2_enabled(cfg)
    adaptive_v3_enabled = is_adaptive_v3_enabled(cfg)
    adaptive_v4_enabled = is_adaptive_v4_enabled(cfg)
    adaptive_v6_enabled = is_adaptive_v6_enabled(cfg)
    adaptive_v7_enabled = is_adaptive_v7_enabled(cfg)
    learnable_tau_loss_enabled = is_learnable_tau_loss_enabled(cfg)
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
            and (
                adaptive_v2_enabled
                or adaptive_v3_enabled
                or adaptive_v4_enabled
                or adaptive_v6_enabled
                or adaptive_v7_enabled
                or learnable_tau_loss_enabled
                or any(
                    float(getattr(struct_cfg, key, 0.0)) != 0.0
                    for key in (
                        "LAMBDA_STRUCTURE_TEXT",
                        "LAMBDA_RGB_STRUCTURE_CONSISTENCY",
                        "LAMBDA_CHUNK_ALIGN",
                    )
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
    if cfg.MODEL.CLIP_MODEL == "structxlip":
        cfg.DATASET.NAME = cfg.DATASET.NAME + f"_st_{getattr(cfg.STRUCTXLIP, 'LAMBDA_STRUCTURE_TEXT', 0.0)}_rs_{getattr(cfg.STRUCTXLIP, 'LAMBDA_RGB_STRUCTURE_CONSISTENCY', 0.0)}_chunk_{getattr(cfg.STRUCTXLIP, 'LAMBDA_CHUNK_ALIGN', 0.0)}"
        if is_learnable_tau_loss_enabled(cfg):
            tau_weight = get_learnable_tau_loss_config(cfg)["overall_weight"]
            cfg.DATASET.NAME = cfg.DATASET.NAME + f"_learnable_tau_w{tau_weight:g}"
    run_output_dir = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}")
    os.makedirs(run_output_dir, exist_ok=True)
    structxlip_diagnostics_path = os.path.join(run_output_dir, "structxlip_train_diagnostics.csv")

    logger = logger_config(os.path.join(run_output_dir, "log.txt"))
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
    learnable_tau_cfg = get_learnable_tau_loss_config(cfg)
    learnable_tau_loss_enabled = (cfg.MODEL.CLIP_MODEL == "structxlip" and learnable_tau_cfg["enabled"])
    learnable_tau_loss = None
    if learnable_tau_loss_enabled:
        learnable_tau_loss = SegWithStructXLIPLoss(
            max_tau_scale=learnable_tau_cfg["max_tau_scale"],
            init_temperature=learnable_tau_cfg["init_temperature"],
            min_temperature=learnable_tau_cfg["min_temperature"],
            ignore_index=learnable_tau_cfg["ignore_index"],
            eps=learnable_tau_cfg["eps"],
            aux_weight=learnable_tau_cfg["overall_weight"],
            tau_regularizer_weight=learnable_tau_cfg["tau_regularizer_weight"],
        ).to(cfg.MODEL.DEVICE)
        tau_lr = float(cfg.TRAIN.LEARNING_RATE) * learnable_tau_cfg["lr_mult"]
        optimizer = torch.optim.Adam([
            {"params": filter(lambda p: p.requires_grad, model.parameters()), "lr": cfg.TRAIN.LEARNING_RATE},
            {"params": learnable_tau_loss.parameters(), "lr": tau_lr, "weight_decay": 0.0},
        ], lr=cfg.TRAIN.LEARNING_RATE)
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.TRAIN.LEARNING_RATE)
    num_epochs = cfg.TRAIN.NUM_EPOCHS
    use_clip_loss = bool(getattr(cfg.TRAIN, "USE_CLIP_LOSS", True))
    clip_loss_weight = float(getattr(cfg.TRAIN, "CLIP_WEIGHT", 0.0))
    adaptive_v2_cfg = get_adaptive_v2_config(cfg)
    adaptive_v3_cfg = get_adaptive_v3_config(cfg)
    adaptive_v4_cfg = get_adaptive_v4_config(cfg)
    adaptive_v6_cfg = get_adaptive_v6_config(cfg)
    adaptive_v7_cfg = get_adaptive_v7_config(cfg)
    adaptive_v7_enabled = (
        cfg.MODEL.CLIP_MODEL == "structxlip"
        and adaptive_v7_cfg["enabled"]
        and not learnable_tau_loss_enabled
    )
    adaptive_v6_enabled = (
        cfg.MODEL.CLIP_MODEL == "structxlip"
        and adaptive_v6_cfg["enabled"]
        and not learnable_tau_loss_enabled
        and not adaptive_v7_enabled
    )
    adaptive_v4_enabled = (
        cfg.MODEL.CLIP_MODEL == "structxlip"
        and adaptive_v4_cfg["enabled"]
        and not learnable_tau_loss_enabled
        and not adaptive_v6_enabled
        and not adaptive_v7_enabled
    )
    adaptive_v3_enabled = (
        cfg.MODEL.CLIP_MODEL == "structxlip"
        and adaptive_v3_cfg["enabled"]
        and not learnable_tau_loss_enabled
        and not adaptive_v4_enabled
        and not adaptive_v6_enabled
        and not adaptive_v7_enabled
    )
    adaptive_v2_enabled = (
        cfg.MODEL.CLIP_MODEL == "structxlip"
        and adaptive_v2_cfg["enabled"]
        and not learnable_tau_loss_enabled
        and not adaptive_v3_enabled
        and not adaptive_v4_enabled
        and not adaptive_v6_enabled
        and not adaptive_v7_enabled
    )
    logger.info(f"CLIP auxiliary loss enabled: {use_clip_loss}, weight: {clip_loss_weight}")
    if cfg.MODEL.CLIP_MODEL == "structxlip":
        logger.info(
            "StructXLIP learnable tau loss enabled: "
            f"{learnable_tau_loss_enabled}, "
            f"max_tau_scale: {learnable_tau_cfg['max_tau_scale']}, "
            f"init_temperature: {learnable_tau_cfg['init_temperature']}, "
            f"min_temperature: {learnable_tau_cfg['min_temperature']}, "
            f"lr_mult: {learnable_tau_cfg['lr_mult']}, "
            f"overall_weight: {learnable_tau_cfg['overall_weight']}, "
            f"tau_reg_weight: {learnable_tau_cfg['tau_regularizer_weight']}"
        )
        logger.info(
            "StructXLIP adaptive v2 enabled: "
            f"{adaptive_v2_enabled}, "
            f"eps: {adaptive_v2_cfg['eps']}"
        )
        logger.info(
            "StructXLIP adaptive v3 enabled: "
            f"{adaptive_v3_enabled}, "
            f"eps: {adaptive_v3_cfg['eps']}, "
            f"alpha_min: {adaptive_v3_cfg['alpha_min']}, "
            f"alpha_max: {adaptive_v3_cfg['alpha_max']}, "
            f"tau: {adaptive_v3_cfg['tau']}"
        )
        logger.info(
            "StructXLIP adaptive v4 enabled: "
            f"{adaptive_v4_enabled}, "
            f"eps: {adaptive_v4_cfg['eps']}, "
            f"alpha_min: {adaptive_v4_cfg['alpha_min']}, "
            f"alpha_max: {adaptive_v4_cfg['alpha_max']}"
        )
        logger.info(
            "StructXLIP adaptive v6 enabled: "
            f"{adaptive_v6_enabled}, "
            f"gamma: {adaptive_v6_cfg['gamma']}, "
            f"beta: {adaptive_v6_cfg['beta']}, "
            f"eta: {adaptive_v6_cfg['eta']}, "
            f"eps: {adaptive_v6_cfg['eps']}, "
            f"val_grad_batches: {adaptive_v6_cfg['val_grad_batches']}"
        )
        logger.info(
            "StructXLIP adaptive v7 enabled: "
            f"{adaptive_v7_enabled}, "
            f"gamma: {adaptive_v7_cfg['gamma']}, "
            f"beta: {adaptive_v7_cfg['beta']}, "
            f"eta: {adaptive_v7_cfg['eta']}, "
            f"reward_ema: {adaptive_v7_cfg['reward_ema']}, "
            f"logit_clip: {adaptive_v7_cfg['logit_clip']}, "
            f"lambda_min: {adaptive_v7_cfg['lambda_min']}, "
            f"eps: {adaptive_v7_cfg['eps']}, "
            f"val_grad_batches: {adaptive_v7_cfg['val_grad_batches']}"
        )

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
        if learnable_tau_loss is not None and "learnable_tau_loss" in checkpoint and checkpoint["learnable_tau_loss"] is not None:
            learnable_tau_loss.load_state_dict(checkpoint["learnable_tau_loss"])
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
    diagnostic_parameters = select_structxlip_gradient_parameters(model)
    v6_ema = np.zeros(3, dtype=np.float64)
    v6_weights = np.ones(3, dtype=np.float64)
    v7_ema = np.zeros(3, dtype=np.float64)
    v7_logits = np.zeros(3, dtype=np.float64)
    v7_rewards = np.zeros(3, dtype=np.float64)
    v7_reward_tilde = np.zeros(3, dtype=np.float64)
    if cfg.MODEL.CLIP_MODEL == "structxlip" and start_epoch == 0:
        with open(structxlip_diagnostics_path, "w", newline="") as csv_file:
            csv.DictWriter(csv_file, fieldnames=STRUCTXLIP_DIAGNOSTIC_COLUMNS).writeheader()
        logger.info(f"StructXLIP diagnostics CSV: {structxlip_diagnostics_path}")

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
        epoch_struct_diagnostics = {
            column: [] for column in STRUCTXLIP_DIAGNOSTIC_COLUMNS if column != "epoch"
        }
        v6_grad_accum = {"st": None, "rs": None, "chunk": None}
        v6_grad_counts = {"st": 0, "rs": 0, "chunk": 0}

        for i, batch in enumerate(tqdm(train_dataloader)):

            model_kwargs = {
                "image": batch["image"].to(cfg.MODEL.DEVICE),
                "text": batch["text_prompt"],
                "return_clip_loss": learnable_tau_loss_enabled or (use_clip_loss and clip_loss_weight != 0) or adaptive_v2_enabled or adaptive_v3_enabled or adaptive_v4_enabled or adaptive_v6_enabled or adaptive_v7_enabled,
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
            weighted_clip_loss = seg_loss.new_zeros(())
            if use_clip_loss and clip_loss_weight != 0:
                weighted_clip_loss = clip_loss_weight * clip_loss
            main_loss = seg_loss + weighted_clip_loss
            loss = main_loss
            active_struct_lambdas = None
            active_gradient_diagnostics = None
            active_v6_norm_losses = None
            active_v6_aux_loss = None
            active_learnable_tau_parts = None
            if structxlip_loss is not None:
                if learnable_tau_loss_enabled and learnable_tau_loss is not None:
                    raw_struct_losses = getattr(model, "last_structxlip_loss_tensors", {})
                    zero = seg_loss.new_zeros(())
                    loss_st_raw = raw_struct_losses.get("loss_st", zero)
                    loss_rs_raw = raw_struct_losses.get("loss_rs", zero)
                    loss_chunk_raw = raw_struct_losses.get("loss_chunk_align", zero)
                    if all(torch.is_tensor(loss_value) for loss_value in (loss_st_raw, loss_rs_raw, loss_chunk_raw)):
                        structxlip_loss, active_learnable_tau_parts = learnable_tau_loss.forward_raw_losses(
                            loss_st_raw,
                            loss_rs_raw,
                            loss_chunk_raw,
                            getattr(model, "lambda_scribble_text", 0.0),
                            getattr(model, "lambda_rgb_scribble", 0.0),
                            getattr(model, "lambda_chunk", 0.0),
                        )
                        active_struct_lambdas = {
                            "lambda_st": safe_float(active_learnable_tau_parts["lambda_st"]),
                            "lambda_rs": safe_float(active_learnable_tau_parts["lambda_rs"]),
                            "lambda_chunk": safe_float(active_learnable_tau_parts["lambda_chunk"]),
                        }
                        model.last_structxlip_lambdas = active_struct_lambdas
                        active_gradient_diagnostics = {
                            "weighted_loss_st": safe_float(active_learnable_tau_parts["weighted_loss_st"]),
                            "weighted_loss_rs": safe_float(active_learnable_tau_parts["weighted_loss_rs"]),
                            "weighted_loss_chunk_align": safe_float(active_learnable_tau_parts["weighted_loss_chunk_align"]),
                            "weighted_struct_total": safe_float(active_learnable_tau_parts["weighted_struct_total"]),
                            "tau_reg_loss": safe_float(active_learnable_tau_parts["tau_reg_loss"]),
                            "struct_objective_total": safe_float(active_learnable_tau_parts["struct_objective_total"]),
                            "struct_over_seg": safe_div(active_learnable_tau_parts["weighted_struct_total"], seg_loss),
                            "weighted_st_over_seg": safe_div(active_learnable_tau_parts["weighted_loss_st"], seg_loss),
                            "weighted_rs_over_seg": safe_div(active_learnable_tau_parts["weighted_loss_rs"], seg_loss),
                            "weighted_chunk_over_seg": safe_div(active_learnable_tau_parts["weighted_loss_chunk_align"], seg_loss),
                            "gamma": learnable_tau_cfg["overall_weight"],
                            "gamma_aux_over_seg": safe_div(active_learnable_tau_parts["struct_objective_total"], seg_loss),
                        }
                    else:
                        structxlip_loss = seg_loss.new_zeros(())
                elif adaptive_v7_enabled or adaptive_v6_enabled or adaptive_v4_enabled or adaptive_v3_enabled or adaptive_v2_enabled:
                    raw_struct_losses = getattr(model, "last_structxlip_loss_tensors", {})
                    zero = seg_loss.new_zeros(())
                    loss_st_raw = raw_struct_losses.get("loss_st", zero)
                    loss_rs_raw = raw_struct_losses.get("loss_rs", zero)
                    loss_chunk_raw = raw_struct_losses.get("loss_chunk_align", zero)
                    if adaptive_v7_enabled:
                        eps = float(adaptive_v7_cfg["eps"])
                        beta = float(adaptive_v7_cfg["beta"])
                        gamma = float(adaptive_v7_cfg["gamma"])
                        raw_values = np.array([
                            safe_float(loss_st_raw),
                            safe_float(loss_rs_raw),
                            safe_float(loss_chunk_raw),
                        ], dtype=np.float64)
                        raw_values = np.where(np.isfinite(raw_values), raw_values, 0.0)
                        ema_values = np.maximum(raw_values, 0.0)
                        v7_ema = beta * v7_ema + (1.0 - beta) * ema_values
                        scales = np.maximum(v7_ema, eps)
                        loss_st_for_norm = torch.clamp(loss_st_raw, min=0.0) if torch.is_tensor(loss_st_raw) else zero
                        loss_rs_for_norm = torch.clamp(loss_rs_raw, min=0.0) if torch.is_tensor(loss_rs_raw) else zero
                        loss_chunk_for_norm = torch.clamp(loss_chunk_raw, min=0.0) if torch.is_tensor(loss_chunk_raw) else zero
                        norm_loss_st = loss_st_for_norm / scales[0]
                        norm_loss_rs = loss_rs_for_norm / scales[1]
                        norm_loss_chunk = loss_chunk_for_norm / scales[2]
                        active_v6_norm_losses = {
                            "norm_loss_st": norm_loss_st,
                            "norm_loss_rs": norm_loss_rs,
                            "norm_loss_chunk": norm_loss_chunk,
                        }
                        simplex = v7_lambdas_from_logits(v7_logits, adaptive_v7_cfg)
                        active_struct_lambdas = {
                            "lambda_st": safe_float(simplex[0]),
                            "lambda_rs": safe_float(simplex[1]),
                            "lambda_chunk": safe_float(simplex[2]),
                        }
                        active_v6_aux_loss = (
                            simplex[0] * norm_loss_st
                            + simplex[1] * norm_loss_rs
                            + simplex[2] * norm_loss_chunk
                        )
                        structxlip_loss = gamma * active_v6_aux_loss
                        v6_grad_accum["st"], v6_grad_counts["st"] = accumulate_gradient_vector(
                            v6_grad_accum["st"], v6_grad_counts["st"], norm_loss_st, diagnostic_parameters
                        )
                        v6_grad_accum["rs"], v6_grad_counts["rs"] = accumulate_gradient_vector(
                            v6_grad_accum["rs"], v6_grad_counts["rs"], norm_loss_rs, diagnostic_parameters
                        )
                        v6_grad_accum["chunk"], v6_grad_counts["chunk"] = accumulate_gradient_vector(
                            v6_grad_accum["chunk"], v6_grad_counts["chunk"], norm_loss_chunk, diagnostic_parameters
                        )
                        active_gradient_diagnostics = {
                            "ema_st": safe_float(v7_ema[0]),
                            "ema_rs": safe_float(v7_ema[1]),
                            "ema_chunk": safe_float(v7_ema[2]),
                            "w_st": safe_float(simplex[0]),
                            "w_rs": safe_float(simplex[1]),
                            "w_chunk": safe_float(simplex[2]),
                            "a_st": safe_float(v7_logits[0]),
                            "a_rs": safe_float(v7_logits[1]),
                            "a_chunk": safe_float(v7_logits[2]),
                            "reward_st": safe_float(v7_rewards[0]),
                            "reward_rs": safe_float(v7_rewards[1]),
                            "reward_chunk": safe_float(v7_rewards[2]),
                            "reward_tilde_st": safe_float(v7_reward_tilde[0]),
                            "reward_tilde_rs": safe_float(v7_reward_tilde[1]),
                            "reward_tilde_chunk": safe_float(v7_reward_tilde[2]),
                            "norm_loss_st": safe_float(norm_loss_st),
                            "norm_loss_rs": safe_float(norm_loss_rs),
                            "norm_loss_chunk": safe_float(norm_loss_chunk),
                            "aux_loss_mean": safe_float(active_v6_aux_loss),
                            "gamma": gamma,
                            "gamma_aux_over_seg": safe_div(structxlip_loss, seg_loss),
                        }
                    elif adaptive_v6_enabled:
                        eps = float(adaptive_v6_cfg["eps"])
                        beta = float(adaptive_v6_cfg["beta"])
                        gamma = float(adaptive_v6_cfg["gamma"])
                        raw_values = np.array([
                            safe_float(loss_st_raw),
                            safe_float(loss_rs_raw),
                            safe_float(loss_chunk_raw),
                        ], dtype=np.float64)
                        raw_values = np.where(np.isfinite(raw_values), raw_values, 0.0)
                        v6_ema = beta * v6_ema + (1.0 - beta) * raw_values
                        scales = np.maximum(v6_ema, eps)
                        norm_loss_st = loss_st_raw / scales[0] if torch.is_tensor(loss_st_raw) else zero
                        norm_loss_rs = loss_rs_raw / scales[1] if torch.is_tensor(loss_rs_raw) else zero
                        norm_loss_chunk = loss_chunk_raw / scales[2] if torch.is_tensor(loss_chunk_raw) else zero
                        active_v6_norm_losses = {
                            "norm_loss_st": norm_loss_st,
                            "norm_loss_rs": norm_loss_rs,
                            "norm_loss_chunk": norm_loss_chunk,
                        }
                        simplex = normalize_v6_weights(v6_weights)
                        active_struct_lambdas = {
                            "lambda_st": safe_float(simplex[0]),
                            "lambda_rs": safe_float(simplex[1]),
                            "lambda_chunk": safe_float(simplex[2]),
                        }
                        active_v6_aux_loss = (
                            simplex[0] * norm_loss_st
                            + simplex[1] * norm_loss_rs
                            + simplex[2] * norm_loss_chunk
                        )
                        structxlip_loss = gamma * active_v6_aux_loss
                        v6_grad_accum["st"], v6_grad_counts["st"] = accumulate_gradient_vector(
                            v6_grad_accum["st"], v6_grad_counts["st"], norm_loss_st, diagnostic_parameters
                        )
                        v6_grad_accum["rs"], v6_grad_counts["rs"] = accumulate_gradient_vector(
                            v6_grad_accum["rs"], v6_grad_counts["rs"], norm_loss_rs, diagnostic_parameters
                        )
                        v6_grad_accum["chunk"], v6_grad_counts["chunk"] = accumulate_gradient_vector(
                            v6_grad_accum["chunk"], v6_grad_counts["chunk"], norm_loss_chunk, diagnostic_parameters
                        )
                        active_gradient_diagnostics = {
                            "ema_st": safe_float(v6_ema[0]),
                            "ema_rs": safe_float(v6_ema[1]),
                            "ema_chunk": safe_float(v6_ema[2]),
                            "w_st": safe_float(simplex[0]),
                            "w_rs": safe_float(simplex[1]),
                            "w_chunk": safe_float(simplex[2]),
                            "norm_loss_st": safe_float(norm_loss_st),
                            "norm_loss_rs": safe_float(norm_loss_rs),
                            "norm_loss_chunk": safe_float(norm_loss_chunk),
                            "aux_loss_mean": safe_float(active_v6_aux_loss),
                            "gamma": gamma,
                            "gamma_aux_over_seg": safe_div(structxlip_loss, seg_loss),
                        }
                    elif adaptive_v4_enabled:
                        active_struct_lambdas, active_gradient_diagnostics = compute_adaptive_v4_lambdas(
                            main_loss,
                            raw_struct_losses,
                            diagnostic_parameters,
                            adaptive_v4_cfg,
                            epoch + 1,
                            num_epochs,
                        )
                    elif adaptive_v3_enabled:
                        active_struct_lambdas, active_gradient_diagnostics = compute_adaptive_v3_lambdas(
                            main_loss,
                            raw_struct_losses,
                            diagnostic_parameters,
                            adaptive_v3_cfg,
                        )
                    else:
                        active_struct_lambdas, active_gradient_diagnostics = compute_adaptive_v2_lambdas(
                            main_loss,
                            raw_struct_losses,
                            diagnostic_parameters,
                            adaptive_v2_cfg,
                        )
                    if not adaptive_v7_enabled and not adaptive_v6_enabled:
                        structxlip_loss = (
                            active_struct_lambdas["lambda_st"] * loss_st_raw
                            + active_struct_lambdas["lambda_rs"] * loss_rs_raw
                            + active_struct_lambdas["lambda_chunk"] * loss_chunk_raw
                        )
                    model.last_structxlip_lambdas = active_struct_lambdas
                loss = loss + structxlip_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite total loss at epoch={epoch + 1}, iter={i}, batch={describe_batch(batch)}")

            if cfg.MODEL.CLIP_MODEL == "structxlip":
                zero = seg_loss.new_zeros(())
                struct_loss_tensors = getattr(model, "last_structxlip_loss_tensors", {})
                struct_lambdas = active_struct_lambdas or getattr(model, "last_structxlip_lambdas", {})

                lambda_st = safe_float(struct_lambdas.get("lambda_st", getattr(model, "lambda_scribble_text", 0.0)))
                lambda_rs = safe_float(struct_lambdas.get("lambda_rs", getattr(model, "lambda_rgb_scribble", 0.0)))
                lambda_chunk = safe_float(struct_lambdas.get("lambda_chunk", getattr(model, "lambda_chunk", 0.0)))

                loss_st_tensor = struct_loss_tensors.get("loss_st", zero)
                loss_rs_tensor = struct_loss_tensors.get("loss_rs", zero)
                loss_chunk_tensor = struct_loss_tensors.get("loss_chunk_align", zero)
                if (adaptive_v7_enabled or adaptive_v6_enabled) and active_v6_norm_losses is not None:
                    gamma = float(adaptive_v7_cfg["gamma"] if adaptive_v7_enabled else adaptive_v6_cfg["gamma"])
                    weighted_loss_st = gamma * lambda_st * active_v6_norm_losses["norm_loss_st"]
                    weighted_loss_rs = gamma * lambda_rs * active_v6_norm_losses["norm_loss_rs"]
                    weighted_loss_chunk = gamma * lambda_chunk * active_v6_norm_losses["norm_loss_chunk"]
                elif learnable_tau_loss_enabled and active_learnable_tau_parts is not None:
                    weighted_loss_st = active_learnable_tau_parts["weighted_loss_st"]
                    weighted_loss_rs = active_learnable_tau_parts["weighted_loss_rs"]
                    weighted_loss_chunk = active_learnable_tau_parts["weighted_loss_chunk_align"]
                else:
                    weighted_loss_st = lambda_st * loss_st_tensor if torch.is_tensor(loss_st_tensor) else zero
                    weighted_loss_rs = lambda_rs * loss_rs_tensor if torch.is_tensor(loss_rs_tensor) else zero
                    weighted_loss_chunk = lambda_chunk * loss_chunk_tensor if torch.is_tensor(loss_chunk_tensor) else zero
                weighted_struct_total = weighted_loss_st + weighted_loss_rs + weighted_loss_chunk

                struct_diag_values = {
                    "train_loss": safe_float(loss),
                    "seg_loss": safe_float(seg_loss),
                    "clip_loss": safe_float(clip_loss),
                    "weighted_clip_loss": safe_float(weighted_clip_loss),
                    "loss_st": safe_float(loss_st_tensor),
                    "loss_rs": safe_float(loss_rs_tensor),
                    "loss_chunk_align": safe_float(loss_chunk_tensor),
                    "weighted_loss_st": safe_float(weighted_loss_st),
                    "weighted_loss_rs": safe_float(weighted_loss_rs),
                    "weighted_loss_chunk_align": safe_float(weighted_loss_chunk),
                    "weighted_struct_total": safe_float(weighted_struct_total),
                    "tau_reg_loss": 0.0,
                    "struct_objective_total": safe_float(weighted_struct_total),
                    "struct_over_seg": safe_div(weighted_struct_total, seg_loss),
                    "weighted_st_over_seg": safe_div(weighted_loss_st, seg_loss),
                    "weighted_rs_over_seg": safe_div(weighted_loss_rs, seg_loss),
                    "weighted_chunk_over_seg": safe_div(weighted_loss_chunk, seg_loss),
                    "s_st": 0.0,
                    "s_rs": 0.0,
                    "s_chunk": 0.0,
                    "z_st": 0.0,
                    "z_rs": 0.0,
                    "z_chunk": 0.0,
                    "alpha": 0.0,
                    "p_st": 0.0,
                    "p_rs": 0.0,
                    "p_chunk": 0.0,
                    "lambda_st": lambda_st,
                    "lambda_rs": lambda_rs,
                    "lambda_chunk": lambda_chunk,
                    "ema_st": 0.0,
                    "ema_rs": 0.0,
                    "ema_chunk": 0.0,
                    "w_st": 0.0,
                    "w_rs": 0.0,
                    "w_chunk": 0.0,
                    "a_st": 0.0,
                    "a_rs": 0.0,
                    "a_chunk": 0.0,
                    "reward_st_obs": 0.0,
                    "reward_rs_obs": 0.0,
                    "reward_chunk_obs": 0.0,
                    "reward_st": 0.0,
                    "reward_rs": 0.0,
                    "reward_chunk": 0.0,
                    "reward_tilde_st": 0.0,
                    "reward_tilde_rs": 0.0,
                    "reward_tilde_chunk": 0.0,
                    "norm_loss_st": 0.0,
                    "norm_loss_rs": 0.0,
                    "norm_loss_chunk": 0.0,
                    "aux_loss_mean": 0.0,
                    "gamma": 0.0,
                    "gamma_aux_over_seg": 0.0,
                }
                if active_gradient_diagnostics is not None:
                    struct_diag_values.update(active_gradient_diagnostics)
                else:
                    struct_diag_values.update(compute_gradient_diagnostics(
                        main_loss,
                        weighted_loss_st,
                        weighted_loss_rs,
                        weighted_loss_chunk,
                        diagnostic_parameters,
                    ))
                for key, value in struct_diag_values.items():
                    if key in epoch_struct_diagnostics:
                        epoch_struct_diagnostics[key].append(value)

            optimizer.zero_grad()
            loss.backward()
            grad_clip_norm = float(getattr(cfg.TRAIN, "GRAD_CLIP_NORM", 1.0))
            if grad_clip_norm > 0:
                clip_params = list(filter(lambda p: p.requires_grad, model.parameters()))
                if learnable_tau_loss is not None:
                    clip_params += list(learnable_tau_loss.parameters())
                torch.nn.utils.clip_grad_norm_(
                    clip_params,
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
        struct_diag_means = {
            column: (mean(values) if values else 0.0)
            for column, values in epoch_struct_diagnostics.items()
        }
        # Validation phase
        mean_val_loss, mean_val_dice, val_loss_parts = evaluate_validation_loss(model, val_dataloader, cfg.MODEL.DEVICE, ce_loss, dice_loss, cfg)
        if adaptive_v7_enabled:
            val_grad_main = compute_v6_validation_main_gradient(
                model,
                val_dataloader,
                cfg.MODEL.DEVICE,
                ce_loss,
                dice_loss,
                cfg,
                diagnostic_parameters,
                int(adaptive_v7_cfg["val_grad_batches"]),
            )
            branch_grads = [
                average_gradient_vector(v6_grad_accum["st"], v6_grad_counts["st"]),
                average_gradient_vector(v6_grad_accum["rs"], v6_grad_counts["rs"]),
                average_gradient_vector(v6_grad_accum["chunk"], v6_grad_counts["chunk"]),
            ]
            rewards_obs = [None, None, None]
            if val_grad_main is not None and val_grad_main.numel() > 0:
                reward_ema = float(adaptive_v7_cfg["reward_ema"])
                if not np.isfinite(reward_ema):
                    reward_ema = 0.8
                reward_ema = min(max(reward_ema, 0.0), 1.0)
                for reward_idx, branch_grad in enumerate(branch_grads):
                    if branch_grad is None or branch_grad.numel() == 0:
                        continue
                    reward_obs = safe_cosine(val_grad_main, branch_grad, eps=float(adaptive_v7_cfg["eps"]))
                    if not np.isfinite(reward_obs):
                        continue
                    rewards_obs[reward_idx] = reward_obs
                    v7_rewards[reward_idx] = reward_ema * v7_rewards[reward_idx] + (1.0 - reward_ema) * reward_obs
                v7_logits, v7_reward_tilde = update_v7_logits(v7_logits, v7_rewards, adaptive_v7_cfg)
            next_simplex = v7_lambdas_from_logits(v7_logits, adaptive_v7_cfg)
            next_lambdas = v7_lambdas_dict(v7_logits, adaptive_v7_cfg)
            struct_diag_means.update({
                "reward_st_obs": safe_float(rewards_obs[0]),
                "reward_rs_obs": safe_float(rewards_obs[1]),
                "reward_chunk_obs": safe_float(rewards_obs[2]),
                "reward_st": safe_float(v7_rewards[0]),
                "reward_rs": safe_float(v7_rewards[1]),
                "reward_chunk": safe_float(v7_rewards[2]),
                "reward_tilde_st": safe_float(v7_reward_tilde[0]),
                "reward_tilde_rs": safe_float(v7_reward_tilde[1]),
                "reward_tilde_chunk": safe_float(v7_reward_tilde[2]),
                "a_st": safe_float(v7_logits[0]),
                "a_rs": safe_float(v7_logits[1]),
                "a_chunk": safe_float(v7_logits[2]),
                "w_st": safe_float(next_simplex[0]),
                "w_rs": safe_float(next_simplex[1]),
                "w_chunk": safe_float(next_simplex[2]),
                "lambda_st": next_lambdas["lambda_st"],
                "lambda_rs": next_lambdas["lambda_rs"],
                "lambda_chunk": next_lambdas["lambda_chunk"],
                "ema_st": safe_float(v7_ema[0]),
                "ema_rs": safe_float(v7_ema[1]),
                "ema_chunk": safe_float(v7_ema[2]),
            })
        elif adaptive_v6_enabled:
            val_grad_main = compute_v6_validation_main_gradient(
                model,
                val_dataloader,
                cfg.MODEL.DEVICE,
                ce_loss,
                dice_loss,
                cfg,
                diagnostic_parameters,
                int(adaptive_v6_cfg["val_grad_batches"]),
            )
            branch_grads = [
                average_gradient_vector(v6_grad_accum["st"], v6_grad_counts["st"]),
                average_gradient_vector(v6_grad_accum["rs"], v6_grad_counts["rs"]),
                average_gradient_vector(v6_grad_accum["chunk"], v6_grad_counts["chunk"]),
            ]
            rewards = [None, None, None]
            if val_grad_main is not None and val_grad_main.numel() > 0:
                rewards = [
                    safe_cosine(val_grad_main, branch_grad, eps=float(adaptive_v6_cfg["eps"]))
                    if branch_grad is not None and branch_grad.numel() > 0 else None
                    for branch_grad in branch_grads
                ]
                v6_weights = update_v6_simplex_weights(v6_weights, rewards, float(adaptive_v6_cfg["eta"]))
            else:
                v6_weights = normalize_v6_weights(v6_weights)
            next_lambdas = v6_lambdas_from_weights(v6_weights)
            struct_diag_means.update({
                "reward_st": safe_float(rewards[0]),
                "reward_rs": safe_float(rewards[1]),
                "reward_chunk": safe_float(rewards[2]),
                "w_st": safe_float(v6_weights[0]),
                "w_rs": safe_float(v6_weights[1]),
                "w_chunk": safe_float(v6_weights[2]),
                "lambda_st": next_lambdas["lambda_st"],
                "lambda_rs": next_lambdas["lambda_rs"],
                "lambda_chunk": next_lambdas["lambda_chunk"],
                "ema_st": safe_float(v6_ema[0]),
                "ema_rs": safe_float(v6_ema[1]),
                "ema_chunk": safe_float(v6_ema[2]),
            })
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
                f"Train weighted_st: {struct_diag_means.get('weighted_loss_st', 0.0):.4f} | "
                f"Train weighted_rs: {struct_diag_means.get('weighted_loss_rs', 0.0):.4f} | "
                f"Train weighted_chunk: {struct_diag_means.get('weighted_loss_chunk_align', 0.0):.4f} | "
                f"Train weighted_struct_total: {struct_diag_means.get('weighted_struct_total', 0.0):.4f} | "
                f"TauReg: {struct_diag_means.get('tau_reg_loss', 0.0):.4f} | "
                f"StructObj: {struct_diag_means.get('struct_objective_total', 0.0):.4f} | "
                f"Struct/Seg: {struct_diag_means.get('struct_over_seg', 0.0):.4f} | "
                f"ST/Seg: {struct_diag_means.get('weighted_st_over_seg', 0.0):.4f} | "
                f"RS/Seg: {struct_diag_means.get('weighted_rs_over_seg', 0.0):.4f} | "
                f"Chunk/Seg: {struct_diag_means.get('weighted_chunk_over_seg', 0.0):.4f} | "
                f"GradNorm main/st/rs/chunk: "
                f"{struct_diag_means.get('grad_norm_main', 0.0):.4f}/"
                f"{struct_diag_means.get('grad_norm_st', 0.0):.4f}/"
                f"{struct_diag_means.get('grad_norm_rs', 0.0):.4f}/"
                f"{struct_diag_means.get('grad_norm_chunk', 0.0):.4f} | "
                f"Cos main-st/rs/chunk: "
                f"{struct_diag_means.get('cos_main_st', 0.0):.4f}/"
                f"{struct_diag_means.get('cos_main_rs', 0.0):.4f}/"
                f"{struct_diag_means.get('cos_main_chunk', 0.0):.4f} | "
                f"Z st/rs/chunk: "
                f"{struct_diag_means.get('z_st', 0.0):.4f}/"
                f"{struct_diag_means.get('z_rs', 0.0):.4f}/"
                f"{struct_diag_means.get('z_chunk', 0.0):.4f} | "
                f"Alpha: {struct_diag_means.get('alpha', 0.0):.4f} | "
                f"P st/rs/chunk: "
                f"{struct_diag_means.get('p_st', 0.0):.4f}/"
                f"{struct_diag_means.get('p_rs', 0.0):.4f}/"
                f"{struct_diag_means.get('p_chunk', 0.0):.4f} | "
                f"Lambda st/rs/chunk: "
                f"{struct_diag_means.get('lambda_st', 0.0):.4f}/"
                f"{struct_diag_means.get('lambda_rs', 0.0):.4f}/"
                f"{struct_diag_means.get('lambda_chunk', 0.0):.4f} | "
                f"EMA st/rs/chunk: "
                f"{struct_diag_means.get('ema_st', 0.0):.4f}/"
                f"{struct_diag_means.get('ema_rs', 0.0):.4f}/"
                f"{struct_diag_means.get('ema_chunk', 0.0):.4f} | "
                f"W st/rs/chunk: "
                f"{struct_diag_means.get('w_st', 0.0):.4f}/"
                f"{struct_diag_means.get('w_rs', 0.0):.4f}/"
                f"{struct_diag_means.get('w_chunk', 0.0):.4f} | "
                f"Logit st/rs/chunk: "
                f"{struct_diag_means.get('a_st', 0.0):.4f}/"
                f"{struct_diag_means.get('a_rs', 0.0):.4f}/"
                f"{struct_diag_means.get('a_chunk', 0.0):.4f} | "
                f"RewardObs st/rs/chunk: "
                f"{struct_diag_means.get('reward_st_obs', 0.0):.4f}/"
                f"{struct_diag_means.get('reward_rs_obs', 0.0):.4f}/"
                f"{struct_diag_means.get('reward_chunk_obs', 0.0):.4f} | "
                f"Reward st/rs/chunk: "
                f"{struct_diag_means.get('reward_st', 0.0):.4f}/"
                f"{struct_diag_means.get('reward_rs', 0.0):.4f}/"
                f"{struct_diag_means.get('reward_chunk', 0.0):.4f} | "
                f"RewardTilde st/rs/chunk: "
                f"{struct_diag_means.get('reward_tilde_st', 0.0):.4f}/"
                f"{struct_diag_means.get('reward_tilde_rs', 0.0):.4f}/"
                f"{struct_diag_means.get('reward_tilde_chunk', 0.0):.4f} | "
                f"NormLoss st/rs/chunk: "
                f"{struct_diag_means.get('norm_loss_st', 0.0):.4f}/"
                f"{struct_diag_means.get('norm_loss_rs', 0.0):.4f}/"
                f"{struct_diag_means.get('norm_loss_chunk', 0.0):.4f} | "
                f"AuxMean: {struct_diag_means.get('aux_loss_mean', 0.0):.4f} | "
                f"Gamma: {struct_diag_means.get('gamma', 0.0):.4f} | "
                f"GammaAux/Seg: {struct_diag_means.get('gamma_aux_over_seg', 0.0):.4f} | "
            )
        log_msg += (
            f"Val Total: {mean_val_loss:.4f} | "
            f"Val BCE: {val_loss_parts['bce']:.4f} | "
            f"Val DiceLoss: {val_loss_parts['dice_loss']:.4f} | "
            f"Val DiceMetric: {mean_val_dice:.4f}"
        )
        logger.info(log_msg)
        if cfg.MODEL.CLIP_MODEL == "structxlip":
            diag_row = {column: 0.0 for column in STRUCTXLIP_DIAGNOSTIC_COLUMNS}
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
            append_epoch_diagnostics_csv(structxlip_diagnostics_path, diag_row)

        # Save the best model based on validation loss
        if mean_val_dice > best_dice:
            logger.info(f"New best Dice: {best_dice:.4f} → {mean_val_dice:.4f}")
            best_dice = mean_val_dice
            torch.save({
                "model": model.state_dict(),
                "learnable_tau_loss": learnable_tau_loss.state_dict() if learnable_tau_loss is not None else None,
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
            "learnable_tau_loss": learnable_tau_loss.state_dict() if learnable_tau_loss is not None else None,
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