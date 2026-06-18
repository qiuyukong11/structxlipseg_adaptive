"""Adaptive StructXLIP-inspired segmentation loss.

This module keeps the primary segmentation objective outside the adaptive
contrastive-temperature mechanism and learns one log-temperature per auxiliary
structural contrastive loss.
"""

from __future__ import annotations

from typing import Dict, Tuple

import math
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SegWithStructXLIPLoss(nn.Module):
    """Segmentation + StructXLIP-inspired auxiliary contrastive losses.

    Inputs expected by ``forward``:
        seg_pred:     [B, K, H, W] segmentation logits.
        seg_gt:       [B, H, W] integer segmentation labels.
        f_color:      [B, C] global color-image visual features.
        f_edge:       [B, C] global edge/structure visual features.
        f_local_edge: [B, P, C] local edge patch features.
        f_tokens:     [B, T, C] text token features.
        t_struct:     [B, C] global structural text features.

    The module returns ``(total_loss, loss_dict)`` where ``loss_dict`` contains
    detached scalar diagnostics and the clamped temperature scales.
    """

    def __init__(
        self,
        max_tau_scale: float = 100.0,
        init_temperature: float = 0.07,
        min_temperature: float = 0.01,
        ignore_index: int = -100,
        eps: float = 1e-6,
        aux_weight: float = 1.0,
        tau_regularizer_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if init_temperature <= 0.0:
            raise ValueError("init_temperature must be positive")
        if max_tau_scale <= 0.0:
            raise ValueError("max_tau_scale must be positive")
        if min_temperature <= 0.0:
            raise ValueError("min_temperature must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if aux_weight < 0.0:
            raise ValueError("aux_weight must be non-negative")
        if tau_regularizer_weight < 0.0:
            raise ValueError("tau_regularizer_weight must be non-negative")

        init_log_tau = float(np.log(1.0 / init_temperature))
        self.log_tau_global = nn.Parameter(torch.tensor(init_log_tau, dtype=torch.float32))
        self.log_tau_local = nn.Parameter(torch.tensor(init_log_tau, dtype=torch.float32))
        self.log_tau_drift = nn.Parameter(torch.tensor(init_log_tau, dtype=torch.float32))

        self.max_tau_scale = float(max_tau_scale)
        self.min_temperature = float(min_temperature)
        self.effective_max_tau_scale = min(self.max_tau_scale, 1.0 / self.min_temperature)
        self.ignore_index = int(ignore_index)
        self.eps = float(eps)
        self.aux_weight = float(aux_weight)
        self.init_tau_scale = float(1.0 / init_temperature)
        self.tau_regularizer_weight = float(tau_regularizer_weight)

    def forward(
        self,
        seg_pred: Tensor,
        seg_gt: Tensor,
        f_color: Tensor,
        f_edge: Tensor,
        f_local_edge: Tensor,
        f_tokens: Tensor,
        t_struct: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute total adaptive StructXLIP segmentation loss.

        Args:
            seg_pred: [B, K, H, W] segmentation logits.
            seg_gt: [B, H, W] integer class labels.
            f_color: [B, C] global color-image features.
            f_edge: [B, C] global edge/structure features.
            f_local_edge: [B, P, C] edge patch features.
            f_tokens: [B, T, C] text token embeddings.
            t_struct: [B, C] global structural text embeddings.
        """
        raise RuntimeError(
            "SegWithStructXLIPLoss.forward is disabled for fair fixed-vs-adaptive comparisons. "
            "Use forward_raw_losses with the original StructXLIP loss tensors instead."
        )

        # Primary segmentation loss over pixel logits: seg_pred [B, K, H, W], seg_gt [B, H, W].
        loss_seg = F.cross_entropy(seg_pred, seg_gt.long(), ignore_index=self.ignore_index)

        # L2-normalized global features: all [B, C].
        f_color_n = F.normalize(f_color.float(), dim=-1, eps=self.eps)
        f_edge_n = F.normalize(f_edge.float(), dim=-1, eps=self.eps)
        t_struct_n = F.normalize(t_struct.float(), dim=-1, eps=self.eps)

        # L2-normalized local features: f_local_edge [B, P, C], f_tokens [B, T, C].
        f_local_edge_n = F.normalize(f_local_edge.float(), dim=-1, eps=self.eps)
        f_tokens_n = F.normalize(f_tokens.float(), dim=-1, eps=self.eps)

        scale_global = self._tau_scale(self.log_tau_global)
        scale_local = self._tau_scale(self.log_tau_local)
        scale_drift = self._tau_scale(self.log_tau_drift)

        # Auxiliary 1: global edge-to-structure-text InfoNCE, logits [B, B].
        logits_global = scale_global * (f_edge_n @ t_struct_n.t())
        loss_global = self._symmetric_infonce(logits_global)

        # Auxiliary 2: cross-batch local patch-token matching.
        # local_sim[b, n, p, t] compares image b's edge patch p to text n's token t.
        local_sim = torch.einsum("bpc,ntc->bnpt", f_local_edge_n, f_tokens_n)
        local_sim = scale_local * local_sim
        # Image-to-text score [B, B]: each patch chooses its best token, then average patches.
        logits_patch_to_token = local_sim.max(dim=3).values.mean(dim=2)
        # Text-to-image score [B, B]: each token chooses its best patch, then average tokens.
        logits_token_to_patch = local_sim.max(dim=2).values.mean(dim=2)
        loss_local = self._bidirectional_ce(logits_patch_to_token, logits_token_to_patch)

        # Auxiliary 3: representation drift restraint between color and edge features, logits [B, B].
        logits_drift = scale_drift * (f_color_n @ f_edge_n.t())
        loss_drift = self._symmetric_infonce(logits_drift)

        weighted_loss_global = self.aux_weight * loss_global
        weighted_loss_local = self.aux_weight * loss_local
        weighted_loss_drift = self.aux_weight * loss_drift
        loss_struct = weighted_loss_global + weighted_loss_local + weighted_loss_drift
        total_loss = loss_seg + loss_struct
        loss_dict = {
            "loss_total": total_loss.detach(),
            "loss_seg": loss_seg.detach(),
            "loss_global": loss_global.detach(),
            "loss_local": loss_local.detach(),
            "loss_drift": loss_drift.detach(),
            "weighted_loss_global": weighted_loss_global.detach(),
            "weighted_loss_local": weighted_loss_local.detach(),
            "weighted_loss_drift": weighted_loss_drift.detach(),
            "loss_struct": loss_struct.detach(),
            "aux_weight": torch.tensor(self.aux_weight, device=seg_pred.device),
            "tau_scale_global": scale_global.detach(),
            "tau_scale_local": scale_local.detach(),
            "tau_scale_drift": scale_drift.detach(),
            "log_tau_global": self.log_tau_global.detach(),
            "log_tau_local": self.log_tau_local.detach(),
            "log_tau_drift": self.log_tau_drift.detach(),
        }
        return total_loss, loss_dict


    def forward_raw_losses(
        self,
        loss_st: Tensor,
        loss_rs: Tensor,
        loss_chunk_align: Tensor,
        base_lambda_st: float,
        base_lambda_rs: float,
        base_lambda_chunk: float,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Scale the original StructXLIP losses without redefining them.

        The three input losses must be the raw tensors produced by
        ``ClipSegStructXLIP.compute_structxlip_losses``. This path changes only
        the auxiliary weights, so fixed-weight and adaptive runs use identical
        loss formulas. The learned temperature scales are normalized by their
        initialization, making the initial effective lambdas equal to the fixed
        StructXLIP lambdas times ``aux_weight``.
        """
        device = loss_st.device if torch.is_tensor(loss_st) else loss_rs.device
        dtype = loss_st.dtype if torch.is_tensor(loss_st) else torch.float32
        base_st = torch.as_tensor(float(base_lambda_st), device=device, dtype=dtype)
        base_rs = torch.as_tensor(float(base_lambda_rs), device=device, dtype=dtype)
        base_chunk = torch.as_tensor(float(base_lambda_chunk), device=device, dtype=dtype)
        aux_weight = torch.as_tensor(self.aux_weight, device=device, dtype=dtype)

        multiplier_st = self._tau_scale(self.log_tau_global).to(device=device, dtype=dtype) / self.init_tau_scale
        multiplier_rs = self._tau_scale(self.log_tau_drift).to(device=device, dtype=dtype) / self.init_tau_scale
        multiplier_chunk = self._tau_scale(self.log_tau_local).to(device=device, dtype=dtype) / self.init_tau_scale

        lambda_st = aux_weight * base_st * multiplier_st
        lambda_rs = aux_weight * base_rs * multiplier_rs
        lambda_chunk = aux_weight * base_chunk * multiplier_chunk

        weighted_loss_st = lambda_st * loss_st
        weighted_loss_rs = lambda_rs * loss_rs
        weighted_loss_chunk = lambda_chunk * loss_chunk_align
        weighted_struct_total = weighted_loss_st + weighted_loss_rs + weighted_loss_chunk

        # Uncertainty-style log penalty: discourages the learned multipliers from
        # collapsing to zero while preserving the original raw loss formulas.
        reg_weight = torch.as_tensor(self.tau_regularizer_weight, device=device, dtype=dtype)
        tau_reg_st = -reg_weight * torch.log(torch.clamp(multiplier_st, min=self.eps))
        tau_reg_rs = -reg_weight * torch.log(torch.clamp(multiplier_rs, min=self.eps))
        tau_reg_chunk = -reg_weight * torch.log(torch.clamp(multiplier_chunk, min=self.eps))
        tau_reg_loss = tau_reg_st + tau_reg_rs + tau_reg_chunk
        loss_struct = weighted_struct_total + tau_reg_loss

        return loss_struct, {
            "loss_st": loss_st.detach(),
            "loss_rs": loss_rs.detach(),
            "loss_chunk_align": loss_chunk_align.detach(),
            "weighted_loss_st": weighted_loss_st.detach(),
            "weighted_loss_rs": weighted_loss_rs.detach(),
            "weighted_loss_chunk_align": weighted_loss_chunk.detach(),
            "weighted_struct_total": weighted_struct_total.detach(),
            "tau_reg_st": tau_reg_st.detach(),
            "tau_reg_rs": tau_reg_rs.detach(),
            "tau_reg_chunk": tau_reg_chunk.detach(),
            "tau_reg_loss": tau_reg_loss.detach(),
            "struct_objective_total": loss_struct.detach(),
            "loss_struct": loss_struct.detach(),
            "lambda_st": lambda_st.detach(),
            "lambda_rs": lambda_rs.detach(),
            "lambda_chunk": lambda_chunk.detach(),
            "tau_scale_global": self._tau_scale(self.log_tau_global).detach(),
            "tau_scale_local": self._tau_scale(self.log_tau_local).detach(),
            "tau_scale_drift": self._tau_scale(self.log_tau_drift).detach(),
            "temperature_global": (1.0 / self._tau_scale(self.log_tau_global)).detach(),
            "temperature_local": (1.0 / self._tau_scale(self.log_tau_local)).detach(),
            "temperature_drift": (1.0 / self._tau_scale(self.log_tau_drift)).detach(),
            "multiplier_st": multiplier_st.detach(),
            "multiplier_rs": multiplier_rs.detach(),
            "multiplier_chunk": multiplier_chunk.detach(),
            "aux_weight": aux_weight.detach(),
            "tau_regularizer_weight": reg_weight.detach(),
            "min_temperature": torch.as_tensor(self.min_temperature, device=device, dtype=dtype),
        }

    def _tau_scale(self, log_tau: Tensor) -> Tensor:
        # scale = 1 / temperature. Clamping scale keeps temperature >= min_temperature.
        scale = torch.exp(log_tau)
        return torch.clamp(scale, max=self.effective_max_tau_scale)

    @staticmethod
    def _labels(batch_size: int, device: torch.device) -> Tensor:
        return torch.arange(batch_size, device=device, dtype=torch.long)

    def _symmetric_infonce(self, logits: Tensor) -> Tensor:
        # logits [B, B], diagonal entries are positive pairs.
        labels = self._labels(logits.shape[0], logits.device)
        loss_a_to_b = F.cross_entropy(logits, labels)
        loss_b_to_a = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_a_to_b + loss_b_to_a)

    def _bidirectional_ce(self, logits_a_to_b: Tensor, logits_b_to_a: Tensor) -> Tensor:
        # Both inputs are [B, B], diagonal entries are positive pairs.
        labels = self._labels(logits_a_to_b.shape[0], logits_a_to_b.device)
        loss_a_to_b = F.cross_entropy(logits_a_to_b, labels)
        loss_b_to_a = F.cross_entropy(logits_b_to_a.t(), labels)
        return 0.5 * (loss_a_to_b + loss_b_to_a)

    @staticmethod
    def _validate_inputs(
        seg_pred: Tensor,
        seg_gt: Tensor,
        f_color: Tensor,
        f_edge: Tensor,
        f_local_edge: Tensor,
        f_tokens: Tensor,
        t_struct: Tensor,
    ) -> None:
        if seg_pred.ndim != 4:
            raise ValueError(f"seg_pred must be [B, K, H, W], got shape {tuple(seg_pred.shape)}")
        if seg_gt.ndim != 3:
            raise ValueError(f"seg_gt must be [B, H, W], got shape {tuple(seg_gt.shape)}")
        if f_color.ndim != 2 or f_edge.ndim != 2 or t_struct.ndim != 2:
            raise ValueError("f_color, f_edge, and t_struct must all be [B, C]")
        if f_local_edge.ndim != 3 or f_tokens.ndim != 3:
            raise ValueError("f_local_edge and f_tokens must be [B, N, C]")

        batch_size = seg_pred.shape[0]
        batch_tensors = {
            "seg_gt": seg_gt,
            "f_color": f_color,
            "f_edge": f_edge,
            "f_local_edge": f_local_edge,
            "f_tokens": f_tokens,
            "t_struct": t_struct,
        }
        for name, tensor in batch_tensors.items():
            if tensor.shape[0] != batch_size:
                raise ValueError(f"{name} batch size {tensor.shape[0]} does not match seg_pred batch size {batch_size}")

        spatial = tuple(seg_pred.shape[-2:])
        if tuple(seg_gt.shape[-2:]) != spatial:
            raise ValueError(f"seg_gt spatial shape {tuple(seg_gt.shape[-2:])} does not match seg_pred {spatial}")

        channel_dim = f_color.shape[-1]
        channel_tensors = {
            "f_edge": f_edge,
            "t_struct": t_struct,
            "f_local_edge": f_local_edge,
            "f_tokens": f_tokens,
        }
        for name, tensor in channel_tensors.items():
            if tensor.shape[-1] != channel_dim:
                raise ValueError(f"{name} channel dim {tensor.shape[-1]} does not match f_color dim {channel_dim}")

        if not math.isfinite(float(channel_dim)) or channel_dim <= 0:
            raise ValueError("feature channel dimension must be positive")
