from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import (
    compute_adaptive_gradbudget_align,
    make_gradbudget_align_state,
    safe_float,
)


def default_cfg():
    return {
        "enabled": True,
        "beta": 0.0,
        "reward_ema": 0.0,
        "temperature": 0.5,
        "gamma_max": 0.5,
        "gamma_min": 0.0,
        "eps": 1.0e-8,
        "clamp_aux_loss_nonneg": True,
        "detach_lambdas": True,
        "min_weight": 0.05,
        "use_positive_alignment_gate": True,
        "lambda_max": 10.0,
    }


def run_alignment_case():
    theta = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5]))
    params = [theta]
    main_loss = (theta * theta).sum()
    loss_st = (theta * theta).sum()
    loss_rs = 10.0 - (theta * theta).sum()
    loss_chunk = ((theta[0] - theta[1]) ** 2) + 0.1 * (theta[2] ** 2)

    lambdas, diag, parts = compute_adaptive_gradbudget_align(
        main_loss,
        {
            "loss_st": loss_st,
            "loss_rs": loss_rs,
            "loss_chunk_align": loss_chunk,
        },
        params,
        default_cfg(),
        make_gradbudget_align_state(),
    )

    lambda_values = torch.tensor([
        lambdas["lambda_st"],
        lambdas["lambda_rs"],
        lambdas["lambda_chunk"],
    ])
    assert torch.isfinite(lambda_values).all(), f"non-finite lambdas: {lambdas}"
    assert safe_float(parts["weighted_struct_total"]) >= 0.0, "weighted_struct_total is negative"
    assert lambdas["lambda_st"] > lambdas["lambda_rs"], (lambdas, diag)
    return lambdas, diag, parts


def run_negative_aux_case():
    theta = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5]))
    params = [theta]
    main_loss = (theta * theta).sum()
    loss_st = -((theta * theta).sum())
    loss_rs = 10.0 - (theta * theta).sum()
    loss_chunk = ((theta[0] - theta[1]) ** 2) + 0.1 * (theta[2] ** 2)

    lambdas, diag, parts = compute_adaptive_gradbudget_align(
        main_loss,
        {
            "loss_st": loss_st,
            "loss_rs": loss_rs,
            "loss_chunk_align": loss_chunk,
        },
        params,
        default_cfg(),
        make_gradbudget_align_state(),
    )

    lambda_values = torch.tensor([
        lambdas["lambda_st"],
        lambdas["lambda_rs"],
        lambdas["lambda_chunk"],
    ])
    assert torch.isfinite(lambda_values).all(), f"non-finite lambdas: {lambdas}"
    assert safe_float(parts["weighted_loss_st"]) >= 0.0, "negative clamped weighted loss_st"
    assert safe_float(parts["weighted_struct_total"]) >= 0.0, "negative weighted_struct_total"
    return lambdas, diag, parts


def main():
    lambdas, diag, parts = run_alignment_case()
    neg_lambdas, neg_diag, neg_parts = run_negative_aux_case()
    print("alignment lambdas:", lambdas)
    print("alignment gamma:", diag["gamma"])
    print("alignment cosines:", diag["cos_main_st"], diag["cos_main_rs"], diag["cos_main_chunk"])
    print("alignment weights:", diag["w_st"], diag["w_rs"], diag["w_chunk"])
    print("weighted_struct_total:", safe_float(parts["weighted_struct_total"]))
    print("negative aux lambdas:", neg_lambdas)
    print("negative aux gamma:", neg_diag["gamma"])
    print("negative aux weighted_struct_total:", safe_float(neg_parts["weighted_struct_total"]))
    print("ADAPTIVE_GRADBUDGET_ALIGN smoke passed")


if __name__ == "__main__":
    main()
