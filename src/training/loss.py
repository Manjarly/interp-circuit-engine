"""
Loss functions and evaluation metrics for Sparse Autoencoders.
Includes Normalized Mean Squared Error (NMSE), Explained Variance, and L0 Sparsity.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn


def compute_sae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    f: torch.Tensor,
    info: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes overall SAE training loss and logging metrics.

    Metrics computed:
        - mse_loss: ||x - x_hat||^2
        - normalized_mse: ||x - x_hat||^2 / ||x - x.mean()||^2
        - explained_variance: 1 - Var(x - x_hat) / Var(x)
        - l0: average number of non-zero active latents per token
        - total_loss: mse_loss + aux_loss (if present in info)
    """
    # Reconstruction MSE
    mse_loss = torch.mean((x - x_hat) ** 2)

    # Variance-normalized MSE (NMSE)
    x_centered = x - torch.mean(x, dim=0, keepdim=True)
    baseline_variance = torch.mean(x_centered ** 2).clamp(min=1e-8)
    normalized_mse = mse_loss / baseline_variance

    # Explained Variance: 1 - Var(error) / Var(original)
    error = x - x_hat
    error_var = torch.var(error, dim=0).mean()
    total_var = torch.var(x, dim=0).mean().clamp(min=1e-8)
    explained_variance = (1.0 - (error_var / total_var)).item()

    # Sparsity
    l0 = (f > 0).float().sum(dim=-1).mean().item()

    # Auxiliary loss (TopK aux loss or Gated L1 loss)
    aux_loss = info.get("aux_loss", torch.tensor(0.0, device=x.device))
    total_loss = mse_loss + aux_loss

    metrics = {
        "loss/total": total_loss.item(),
        "loss/mse": mse_loss.item(),
        "loss/normalized_mse": normalized_mse.item(),
        "loss/aux": aux_loss.item() if isinstance(aux_loss, torch.Tensor) else float(aux_loss),
        "metrics/explained_variance": explained_variance,
        "metrics/l0": l0,
    }

    return total_loss, metrics
