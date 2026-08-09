"""
JumpReLU Sparse Autoencoder (Rajamanoharan et al. 2024).
Applies a threshold step function over pre-activations, allowing immediate non-attenuated
activations above the threshold theta without L1 attenuation.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from .base import BaseSAE
from ..common.config import SAEConfig


class JumpReLUSAE(BaseSAE):
    """
    JumpReLU Sparse Autoencoder:
        z = (x - b_dec) @ W_enc + b_enc
        f(x) = ReLU(z) * (z > threshold)
        x_hat = f(x) @ W_dec + b_dec
    """

    def __init__(self, cfg: SAEConfig):
        super().__init__(cfg)
        self.threshold = cfg.jumprelu_threshold
        self.l1_coeff = cfg.l1_coeff

        self.W_enc = nn.Parameter(
            torch.empty(self.d_in, self.d_sae).normal_(mean=0.0, std=1.0 / (self.d_in**0.5))
        )
        self.b_enc = nn.Parameter(torch.zeros(self.d_sae))

        # Learnable per-feature threshold initialized to log(cfg.jumprelu_threshold)
        self.log_threshold = nn.Parameter(torch.full((self.d_sae,), fill_value=torch.log(torch.tensor(max(self.threshold, 1e-4)))))

        self.W_dec = nn.Parameter(self.W_enc.data.clone().t())
        self.normalize_decoder_columns()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_cent = x - self.b_dec if self.cfg.tied_bias else x
        pre_acts = torch.matmul(x_cent, self.W_enc) + self.b_enc
        threshold = torch.exp(self.log_threshold)
        jump_mask = (pre_acts > threshold).float()
        return pre_acts * jump_mask

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return torch.matmul(f, self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        x_cent = x - self.b_dec if self.cfg.tied_bias else x
        pre_acts = torch.matmul(x_cent, self.W_enc) + self.b_enc
        threshold = torch.exp(self.log_threshold)

        # Straight-Through Estimator (STE) for threshold gradient
        jump_mask_hard = (pre_acts > threshold).float()
        # Sigmoid approximation for smooth backward pass
        jump_mask_soft = torch.sigmoid((pre_acts - threshold) * 10.0)
        jump_mask = jump_mask_hard.detach() + jump_mask_soft - jump_mask_soft.detach()

        f = pre_acts * jump_mask
        x_hat = self.decode(f)

        # L0-like regularizer using soft mask
        l0_proxy_loss = self.l1_coeff * torch.mean(torch.sum(jump_mask_soft, dim=-1))

        info = {
            "l0": jump_mask_hard.sum(dim=-1).mean(),
            "aux_loss": l0_proxy_loss,
            "mean_threshold": threshold.mean(),
            "mean_activation": f[f > 0].mean() if (f > 0).any() else torch.tensor(0.0),
        }

        return x_hat, f, info
