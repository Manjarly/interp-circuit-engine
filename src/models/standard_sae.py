"""
Standard L1-penalized Sparse Autoencoder (Cunningham et al. 2023; Bricken et al., Anthropic 2023).
Classical baseline using L1 regularization over ReLU activations.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from .base import BaseSAE
from ..common.config import SAEConfig


class StandardSAE(BaseSAE):
    """
    Standard Sparse Autoencoder:
        f(x) = ReLU( (x - b_dec) @ W_enc + b_enc )
        x_hat = f(x) @ W_dec + b_dec
    """

    def __init__(self, cfg: SAEConfig):
        super().__init__(cfg)
        self.l1_coeff = cfg.l1_coeff

        self.W_enc = nn.Parameter(
            torch.empty(self.d_in, self.d_sae).normal_(mean=0.0, std=1.0 / (self.d_in**0.5))
        )
        self.b_enc = nn.Parameter(torch.zeros(self.d_sae))

        self.W_dec = nn.Parameter(self.W_enc.data.clone().t())
        self.normalize_decoder_columns()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_cent = x - self.b_dec if self.cfg.tied_bias else x
        pre_acts = torch.matmul(x_cent, self.W_enc) + self.b_enc
        return torch.relu(pre_acts)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return torch.matmul(f, self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        f = self.encode(x)
        x_hat = self.decode(f)

        # L1 loss
        l1_loss = self.l1_coeff * torch.mean(torch.sum(f, dim=-1))

        info = {
            "l0": (f > 0).float().sum(dim=-1).mean(),
            "aux_loss": l1_loss,
            "l1_loss": l1_loss,
            "mean_activation": f[f > 0].mean() if (f > 0).any() else torch.tensor(0.0),
        }

        return x_hat, f, info
