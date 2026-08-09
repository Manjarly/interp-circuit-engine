"""
Gated Sparse Autoencoder (Rajamanoharan et al., Google DeepMind 2024).
Decouples feature detection (gating path) from magnitude estimation (magnitude path),
completely resolving the L1 shrinkage problem while maintaining sparsity.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from .base import BaseSAE
from ..common.config import SAEConfig


class GatedSAE(BaseSAE):
    """
    Gated Sparse Autoencoder.
    Forward Pass:
        1. Gating pathway:   pi(x) = (x - b_dec) @ W_gate + b_gate
        2. Magnitude pathway: r(x)  = (x - b_dec) @ W_mag  + b_mag
        3. Gated activation:  f(x)  = (pi(x) > 0) * ReLU(r(x))
        4. Reconstruction:    x_hat = f(x) @ W_dec + b_dec
    """

    def __init__(self, cfg: SAEConfig):
        super().__init__(cfg)
        self.l1_coeff = cfg.l1_coeff

        # Gating pathway weights & bias
        self.W_gate = nn.Parameter(
            torch.empty(self.d_in, self.d_sae).normal_(mean=0.0, std=1.0 / (self.d_in**0.5))
        )
        self.b_gate = nn.Parameter(torch.zeros(self.d_sae))

        # Magnitude pathway weights & bias (optionally tied to W_gate via scaling)
        # Using separate magnitude weights for maximum expressive power
        self.r_mag = nn.Parameter(torch.zeros(self.d_sae)) # Per-feature log-scale multiplier
        self.b_mag = nn.Parameter(torch.zeros(self.d_sae))

        # Decoder weights: (d_sae, d_in)
        self.W_dec = nn.Parameter(self.W_gate.data.clone().t())
        self.normalize_decoder_columns()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the gated sparse activations f(x).
        """
        x_cent = x - self.b_dec if self.cfg.tied_bias else x

        # Gating pre-activation
        pi_gate = torch.matmul(x_cent, self.W_gate) + self.b_gate
        gate_active = (pi_gate > 0).float()

        # Magnitude pre-activation
        # r_mag scales the effective encoder projection
        W_mag = self.W_gate * torch.exp(self.r_mag)
        mag_pre_acts = torch.matmul(x_cent, W_mag) + self.b_mag
        mag_acts = torch.relu(mag_pre_acts)

        # Gated combination
        f = gate_active * mag_acts
        return f

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs x_hat from sparse latents f.
        """
        return torch.matmul(f, self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass computing reconstruction, latents, and auxiliary gating loss.
        """
        x_cent = x - self.b_dec if self.cfg.tied_bias else x

        pi_gate = torch.matmul(x_cent, self.W_gate) + self.b_gate
        gate_active = (pi_gate > 0).float()

        W_mag = self.W_gate * torch.exp(self.r_mag)
        mag_pre_acts = torch.matmul(x_cent, W_mag) + self.b_mag
        mag_acts = torch.relu(mag_pre_acts)

        f = gate_active * mag_acts
        x_hat = torch.matmul(f, self.W_dec) + self.b_dec

        # Auxiliary gating reconstruction loss (DeepMind formulation)
        # Using ReLU(pi_gate) through detached decoder to guide gating path
        pi_relu = torch.relu(pi_gate)
        x_gate_recon = torch.matmul(pi_relu, self.W_dec.detach()) + self.b_dec.detach()
        aux_gate_loss = torch.mean((x - x_gate_recon) ** 2)

        # L1 sparsity on the gating pathway
        l1_loss = self.l1_coeff * torch.mean(torch.sum(pi_relu, dim=-1))

        info = {
            "l0": (f > 0).float().sum(dim=-1).mean(),
            "aux_loss": aux_gate_loss + l1_loss,
            "l1_loss": l1_loss,
            "mean_activation": f[f > 0].mean() if (f > 0).any() else torch.tensor(0.0),
        }

        return x_hat, f, info
