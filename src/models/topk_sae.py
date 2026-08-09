"""
Top-K Sparse Autoencoder (Gao et al., OpenAI 2024; Anthropic 2024).
Eliminates L1 shrinkage bias by taking the top-K highest activations directly.
Includes auxiliary dead-latent loss for constant gradient flow.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from .base import BaseSAE
from ..common.config import SAEConfig


class TopKSAE(BaseSAE):
    """
    Top-K Sparse Autoencoder.
    Forward pass:
        1. Centering: x_cent = x - b_dec
        2. Pre-activation: z = x_cent @ W_enc + b_enc
        3. Top-K Activation: f = TopK(ReLU(z), k)
        4. Reconstruction: x_hat = f @ W_dec + b_dec
    """

    def __init__(self, cfg: SAEConfig):
        super().__init__(cfg)
        self.k = cfg.k
        self.aux_loss_coeff = cfg.aux_loss_coeff

        # Encoder weights and bias: W_enc (d_in, d_sae)
        self.W_enc = nn.Parameter(
            torch.empty(self.d_in, self.d_sae).normal_(mean=0.0, std=1.0 / (self.d_in**0.5))
        )
        self.b_enc = nn.Parameter(torch.zeros(self.d_sae))

        # Decoder weights: W_dec (d_sae, d_in)
        # Initialized as transpose of W_enc
        self.W_dec = nn.Parameter(self.W_enc.data.clone().t())
        self.normalize_decoder_columns()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes x into Top-K sparse activations f.
        """
        x_cent = x - self.b_dec if self.cfg.tied_bias else x
        pre_acts = torch.matmul(x_cent, self.W_enc) + self.b_enc
        acts = torch.relu(pre_acts)

        k = min(self.k, acts.shape[-1])
        topk_values, topk_indices = torch.topk(acts, k=k, dim=-1)

        # Sparse tensor reconstruction without inplace mutation
        f = torch.zeros_like(acts)
        f = f.scatter(dim=-1, index=topk_indices, src=topk_values)
        return f

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs x_hat from sparse latents f.
        """
        return torch.matmul(f, self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Executes forward pass and returns reconstruction, latents, and auxiliary metrics.
        """
        x_cent = x - self.b_dec if self.cfg.tied_bias else x
        pre_acts = torch.matmul(x_cent, self.W_enc) + self.b_enc
        acts = torch.relu(pre_acts)

        k = min(self.k, acts.shape[-1])
        topk_values, topk_indices = torch.topk(acts, k=k, dim=-1)

        # Non-inplace scatter for safe autograd backward
        f = torch.zeros_like(acts)
        f = f.scatter(dim=-1, index=topk_indices, src=topk_values)

        x_hat = torch.matmul(f, self.W_dec) + self.b_dec

        # Compute auxiliary loss for dead latents
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.training and self.aux_loss_coeff > 0.0:
            residual = (x - x_hat).detach()
            aux_pre_acts = torch.matmul(residual, self.W_enc) + self.b_enc
            aux_acts = torch.relu(aux_pre_acts)

            # Zero out topk features using non-in-place scatter
            zeros = torch.zeros_like(aux_acts)
            masked_aux_acts = aux_acts.scatter(dim=-1, index=topk_indices, src=zeros)

            aux_topk_val, _ = torch.topk(masked_aux_acts, k=min(k, aux_acts.shape[-1]), dim=-1)
            aux_loss = self.aux_loss_coeff * torch.mean(aux_topk_val**2)

        info = {
            "l0": (f > 0).float().sum(dim=-1).mean(),
            "aux_loss": aux_loss,
            "mean_activation": f[f > 0].mean() if (f > 0).any() else torch.tensor(0.0),
        }

        return x_hat, f, info
