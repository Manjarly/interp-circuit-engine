"""
Dead Latent Tracker and Active Feature Resampler (Anthropic / OpenAI dictionary learning).
Monitors feature firing frequency, flags inactive latents, and resamples their weights
to point in the direction of maximum residual reconstruction error.
"""

from typing import Tuple
import torch
import torch.nn as nn
from ..models.base import BaseSAE


class DeadLatentTracker:
    """
    Tracks step-level feature activity and executes dead neuron resampling.
    """

    def __init__(self, d_sae: int, dead_threshold_tokens: int = 50_000, device: str = "cpu"):
        self.d_sae = d_sae
        self.dead_threshold_tokens = dead_threshold_tokens
        self.device = device
        # Tracks tokens processed since each feature last fired
        self.tokens_since_last_fired = torch.zeros(d_sae, device=device, dtype=torch.long)
        self.total_tokens_seen = 0

    @torch.no_grad()
    def update(self, f: torch.Tensor) -> None:
        """
        Updates firing history based on active latents f of shape (batch_size, d_sae).
        """
        batch_size = f.shape[0]
        self.total_tokens_seen += batch_size

        # Boolean mask of which features fired in this batch
        fired = (f > 0).any(dim=0) # (d_sae,)

        # Increment all
        self.tokens_since_last_fired += batch_size
        # Reset counters for features that fired
        self.tokens_since_last_fired[fired] = 0

    def get_dead_latents_mask(self) -> torch.Tensor:
        """
        Returns a boolean mask of features that haven't fired for > dead_threshold_tokens.
        """
        return self.tokens_since_last_fired >= self.dead_threshold_tokens

    def get_dead_count(self) -> int:
        return int(self.get_dead_latents_mask().sum().item())

    @torch.no_grad()
    def resample_dead_latents(
        self,
        sae: BaseSAE,
        optimizer: torch.optim.Optimizer,
        residuals: torch.Tensor,
    ) -> int:
        """
        Resamples dead feature weights to align with high-error residual vectors.

        Args:
            sae: BaseSAE instance
            optimizer: Optimizer to reset momentum states for resampled parameters
            residuals: Residual errors (x - x_hat) of shape (N, d_in)
        """
        dead_mask = self.get_dead_latents_mask()
        dead_indices = torch.where(dead_mask)[0]
        num_dead = len(dead_indices)

        if num_dead == 0 or residuals.shape[0] == 0:
            return 0

        # Sample residuals proportionally to their squared error norms
        residual_norms = torch.norm(residuals, p=2, dim=-1)
        probs = (residual_norms**2) / (residual_norms**2).sum().clamp(min=1e-8)

        # Draw replacement vectors with replacement
        sample_indices = torch.multinomial(probs, num_samples=num_dead, replacement=True)
        sampled_residuals = residuals[sample_indices] # (num_dead, d_in)

        # Normalize sampled residuals to unit length
        unit_residuals = sampled_residuals / torch.norm(sampled_residuals, p=2, dim=-1, keepdim=True).clamp(min=1e-8)

        # Re-initialize decoder columns W_dec: shape (d_sae, d_in)
        if hasattr(sae, "W_dec") and sae.W_dec is not None:
            sae.W_dec.data[dead_indices] = unit_residuals

        # Re-initialize encoder columns W_enc: shape (d_in, d_sae)
        if hasattr(sae, "W_enc") and sae.W_enc is not None:
            # Scale by average norm of alive encoder weights
            alive_indices = torch.where(~dead_mask)[0]
            if len(alive_indices) > 0:
                avg_enc_norm = torch.norm(sae.W_enc.data[:, alive_indices], p=2, dim=0).mean()
            else:
                avg_enc_norm = 1.0 / (sae.d_in**0.5)

            sae.W_enc.data[:, dead_indices] = unit_residuals.t() * avg_enc_norm

        if hasattr(sae, "b_enc") and sae.b_enc is not None:
            sae.b_enc.data[dead_indices] = 0.0

        # Reset optimizer momentum/state for resampled parameters
        for param in sae.parameters():
            state = optimizer.state.get(param)
            if state is not None:
                for key in ["exp_avg", "exp_avg_sq"]:
                    if key in state:
                        if state[key].ndim == 2 and state[key].shape[1] == self.d_sae:
                            state[key][:, dead_indices] = 0.0
                        elif state[key].ndim == 1 and state[key].shape[0] == self.d_sae:
                            state[key][dead_indices] = 0.0

        # Reset firing counter for resampled features
        self.tokens_since_last_fired[dead_indices] = 0
        sae.normalize_decoder_columns()

        return num_dead
