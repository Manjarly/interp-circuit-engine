"""
Unit tests for training metrics, loss computation, and dead latent tracker.
"""

import pytest
import torch
from src.training.loss import compute_sae_loss
from src.training.dead_latent_tracker import DeadLatentTracker
from src.models.topk_sae import TopKSAE
from src.common.config import SAEConfig


def test_compute_sae_loss_metrics():
    batch_size, d_in, d_sae = 16, 64, 256
    x = torch.randn(batch_size, d_in)
    x_hat = x + 0.1 * torch.randn_like(x)
    f = torch.zeros(batch_size, d_sae)
    f[:, :10] = 1.0 # 10 active features

    info = {"aux_loss": torch.tensor(0.05)}
    total_loss, metrics = compute_sae_loss(x, x_hat, f, info)

    assert "loss/total" in metrics
    assert "loss/normalized_mse" in metrics
    assert "metrics/explained_variance" in metrics
    assert metrics["metrics/l0"] == 10.0
    assert metrics["metrics/explained_variance"] > 0.0


def test_dead_latent_tracker_and_resampling():
    d_sae = 32
    d_in = 16
    cfg = SAEConfig(d_in=d_in, d_sae=d_sae, k=4)
    sae = TopKSAE(cfg)
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

    tracker = DeadLatentTracker(d_sae=d_sae, dead_threshold_tokens=50, device="cpu")

    # Simulate batch where only first 10 latents fire
    f = torch.zeros(60, d_sae)
    f[:, :10] = 2.0
    tracker.update(f)

    # Features 10..31 should be dead
    dead_count = tracker.get_dead_count()
    assert dead_count == 22

    # Resample dead latents
    residuals = torch.randn(60, d_in)
    num_resampled = tracker.resample_dead_latents(sae, optimizer, residuals)
    assert num_resampled == 22
    assert tracker.get_dead_count() == 0
