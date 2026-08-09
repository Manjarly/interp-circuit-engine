"""
Unit tests for Sparse Autoencoder architectures.
"""

import pytest
import torch
from src.common.config import SAEConfig
from src.models.topk_sae import TopKSAE
from src.models.gated_sae import GatedSAE
from src.models.standard_sae import StandardSAE
from src.models.jumprelu_sae import JumpReLUSAE
from src.models import create_sae, BaseSAE


@pytest.fixture
def sae_config():
    return SAEConfig(
        architecture="topk",
        d_in=64,
        d_sae=256,
        k=8,
        l1_coeff=1e-3,
        aux_loss_coeff=1e-2,
    )


def test_topk_sae_forward_and_shapes(sae_config):
    sae = TopKSAE(sae_config)
    batch_size = 16
    x = torch.randn(batch_size, sae_config.d_in)

    x_hat, f, info = sae(x)

    assert x_hat.shape == (batch_size, sae_config.d_in)
    assert f.shape == (batch_size, sae_config.d_sae)
    assert "l0" in info

    # Verify exact Top-K sparsity per token
    non_zeros = (f > 0).sum(dim=-1)
    assert torch.all(non_zeros == sae_config.k), f"Expected exact k={sae_config.k} non-zeros, got {non_zeros}"


def test_topk_sae_gradient_flow(sae_config):
    sae = TopKSAE(sae_config)
    x = torch.randn(8, sae_config.d_in, requires_grad=True)

    x_hat, f, info = sae(x)
    loss = torch.mean((x - x_hat) ** 2) + info["aux_loss"]
    loss.backward()

    assert sae.W_enc.grad is not None
    assert sae.W_dec.grad is not None
    assert sae.b_enc.grad is not None
    assert not torch.isnan(sae.W_enc.grad).any()


def test_decoder_unit_norm_constraint(sae_config):
    sae = TopKSAE(sae_config)
    # Disrupt norms artificially
    sae.W_dec.data.mul_(5.0)

    sae.normalize_decoder_columns()
    norms = torch.norm(sae.W_dec.data, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_gated_sae_forward():
    cfg = SAEConfig(architecture="gated", d_in=64, d_sae=256, l1_coeff=1e-3)
    sae = GatedSAE(cfg)
    x = torch.randn(12, cfg.d_in)

    x_hat, f, info = sae(x)

    assert x_hat.shape == (12, cfg.d_in)
    assert f.shape == (12, cfg.d_sae)
    assert "aux_loss" in info


def test_standard_sae_forward():
    cfg = SAEConfig(architecture="standard", d_in=64, d_sae=256, l1_coeff=1e-3)
    sae = StandardSAE(cfg)
    x = torch.randn(10, cfg.d_in)

    x_hat, f, info = sae(x)
    assert x_hat.shape == (10, cfg.d_in)
    assert f.shape == (10, cfg.d_sae)


def test_jumprelu_sae_forward():
    cfg = SAEConfig(architecture="jumprelu", d_in=64, d_sae=256, jumprelu_threshold=0.1)
    sae = JumpReLUSAE(cfg)
    x = torch.randn(10, cfg.d_in)

    x_hat, f, info = sae(x)
    assert x_hat.shape == (10, cfg.d_in)
    assert f.shape == (10, cfg.d_sae)


def test_sae_save_and_load(tmp_path, sae_config):
    sae = TopKSAE(sae_config)
    save_path = tmp_path / "test_sae.pt"

    sae.save_pretrained(save_path)
    assert save_path.exists()

    loaded = BaseSAE.from_pretrained(save_path)
    assert loaded.d_in == sae.d_in
    assert loaded.d_sae == sae.d_sae
    assert torch.allclose(sae.W_dec.data, loaded.W_dec.data)
