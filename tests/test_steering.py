"""
Unit tests for causal activation steering and intervention hooks.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.topk_sae import TopKSAE
from src.common.config import SAEConfig
from src.steering.hooks import SteeringHookEngine
from src.steering.steer_generator import SteeredGenerator


@pytest.fixture(scope="module")
def gpt2_and_sae():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    cfg = SAEConfig(d_in=model.config.hidden_size, d_sae=128, k=8)
    sae = TopKSAE(cfg)
    return model, tokenizer, sae


def test_steering_hook_modifies_activations(gpt2_and_sae):
    model, tokenizer, sae = gpt2_and_sae
    engine = SteeringHookEngine(model, sae, layer_idx=6)

    inputs = tokenizer("Testing concept steering.", return_tensors="pt")

    # Forward without intervention
    out_baseline = model(**inputs).logits

    # Forward with strong concept injection
    engine.set_steering_feature(feature_idx=0, alpha=20.0, intervention_type="addition")
    with engine:
        out_steered = model(**inputs).logits

    engine.clear_interventions()

    # Logits should strictly differ under intervention
    diff = torch.abs(out_steered - out_baseline).mean().item()
    assert diff > 0.1, f"Expected noticeable logit divergence from steering, got diff={diff}"


def test_steered_generator_comparison(gpt2_and_sae):
    model, tokenizer, sae = gpt2_and_sae
    generator = SteeredGenerator(model, tokenizer, sae, layer_idx=6, device=torch.device("cpu"))

    res = generator.generate_comparison(
        prompt="The universe",
        feature_idx=0,
        alpha=5.0,
        max_new_tokens=10,
    )

    assert "baseline_text" in res
    assert "steered_text" in res
    assert len(res["baseline_text"]) > len("The universe")
    assert len(res["steered_text"]) > len("The universe")
