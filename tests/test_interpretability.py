"""
Unit tests for Direct Logit Attribution and feature analysis.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.topk_sae import TopKSAE
from src.common.config import SAEConfig
from src.interpretability.direct_logit_attribution import DirectLogitAttributor


@pytest.fixture(scope="module")
def gpt2_and_sae():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    cfg = SAEConfig(d_in=model.config.hidden_size, d_sae=128, k=8)
    sae = TopKSAE(cfg)
    return model, tokenizer, sae


def test_direct_logit_attribution(gpt2_and_sae):
    model, tokenizer, sae = gpt2_and_sae
    attributor = DirectLogitAttributor(model, tokenizer, sae, device=torch.device("cpu"))

    res = attributor.attribute_feature(feature_idx=0, top_k=5)
    assert "top_positive" in res
    assert "top_negative" in res
    assert len(res["top_positive"]) == 5
    assert len(res["top_negative"]) == 5
    assert isinstance(res["top_positive"][0][0], str)
    assert isinstance(res["top_positive"][0][1], float)
