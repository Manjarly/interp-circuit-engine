"""
Unit tests for forward hook extraction and activation buffering.
"""

import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.extraction.hook_manager import ModelHookManager
from src.extraction.text_sampler import TextSampler, BUILTIN_CORPUS
from src.extraction.buffer import ActivationBuffer
from src.common.config import ExtractionConfig


@pytest.fixture(scope="module")
def gpt2_setup():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    return model, tokenizer


def test_hook_manager_capture(gpt2_setup):
    model, tokenizer = gpt2_setup
    hook_mgr = ModelHookManager(model, layer_idx=6, hook_point="resid_post")

    inputs = tokenizer("Testing hook extraction mechanism.", return_tensors="pt")
    with hook_mgr:
        _ = model(**inputs)
        act = hook_mgr.captured_activation

    assert act is not None
    assert act.ndim == 3 # (batch, seq_len, d_in)
    assert act.shape[-1] == model.config.hidden_size


def test_text_sampler_batches(gpt2_setup):
    _, tokenizer = gpt2_setup
    sampler = TextSampler(tokenizer=tokenizer, seq_len=32, batch_size=4)
    stream = sampler.stream_token_batches(device=torch.device("cpu"))

    batch = next(stream)
    assert batch.shape == (4, 32)
    assert batch.dtype == torch.long


def test_activation_buffer_sample(gpt2_setup):
    model, tokenizer = gpt2_setup
    cfg = ExtractionConfig(
        model_name="gpt2",
        hook_layer=6,
        seq_len=32,
        buffer_size=128,
        batch_size=4,
    )
    buffer = ActivationBuffer(model, tokenizer, cfg, device=torch.device("cpu"))
    batch = buffer.sample_batch(batch_size=16)

    assert batch.shape == (16, model.config.hidden_size)
