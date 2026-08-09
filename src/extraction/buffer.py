"""
Shuffled Activation Buffer.
Streams residual activations from a transformer model, flattens and shuffles tokens to break
intra-document auto-correlations, and yields minibatches for SAE optimization.
"""

from typing import Iterator, Optional, Any
import torch
import torch.nn as nn
from .hook_manager import ModelHookManager
from .text_sampler import TextSampler
from ..common.config import ExtractionConfig


class ActivationBuffer:
    """
    In-memory ring buffer for streaming, shuffling, and sampling model activations.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        cfg: ExtractionConfig,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device

        self.buffer_size = cfg.buffer_size
        self.hook_manager = ModelHookManager(
            model=self.model, layer_idx=cfg.hook_layer, hook_point=cfg.hook_type
        )
        self.text_sampler = TextSampler(
            tokenizer=self.tokenizer,
            dataset_name=cfg.dataset_name,
            dataset_config=cfg.dataset_config,
            seq_len=cfg.seq_len,
            batch_size=cfg.batch_size,
        )

        self.token_stream = self.text_sampler.stream_token_batches(device=self.device)
        self._buffer: Optional[torch.Tensor] = None
        self._ptr = 0

    @torch.no_grad()
    def fill_buffer(self) -> None:
        """
        Runs forward passes through the model and populates the activation buffer.
        """
        activations_list = []
        tokens_collected = 0

        self.hook_manager.attach_hook()
        self.model.eval()

        while tokens_collected < self.buffer_size:
            input_ids = next(self.token_stream)
            _ = self.model(input_ids)

            # Captured shape: (batch_size, seq_len, d_in)
            act = self.hook_manager.captured_activation
            if act is None:
                raise RuntimeError("Hook manager failed to capture activations.")

            # Flatten to (batch_size * seq_len, d_in)
            flat_act = act.view(-1, act.shape[-1])
            activations_list.append(flat_act)
            tokens_collected += flat_act.shape[0]

        self.hook_manager.remove_hook()

        # Concatenate and shuffle tokens
        all_acts = torch.cat(activations_list, dim=0)[: self.buffer_size]
        perm = torch.randperm(all_acts.shape[0], device=all_acts.device)
        self._buffer = all_acts[perm]
        self._ptr = 0

    def sample_batch(self, batch_size: int) -> torch.Tensor:
        """
        Samples a batch of activations. Refills the buffer when depleted.
        """
        if self._buffer is None or (self._ptr + batch_size > self._buffer.shape[0]):
            self.fill_buffer()

        batch = self._buffer[self._ptr : self._ptr + batch_size]
        self._ptr += batch_size
        return batch

    @torch.no_grad()
    def estimate_geometric_median_or_mean(self, num_samples: int = 10000) -> torch.Tensor:
        """
        Computes the empirical mean activation vector across samples for decoder bias initialization.
        """
        if self._buffer is None:
            self.fill_buffer()
        samples = self._buffer[: min(num_samples, self._buffer.shape[0])]
        return torch.mean(samples, dim=0)
