"""
Transformer Layer Hook Manager.
Supports PyTorch forward hooks for residual stream, MLP outputs, and attention outputs
across modern HuggingFace architectures (Qwen, Kimi/Moonshot, LLaMA, Gemma, Mistral, Pythia, GPT-2).
"""

from typing import Optional, Dict, Any, List, Callable
import torch
import torch.nn as nn


class ModelHookManager:
    """
    Manages attaching, storing, and removing PyTorch forward hooks on transformer models.
    """

    def __init__(self, model: nn.Module, layer_idx: int = 6, hook_point: str = "resid_post"):
        self.model = model
        self.layer_idx = layer_idx
        self.hook_point = hook_point
        self._hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.captured_activation: Optional[torch.Tensor] = None

    def _locate_target_module(self) -> nn.Module:
        """
        Locates the target submodule based on model architecture (Qwen, Kimi, Llama, Gemma, etc.).
        """
        layers = None

        # Standard LLaMA / Qwen / Kimi / Mistral / DeepSeek
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        # GPT-2 / GPT-Neo
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            layers = self.model.transformer.h
        # GPT-J / Falcon
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "layers"):
            layers = self.model.transformer.layers
        # Root layers (e.g. some MoE architectures)
        elif hasattr(self.model, "layers"):
            layers = self.model.layers
        # Fallback recursive search
        else:
            for name, module in self.model.named_modules():
                if f"layers.{self.layer_idx}" in name or f"h.{self.layer_idx}" in name:
                    return module
            raise ValueError(f"Could not automatically locate layer {self.layer_idx} in model {type(self.model)}")

        if layers is not None:
            if self.layer_idx >= len(layers):
                raise IndexError(f"layer_idx {self.layer_idx} out of range for model with {len(layers)} layers.")
            layer_module = layers[self.layer_idx]
        else:
            raise ValueError(f"Could not locate layers list in model {type(self.model)}")

        if self.hook_point == "resid_post":
            return layer_module
        elif self.hook_point == "mlp_out":
            if hasattr(layer_module, "mlp"):
                return layer_module.mlp
            elif hasattr(layer_module, "feed_forward"):
                return layer_module.feed_forward
            elif hasattr(layer_module, "block_sparse_moe"):
                return layer_module.block_sparse_moe
        elif self.hook_point == "attn_out":
            if hasattr(layer_module, "attn"):
                return layer_module.attn
            elif hasattr(layer_module, "self_attn"):
                return layer_module.self_attn

        return layer_module

    def attach_hook(self) -> None:
        """
        Attaches the extraction hook.
        """
        if self._hook_handle is not None:
            return  # Already attached

        target_module = self._locate_target_module()

        def hook_fn(module: nn.Module, input: Any, output: Any):
            # Output can be a Tensor or a Tuple (hidden_states, ...)
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output
            self.captured_activation = act.detach()

        self._hook_handle = target_module.register_forward_hook(hook_fn)

    def remove_hook(self) -> None:
        """
        Removes the active hook.
        """
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self.captured_activation = None

    def __enter__(self):
        self.attach_hook()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hook()
