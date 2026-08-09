"""
Causal Activation Steering Hooks.
Attaches intervention hooks to the transformer residual stream during autoregressive decoding,
enabling real-time concept amplification, suppression, clamping, and multi-feature steering.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from ..models.base import BaseSAE
from ..extraction.hook_manager import ModelHookManager


class SteeringHookEngine:
    """
    Manages dynamic activation patching and concept steering on language models.
    """

    def __init__(
        self,
        model: nn.Module,
        sae: BaseSAE,
        layer_idx: int = 6,
        hook_point: str = "resid_post",
    ):
        self.model = model
        self.sae = sae
        self.layer_idx = layer_idx
        self.hook_point = hook_point
        self.hook_manager = ModelHookManager(model, layer_idx=layer_idx, hook_point=hook_point)

        self._active_interventions: Dict[int, float] = {} # {feature_idx: alpha}
        self.intervention_type: str = "addition" # 'addition', 'clamping', 'ablation'
        self.hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def set_steering_feature(self, feature_idx: int, alpha: float, intervention_type: str = "addition") -> None:
        """
        Sets a single feature steering intervention.
        """
        self._active_interventions = {feature_idx: alpha}
        self.intervention_type = intervention_type

    def set_multi_steering(self, feature_alphas: Dict[int, float], intervention_type: str = "addition") -> None:
        """
        Sets multi-feature steering: e.g. {42: +5.0, 108: -3.0}
        """
        self._active_interventions = feature_alphas.copy()
        self.intervention_type = intervention_type

    def clear_interventions(self) -> None:
        """
        Clears all active interventions.
        """
        self._active_interventions.clear()

    def _intervention_hook(self, module: nn.Module, input: Any, output: Any):
        if not self._active_interventions:
            return output

        is_tuple = isinstance(output, tuple)
        act = output[0] if is_tuple else output

        # act shape: (batch_size, seq_len, d_in)
        steered_act = act.clone()

        for feature_idx, alpha in self._active_interventions.items():
            if feature_idx < 0 or feature_idx >= self.sae.d_sae:
                continue

            # Decoder direction for target concept: (d_in,)
            d_i = self.sae.W_dec.data[feature_idx].to(act.device)

            if self.intervention_type == "addition":
                # Add scaled direction: x' = x + alpha * d_i
                steered_act = steered_act + (alpha * d_i)

            elif self.intervention_type == "ablation":
                # Project out direction: x' = x - (x . d_i) * d_i
                d_unit = d_i / torch.norm(d_i, p=2).clamp(min=1e-8)
                proj = torch.sum(steered_act * d_unit, dim=-1, keepdim=True) * d_unit
                steered_act = steered_act - proj

            elif self.intervention_type == "clamping":
                # Encode, clamp latent to alpha, decode
                latents = self.sae.encode(steered_act)
                latents[:, :, feature_idx] = alpha
                recon = self.sae.decode(latents)
                steered_act = recon

        if is_tuple:
            return (steered_act,) + output[1:]
        return steered_act

    def attach(self) -> None:
        """
        Registers the steering forward hook on the model.
        """
        if self.hook_handle is not None:
            return

        target_module = self.hook_manager._locate_target_module()
        self.hook_handle = target_module.register_forward_hook(self._intervention_hook)

    def detach(self) -> None:
        """
        Removes the steering hook.
        """
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.detach()
