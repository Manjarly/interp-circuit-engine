"""
Steered Text Generator.
Compares unsteered baseline generations against real-time concept-steered generations.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from ..models.base import BaseSAE
from .hooks import SteeringHookEngine


class SteeredGenerator:
    """
    Generates text with dynamic SAE concept interventions.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        sae: BaseSAE,
        layer_idx: int = 6,
        hook_point: str = "resid_post",
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sae = sae.to(device)
        self.device = device
        self.hook_engine = SteeringHookEngine(model, sae, layer_idx=layer_idx, hook_point=hook_point)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 40,
        temperature: float = 0.7,
        top_p: float = 0.9,
        feature_idx: Optional[int] = None,
        alpha: float = 0.0,
        intervention_type: str = "addition",
    ) -> Dict[str, str]:
        """
        Runs generation with optional concept steering.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        if feature_idx is not None and abs(alpha) > 1e-5:
            self.hook_engine.set_steering_feature(feature_idx, alpha, intervention_type=intervention_type)
            with self.hook_engine:
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=max(temperature, 1e-2),
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            self.hook_engine.clear_interventions()
        else:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 1e-2),
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        prompt_len = len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
        new_text = full_text[prompt_len:]

        return {
            "prompt": prompt,
            "output_text": full_text,
            "new_tokens": new_text,
        }

    def generate_comparison(
        self,
        prompt: str,
        feature_idx: int,
        alpha: float = 5.0,
        max_new_tokens: int = 40,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Generates both baseline (alpha=0.0) and steered (alpha=alpha) text for direct comparison.
        """
        # Baseline
        baseline = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            alpha=0.0,
        )

        # Steered
        steered = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            feature_idx=feature_idx,
            alpha=alpha,
        )

        return {
            "prompt": prompt,
            "feature_idx": feature_idx,
            "alpha": alpha,
            "baseline_text": baseline["output_text"],
            "steered_text": steered["output_text"],
        }
