"""
Feature Activation Analyzer.
Evaluates firing frequencies, max activation distributions, and discovers the highest-activating
text contexts and token positions for any SAE latent feature.
"""

from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
from ..models.base import BaseSAE
from ..extraction.hook_manager import ModelHookManager


class FeatureAnalyzer:
    """
    Scans corpora to collect top-activating passages and activation density for SAE features.
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
        self.hook_manager = ModelHookManager(model, layer_idx=layer_idx, hook_point=hook_point)

    @torch.no_grad()
    def find_top_activating_contexts(
        self,
        texts: List[str],
        feature_idx: int,
        top_k: int = 5,
        context_window: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Finds the top-K text contexts where feature_idx has the highest activation.
        """
        self.sae.eval()
        self.model.eval()

        records = []

        with self.hook_manager:
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(self.device)
                _ = self.model(**inputs)

                act = self.hook_manager.captured_activation
                if act is None:
                    continue

                latents = self.sae.encode(act)
                feature_acts = latents[0, :, feature_idx]

                max_val, max_pos = torch.max(feature_acts, dim=0)
                if max_val.item() > 0.0:
                    pos = max_pos.item()
                    tokens = inputs.input_ids[0]
                    start = max(0, pos - context_window)
                    end = min(len(tokens), pos + context_window + 1)

                    context_tokens = tokens[start:end]
                    target_token = tokens[pos]

                    records.append({
                        "max_activation": round(max_val.item(), 4),
                        "target_token": self.tokenizer.decode([target_token]).strip(),
                        "context_str": self.tokenizer.decode(context_tokens),
                        "full_text": text,
                    })

        records = sorted(records, key=lambda r: r["max_activation"], reverse=True)
        return records[:top_k]

    @torch.no_grad()
    def compute_feature_densities(self, texts: List[str]) -> torch.Tensor:
        """
        Computes the percentage of tokens where each feature fires (> 0) across texts.
        """
        total_tokens = 0
        active_counts = torch.zeros(self.sae.d_sae, device=self.device)

        with self.hook_manager:
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(self.device)
                _ = self.model(**inputs)
                act = self.hook_manager.captured_activation
                if act is None:
                    continue

                latents = self.sae.encode(act)
                active_counts += (latents[0] > 0).sum(dim=0)
                total_tokens += latents.shape[1]

        if total_tokens == 0:
            return active_counts

        return active_counts / total_tokens
