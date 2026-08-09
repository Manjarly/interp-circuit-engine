"""
Direct Logit Attribution (DLA) for Sparse Autoencoders.
Projects decoder feature vectors d_i directly onto the language model's unembedding matrix (W_U)
to identify the exact vocabulary tokens promoted or suppressed by each latent feature.
"""

from typing import List, Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
from ..models.base import BaseSAE


class DirectLogitAttributor:
    """
    Computes direct projections of SAE features into token vocabulary space.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        sae: BaseSAE,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sae = sae.to(device)
        self.device = device

        self.W_U = self._extract_unembedding_matrix()

    def _extract_unembedding_matrix(self) -> torch.Tensor:
        """
        Extracts the language model's unembedding weight matrix W_U: shape (d_in, vocab_size).
        """
        if hasattr(self.model, "lm_head") and hasattr(self.model.lm_head, "weight"):
            # lm_head.weight shape: (vocab_size, d_in) -> transpose to (d_in, vocab_size)
            return self.model.lm_head.weight.data.t().to(self.device)
        elif hasattr(self.model, "get_output_embeddings"):
            emb = self.model.get_output_embeddings()
            if emb is not None and hasattr(emb, "weight"):
                return emb.weight.data.t().to(self.device)
        raise ValueError(f"Could not extract unembedding matrix from model {type(self.model)}")

    @torch.no_grad()
    def attribute_feature(
        self,
        feature_idx: int,
        top_k: int = 10,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Computes the top positive and negative vocabulary tokens for feature_idx.
        """
        d_i = self.sae.W_dec.data[feature_idx] # (d_in,)

        logits = torch.matmul(d_i.unsqueeze(0), self.W_U).squeeze(0)

        pos_values, pos_indices = torch.topk(logits, k=top_k, largest=True)
        neg_values, neg_indices = torch.topk(logits, k=top_k, largest=False)

        top_pos = [
            (self.tokenizer.decode([idx.item()]).strip(), round(val.item(), 4))
            for idx, val in zip(pos_indices, pos_values)
        ]
        top_neg = [
            (self.tokenizer.decode([idx.item()]).strip(), round(val.item(), 4))
            for idx, val in zip(neg_indices, neg_values)
        ]

        return {
            "top_positive": top_pos,
            "top_negative": top_neg,
        }

    @torch.no_grad()
    def attribute_all_features(self, top_k: int = 5) -> Dict[int, Dict[str, List[Tuple[str, float]]]]:
        """
        Batch attributes all SAE features.
        """
        results = {}
        for idx in range(self.sae.d_sae):
            results[idx] = self.attribute_feature(idx, top_k=top_k)
        return results
