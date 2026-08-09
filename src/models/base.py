"""
Base Sparse Autoencoder abstract class with serialization, normalization, and evaluation utilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from pathlib import Path
from ..common.config import SAEConfig


class BaseSAE(nn.Module, ABC):
    """
    Abstract Base Class for Sparse Autoencoders.
    All subclasses must implement `encode`, `decode`, and `forward`.
    """

    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        self.d_in = cfg.d_in
        self.d_sae = cfg.d_sae
        self.normalize_decoder_weights = cfg.normalize_decoder_weights

        # Pre-decoder geometric bias (geometric median / mean of dataset activations)
        self.b_dec = nn.Parameter(torch.zeros(self.d_in))

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Maps input activations x (..., d_in) to sparse latent activations f (..., d_sae).
        """
        pass

    @abstractmethod
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs input activations x_hat (..., d_in) from sparse latents f (..., d_sae).
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass.
        Returns:
            x_reconstructed: (..., d_in)
            latents: (..., d_sae)
            info_dict: dict containing sparsity, intermediate terms, or auxiliary loss tensors.
        """
        pass

    @torch.no_grad()
    def normalize_decoder_columns(self) -> None:
        """
        Enforces unit norm on decoder weight columns: ||W_dec[:, i]||_2 = 1.
        Essential constraint to prevent feature shrinkage / norm cheating in SAEs.
        """
        if hasattr(self, "W_dec") and self.W_dec is not None:
            # W_dec shape: (d_sae, d_in) -> columns are features of length d_in
            norms = torch.norm(self.W_dec.data, p=2, dim=-1, keepdim=True)
            # Avoid division by zero
            norms = torch.clamp(norms, min=1e-8)
            self.W_dec.data.div_(norms)

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder(self) -> None:
        """
        Projects out the component of the decoder gradient that is parallel to the decoder weights:
        grad(W_dec) = grad(W_dec) - (grad(W_dec) . W_dec_norm) * W_dec_norm
        Ensures gradient updates solely rotate the feature directions on the unit hypersphere.
        """
        if hasattr(self, "W_dec") and self.W_dec.grad is not None:
            w_dec = self.W_dec.data
            w_norm = torch.norm(w_dec, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            w_unit = w_dec / w_norm
            proj = torch.sum(self.W_dec.grad * w_unit, dim=-1, keepdim=True) * w_unit
            self.W_dec.grad.sub_(proj)

    def save_pretrained(self, save_path: str | Path) -> None:
        """
        Serializes weights and architecture configuration to disk.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cfg": self.cfg,
                "state_dict": self.state_dict(),
                "architecture": self.cfg.architecture,
            },
            save_path,
        )

    @classmethod
    def from_pretrained(cls, load_path: str | Path, device: str = "cpu") -> "BaseSAE":
        """
        Loads a serialized SAE checkpoint into its corresponding concrete architecture.
        """
        from . import SAE_REGISTRY

        try:
            checkpoint = torch.load(load_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(load_path, map_location=device)

        cfg = checkpoint["cfg"]
        arch = getattr(cfg, "architecture", "topk").lower()

        target_cls = SAE_REGISTRY.get(arch, cls)
        if target_cls is BaseSAE:
            target_cls = SAE_REGISTRY["topk"]

        model = target_cls(cfg)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model
