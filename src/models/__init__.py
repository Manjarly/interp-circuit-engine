"""
Sparse Autoencoder model registry and factory functions.
"""

from typing import Dict, Type
from .base import BaseSAE
from .topk_sae import TopKSAE
from .gated_sae import GatedSAE
from .standard_sae import StandardSAE
from .jumprelu_sae import JumpReLUSAE
from ..common.config import SAEConfig

SAE_REGISTRY: Dict[str, Type[BaseSAE]] = {
    "topk": TopKSAE,
    "gated": GatedSAE,
    "standard": StandardSAE,
    "jumprelu": JumpReLUSAE,
}


def create_sae(cfg: SAEConfig) -> BaseSAE:
    """
    Factory function to instantiate an SAE based on configuration.
    """
    arch = cfg.architecture.lower()
    if arch not in SAE_REGISTRY:
        raise ValueError(f"Unknown SAE architecture: '{arch}'. Choose from {list(SAE_REGISTRY.keys())}")
    return SAE_REGISTRY[arch](cfg)


__all__ = [
    "BaseSAE",
    "TopKSAE",
    "GatedSAE",
    "StandardSAE",
    "JumpReLUSAE",
    "create_sae",
    "SAE_REGISTRY",
]
