"""
Hardware device and precision resolver supporting Apple Silicon (MPS), NVIDIA CUDA, and CPU.
"""

from typing import Union
import torch


def get_optimal_device(preferred_device: str = "auto") -> torch.device:
    """
    Resolves the most optimal compute device.
    Prioritizes CUDA -> MPS (Apple Silicon GPU) -> CPU.
    """
    if preferred_device != "auto":
        return torch.device(preferred_device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def resolve_dtype(dtype_str: str = "float32", device: Union[str, torch.device] = "cpu") -> torch.dtype:
    """
    Safely resolves torch dtype, handling device-specific constraints (e.g. bfloat16 on MPS).
    """
    dev = torch.device(device) if isinstance(device, str) else device

    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    dtype = mapping.get(dtype_str.lower(), torch.float32)

    # Some older MPS builds do not support all ops in bfloat16
    if dev.type == "mps" and dtype == torch.bfloat16:
        # Check if basic operations work or fallback to float32
        try:
            _ = torch.ones((2, 2), dtype=torch.bfloat16, device=dev)
        except Exception:
            dtype = torch.float32

    return dtype
