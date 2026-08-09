"""
interp-circuit-engine: Mechanistic Interpretability & Latent Activation Steering Suite.
"""

import os
# Ensure pure PyTorch backend without TensorFlow/protobuf conflicts
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

__version__ = "0.1.0"
