"""
Hierarchical configuration schema for Sparse Autoencoder training, extraction, and steering.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import yaml
from pathlib import Path


@dataclass
class SAEConfig:
    architecture: str = "topk"  # 'topk', 'gated', 'standard', 'jumprelu'
    d_in: int = 768              # Transformer activation dimension (e.g. 768 for GPT-2 Small, 2304 for Gemma-2-2b)
    d_sae: int = 3072            # Number of latent features (expansion factor, e.g. 4x or 8x d_in)
    k: int = 32                  # Top-K sparsity (exact active features per token for TopK SAE)
    l1_coeff: float = 1e-3       # L1 sparsity penalty for standard SAE / Gated SAE
    aux_loss_coeff: float = 1e-2 # Auxiliary loss for dead latents
    jumprelu_threshold: float = 0.1 # Threshold for JumpReLU SAE
    tied_bias: bool = True       # Whether b_dec is subtracted before encoding
    normalize_decoder_weights: bool = True # Enforce ||W_dec[:, i]||_2 = 1


@dataclass
class ExtractionConfig:
    model_name: str = "gpt2"     # HuggingFace model ID (e.g. "gpt2", "google/gemma-2-2b", "EleutherAI/pythia-70m")
    hook_layer: int = 6          # Layer index to extract from
    hook_type: str = "resid_post"# 'resid_post', 'mlp_out', 'attn_out'
    seq_len: int = 128           # Token sequence length for extraction
    buffer_size: int = 65536     # Number of activation vectors in the shuffled memory buffer
    batch_size: int = 16         # Batch size of text documents during extraction
    dataset_name: str = "wikitext" # Dataset or fallback text corpus
    dataset_config: str = "wikitext-2-raw-v1"


@dataclass
class TrainingConfig:
    total_training_tokens: int = 5_000_000 # Total activation tokens to train on
    batch_size: int = 1024       # Minibatch size for SAE training steps
    lr: float = 5e-4             # Peak learning rate
    lr_warmup_steps: int = 500   # Linear warmup steps
    lr_decay_steps: int = 4000   # Cosine decay steps
    weight_decay: float = 0.0
    clip_grad_norm: float = 1.0  # Max gradient norm
    dead_neuron_threshold: int = 50_000 # Tokens since last firing to declare dead
    resample_dead_latents: bool = True  # Resample dead neurons
    resample_frequency_steps: int = 1000 # Check for dead neurons every N steps
    eval_frequency_steps: int = 200     # Evaluation / metric logging frequency
    checkpoint_frequency_steps: int = 1000
    save_dir: str = "checkpoints/gpt2_layer6_topk"
    device: str = "auto"
    seed: int = 42


@dataclass
class SteeringConfig:
    target_feature_idx: int = 0
    steering_coefficient: float = 5.0 # Alpha multiplier (positive for amplification, negative for suppression)
    intervention_type: str = "addition" # 'addition', 'clamping', 'ablation'
    clamp_value: float = 10.0
    prompt: str = "The future of artificial intelligence is"
    max_new_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class ExperimentConfig:
    sae: SAEConfig = field(default_factory=SAEConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ExperimentConfig":
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}

        sae = SAEConfig(**data.get("sae", {}))
        extraction = ExtractionConfig(**data.get("extraction", {}))
        training = TrainingConfig(**data.get("training", {}))
        steering = SteeringConfig(**data.get("steering", {}))

        return cls(sae=sae, extraction=extraction, training=training, steering=steering)

    def to_yaml(self, yaml_path: str | Path) -> None:
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
