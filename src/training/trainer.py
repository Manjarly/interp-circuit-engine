"""
Sparse Autoencoder Trainer.
Manages optimization, warmups, cosine learning rate decay, gradient clipping,
unit-norm projection, and dead feature tracking.
"""

from typing import Dict, Any, Optional, List
import math
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn

from ..models.base import BaseSAE
from ..extraction.buffer import ActivationBuffer
from .loss import compute_sae_loss
from .dead_latent_tracker import DeadLatentTracker
from ..common.config import ExperimentConfig


class SAETrainer:
    """
    Main training controller for Sparse Autoencoders.
    """

    def __init__(
        self,
        sae: BaseSAE,
        buffer: ActivationBuffer,
        cfg: ExperimentConfig,
        device: torch.device,
    ):
        self.sae = sae.to(device)
        self.buffer = buffer
        self.cfg = cfg
        self.device = device
        self.t_cfg = cfg.training

        # Initialize decoder bias to mean activation
        print("Estimating activation mean for geometric centering...")
        mean_act = self.buffer.estimate_geometric_median_or_mean()
        self.sae.b_dec.data = mean_act.clone().to(device)

        # Optimizer
        self.optimizer = Adam(
            self.sae.parameters(),
            lr=self.t_cfg.lr,
            weight_decay=self.t_cfg.weight_decay,
            betas=(0.9, 0.999),
        )

        # Learning rate schedule with linear warmup + cosine decay
        total_steps = self.t_cfg.total_training_tokens // self.t_cfg.batch_size
        self.total_steps = max(total_steps, 1)

        def lr_lambda(step: int) -> float:
            if step < self.t_cfg.lr_warmup_steps:
                return float(step) / float(max(1, self.t_cfg.lr_warmup_steps))
            progress = float(step - self.t_cfg.lr_warmup_steps) / float(
                max(1, self.total_steps - self.t_cfg.lr_warmup_steps)
            )
            return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        # Dead latent tracker
        self.dead_tracker = DeadLatentTracker(
            d_sae=self.sae.d_sae,
            dead_threshold_tokens=self.t_cfg.dead_neuron_threshold,
            device=self.device,
        )

        self.history: List[Dict[str, float]] = []

    def train_step(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Executes a single optimization step on minibatch x.
        """
        self.sae.train()
        self.optimizer.zero_grad()

        x_hat, f, info = self.sae(x)
        total_loss, metrics = compute_sae_loss(x, x_hat, f, info)

        total_loss.backward()

        # Remove parallel gradient component if configured
        if self.sae.normalize_decoder_weights:
            self.sae.remove_gradient_parallel_to_decoder()

        if self.t_cfg.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.sae.parameters(), self.t_cfg.clip_grad_norm)

        self.optimizer.step()
        self.scheduler.step()

        # Enforce unit norm constraint on decoder columns
        if self.sae.normalize_decoder_weights:
            self.sae.normalize_decoder_columns()

        # Update dead neuron activity
        self.dead_tracker.update(f)

        metrics["loss/lr"] = self.scheduler.get_last_lr()[0]
        metrics["metrics/dead_latents"] = self.dead_tracker.get_dead_count()

        return metrics

    def train(self, max_steps: Optional[int] = None) -> List[Dict[str, float]]:
        """
        Runs the full training loop.
        """
        steps = max_steps or self.total_steps
        save_dir = Path(self.t_cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[bold green]Starting SAE Training[/bold green] | Steps: {steps} | Batch size: {self.t_cfg.batch_size}")

        recent_residuals = []

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[cyan]Training SAE...", total=steps)

            for step in range(1, steps + 1):
                x = self.buffer.sample_batch(self.t_cfg.batch_size).to(self.device)
                metrics = self.train_step(x)
                metrics["step"] = step

                # Track residuals for potential dead latent resampling
                if self.t_cfg.resample_dead_latents and (step % self.t_cfg.resample_frequency_steps == 0):
                    with torch.no_grad():
                        x_hat, _, _ = self.sae(x)
                        residuals = x - x_hat
                        num_resampled = self.dead_tracker.resample_dead_latents(
                            sae=self.sae,
                            optimizer=self.optimizer,
                            residuals=residuals,
                        )
                        if num_resampled > 0:
                            print(f"\n[Step {step}] Resampled {num_resampled} dead latents.")

                if step % self.t_cfg.eval_frequency_steps == 0 or step == 1:
                    self.history.append(metrics)
                    progress.update(
                        task,
                        advance=self.t_cfg.eval_frequency_steps if step > 1 else 1,
                        description=(
                            f"[cyan]Step {step}/{steps} | "
                            f"NMSE: {metrics['loss/normalized_mse']:.4f} | "
                            f"L0: {metrics['metrics/l0']:.1f} | "
                            f"Dead: {metrics['metrics/dead_latents']}"
                        ),
                    )
                else:
                    progress.update(task, advance=1)

                if step % self.t_cfg.checkpoint_frequency_steps == 0 or step == steps:
                    ckpt_path = save_dir / f"sae_step_{step}.pt"
                    self.sae.save_pretrained(ckpt_path)

        # Save final model and config
        final_path = save_dir / "sae_final.pt"
        self.sae.save_pretrained(final_path)
        self.cfg.to_yaml(save_dir / "config.yaml")
        print(f"\n[bold green]Training complete! Model saved to {final_path}[/bold green]\n")

        return self.history
