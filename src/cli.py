"""
Unified Command Line Interface for interp-circuit-engine.
Supports training SAEs, analyzing feature semantics, steering generation, and launching the Web UI.
"""

from typing import Optional
from pathlib import Path
import sys
import typer
from rich.console import Console
from rich.table import Table
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common.config import ExperimentConfig
from .common.device import get_optimal_device
from .models import create_sae, BaseSAE
from .extraction.buffer import ActivationBuffer
from .training.trainer import SAETrainer
from .interpretability.direct_logit_attribution import DirectLogitAttributor
from .interpretability.feature_analyzer import FeatureAnalyzer
from .interpretability.auto_labeler import FeatureAutoLabeler
from .steering.steer_generator import SteeredGenerator

app = typer.Typer(help="interp-circuit-engine: Mechanistic Interpretability & Concept Steering Suite")
console = Console()


@app.command()
def train(
    config_path: str = typer.Option("configs/gpt2_small_layer6.yaml", "--config", "-c", help="Path to YAML configuration file"),
    steps: Optional[int] = typer.Option(None, "--steps", "-s", help="Override total training steps"),
):
    """
    Trains a Sparse Autoencoder on the residual stream of a target language model.
    """
    console.print(f"[bold blue]Loading configuration from {config_path}...[/bold blue]")
    cfg = ExperimentConfig.from_yaml(config_path)

    device = get_optimal_device(cfg.training.device)
    console.print(f"[bold green]Resolved device: {device}[/bold green]")

    # Load host model & tokenizer
    console.print(f"Loading host language model: [cyan]{cfg.extraction.model_name}[/cyan]...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.extraction.model_name)
    model = AutoModelForCausalLM.from_pretrained(cfg.extraction.model_name).to(device)

    # Initialize SAE & Buffer
    sae = create_sae(cfg.sae).to(device)
    console.print(f"Instantiated [magenta]{cfg.sae.architecture.upper()} SAE[/magenta] (d_in={cfg.sae.d_in}, d_sae={cfg.sae.d_sae}, k={cfg.sae.k})")

    buffer = ActivationBuffer(model, tokenizer, cfg.extraction, device)
    trainer = SAETrainer(sae, buffer, cfg, device)

    # Train
    trainer.train(max_steps=steps)


@app.command()
def analyze(
    checkpoint_path: str = typer.Option("checkpoints/gpt2_layer6_topk/sae_final.pt", "--checkpoint", "-m", help="Path to trained SAE checkpoint"),
    model_name: str = typer.Option("gpt2", "--model", help="HuggingFace model ID"),
    feature_idx: int = typer.Option(0, "--feature", "-f", help="Feature index to inspect"),
    top_k: int = typer.Option(8, "--top-k", "-k", help="Number of top tokens to display"),
):
    """
    Analyzes direct logit attribution and semantic concept projections for a feature.
    """
    device = get_optimal_device("auto")
    console.print(f"Loading SAE from [cyan]{checkpoint_path}[/cyan] on {device}...")
    sae = BaseSAE.from_pretrained(checkpoint_path, device=str(device))

    console.print(f"Loading base model [cyan]{model_name}[/cyan]...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    attributor = DirectLogitAttributor(model, tokenizer, sae, device)
    dla = attributor.attribute_feature(feature_idx, top_k=top_k)

    table = Table(title=f"Direct Logit Attribution: Feature #{feature_idx}")
    table.add_column("Rank", justify="center")
    table.add_column("Promoted Token", style="green")
    table.add_column("Logit Boost", justify="right", style="bold green")
    table.add_column("Suppressed Token", style="red")
    table.add_column("Logit Penalty", justify="right", style="bold red")

    for i in range(top_k):
        pos_tok, pos_val = dla["top_positive"][i]
        neg_tok, neg_val = dla["top_negative"][i]
        table.add_row(str(i + 1), f"'{pos_tok}'", f"+{pos_val:.3f}", f"'{neg_tok}'", f"{neg_val:.3f}")

    console.print(table)


@app.command()
def steer(
    checkpoint_path: str = typer.Option("checkpoints/gpt2_layer6_topk/sae_final.pt", "--checkpoint", "-m", help="Path to trained SAE checkpoint"),
    model_name: str = typer.Option("gpt2", "--model", help="HuggingFace model ID"),
    layer_idx: int = typer.Option(6, "--layer", "-l", help="Layer index for steering hook"),
    feature_idx: int = typer.Option(0, "--feature", "-f", help="Target feature index"),
    alpha: float = typer.Option(6.0, "--alpha", "-a", help="Steering strength coefficient"),
    prompt: str = typer.Option("The future of technology will be", "--prompt", "-p", help="Generation prompt"),
    tokens: int = typer.Option(40, "--tokens", "-t", help="Max new tokens"),
):
    """
    Demonstrates side-by-side text generation with and without concept steering.
    """
    device = get_optimal_device("auto")
    console.print(f"Loading SAE from [cyan]{checkpoint_path}[/cyan] on {device}...")
    sae = BaseSAE.from_pretrained(checkpoint_path, device=str(device))

    console.print(f"Loading base model [cyan]{model_name}[/cyan]...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    generator = SteeredGenerator(model, tokenizer, sae, layer_idx=layer_idx, device=device)
    res = generator.generate_comparison(
        prompt=prompt,
        feature_idx=feature_idx,
        alpha=alpha,
        max_new_tokens=tokens,
    )

    console.print(f"\n[bold yellow]Prompt:[/bold yellow] \"{prompt}\"")
    console.print(f"[bold cyan]Steering Target:[/bold cyan] Feature #{feature_idx} | Alpha: {alpha}\n")

    console.print("[bold red]--- Baseline Output (Unsteered, Alpha=0.0) ---[/bold red]")
    console.print(f"{res['baseline_text']}\n")

    console.print("[bold green]--- Steered Output (Alpha={alpha}) ---[/bold green]".format(alpha=alpha))
    console.print(f"{res['steered_text']}\n")


@app.command()
def ui():
    """
    Launches the interactive Streamlit Web Dashboard.
    """
    import subprocess
    app_path = Path(__file__).parent / "dashboard" / "app.py"
    console.print(f"[bold green]Launching Streamlit Dashboard from {app_path}...[/bold green]")
    subprocess.run(["streamlit", "run", str(app_path)])


if __name__ == "__main__":
    app()
