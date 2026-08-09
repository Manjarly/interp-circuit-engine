# 🧠 Interp Circuit Engine (`interp-circuit-engine`)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-15%20passed-brightgreen.svg)](tests/)

**An End-to-End Research & Systems Suite for Mechanistic Interpretability, Sparse Autoencoders (SAEs), and Causal Concept Steering on State-of-the-Art Open-Weights LLMs in PyTorch.**

Supports modern transformer architectures (**`Qwen/Qwen2.5`**, **`google/gemma-2`**, **`meta-llama/Llama-3`**, **`HuggingFaceTB/SmolLM2`**, **`GPT-2`**).

---

## 🌟 Key Capabilities

* **🤖 Modern Architecture Support (2024 SOTA):**
  * Fully compatible with RoPE, Grouped-Query Attention (GQA), SwiGLU, and RMSNorm architectures like **Qwen 2.5 (0.5B/1.5B/7B)**, **Gemma 2 (2B/9B)**, and **SmolLM2**.
* **🔬 State-of-the-Art SAE Architectures:**
  * **Top-K SAE** (*Gao et al., OpenAI / Anthropic 2024*): Eliminates $L_1$ shrinkage bias with exact $k$-sparsity per token and dead-latent auxiliary losses.
  * **Gated SAE** (*Rajamanoharan et al., Google DeepMind 2024*): Decouples feature gating from magnitude estimation.
  * **JumpReLU SAE**: Threshold-based activation with straight-through estimator (STE).
  * **Standard $L_1$ SAE**: Classical baseline with unit-norm constrained decoders ($\|W_{\text{dec}}[:, i]\|_2 = 1$).
* **⚡ High-Throughput Streaming Extraction Buffer:**
  * PyTorch forward hooks for transformer residual streams (`resid_post`, `mlp_out`, `attn_out`).
  * In-memory token-shuffling ring buffer to break sequential intra-document auto-correlations.
  * Native Apple Silicon (MPS), CUDA, and CPU acceleration.
* **🛡️ Active Dead Latent Tracker & Resampler:**
  * Detects inactive neurons and resamples weights to align with high-error residual vectors ($\mathbf{x} - \mathbf{\hat{x}}$).
* **📊 Direct Logit Attribution (DLA) & Auto-Labeling:**
  * Projects decoder feature directions $\mathbf{d}_i$ onto the unembedding matrix $W_U$ to identify tokens promoted or suppressed.
* **🎯 Causal Activation Steering Engine:**
  * Real-time autoregressive text generation intervention: $\mathbf{x}_{\text{steered}} = \mathbf{x} + \alpha \cdot \mathbf{d}_i$.
  * Supports addition, clamping, ablation (subspace removal), and multi-concept combinations.
* **🖥️ Interactive Streamlit Web UI:**
  * Visual dictionary browser, live token attribution charts, and side-by-side prompt generation with dynamic steering sliders.

---

## 🚀 Quickstart

### 1. Launch the Interactive Web Dashboard

```bash
streamlit run src/dashboard/app.py
```
Open **`http://localhost:8501`** in your browser!

### 2. Train on Modern LLMs (e.g. Qwen 2.5)

```bash
# Train Top-K SAE on Qwen 2.5 Layer 12
python -m src.cli train --config configs/qwen2.5_0.5b_layer12.yaml --steps 1000
```

### 3. Inspect Feature Semantics (Direct Logit Attribution)

```bash
python -m src.cli analyze \
  --checkpoint checkpoints/qwen2.5_layer12_topk/sae_final.pt \
  --model Qwen/Qwen2.5-0.5B \
  --feature 42 \
  --top-k 8
```

### 4. Run Causal Concept Steering in Terminal

```bash
python -m src.cli steer \
  --checkpoint checkpoints/qwen2.5_layer12_topk/sae_final.pt \
  --model Qwen/Qwen2.5-0.5B \
  --layer 12 \
  --feature 42 \
  --alpha 8.0 \
  --prompt "The key breakthrough in artificial intelligence came when" \
  --tokens 40
```

---

## 🧪 Testing

Run the complete test suite:

```bash
pytest tests/ -v
```

---

## 📜 License

MIT License. Free for academic, research, and commercial use.
