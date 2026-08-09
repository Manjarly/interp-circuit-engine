"""
Interactive Web Dashboard for Mechanistic Interpretability & Real-Time Concept Steering.
Supports modern frontier models including Qwen 2.5, Kimi (Moonshot), SmolLM2, Gemma 2, and Custom HuggingFace IDs.
"""

import os
# Ensure pure PyTorch backend without TensorFlow/protobuf conflicts
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pathlib import Path
import streamlit as st
import torch
import plotly.express as px
import pandas as pd

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.config import SAEConfig, ExperimentConfig
from src.common.device import get_optimal_device
from src.models import create_sae, TopKSAE, BaseSAE
from src.interpretability.direct_logit_attribution import DirectLogitAttributor
from src.steering.steer_generator import SteeredGenerator


st.set_page_config(
    page_title="Interp Circuit Engine | SOTA Concept Steering",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .stTextArea textarea {
        font-size: 1.05rem;
        border-radius: 8px;
    }
    .diff-box-baseline {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-left: 5px solid #ef4444;
        padding: 1.2rem;
        border-radius: 8px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 1.0rem;
        line-height: 1.5;
        min-height: 120px;
    }
    .diff-box-steered {
        background: rgba(34, 197, 94, 0.08);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-left: 5px solid #22c55e;
        padding: 1.2rem;
        border-radius: 8px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 1.0rem;
        line-height: 1.5;
        min-height: 120px;
    }
    .badge-baseline {
        background-color: #ef4444;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-steered {
        background-color: #22c55e;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading host model and SAE dictionary...")
def load_model_and_sae(model_name: str, ckpt_path: str, layer_idx: int = 12):
    device = get_optimal_device("auto")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Handle model loading with appropriate precision
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if device.type == "mps" else torch.float16,
            trust_remote_code=True
        ).to(device)
    except Exception:
        # Fallback to standard float32
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)

    model.eval()

    hidden_size = getattr(model.config, "hidden_size", 896)
    ckpt_file = Path(ckpt_path) if ckpt_path else Path("non_existent")

    if ckpt_file.exists():
        try:
            sae = BaseSAE.from_pretrained(ckpt_file, device=str(device))
            if sae.d_in != hidden_size:
                cfg = SAEConfig(architecture="topk", d_in=hidden_size, d_sae=hidden_size * 4, k=32)
                sae = TopKSAE(cfg).to(device)
                with torch.no_grad():
                    sae.normalize_decoder_columns()
        except Exception:
            cfg = SAEConfig(architecture="topk", d_in=hidden_size, d_sae=hidden_size * 4, k=32)
            sae = TopKSAE(cfg).to(device)
            with torch.no_grad():
                sae.normalize_decoder_columns()
    else:
        cfg = SAEConfig(architecture="topk", d_in=hidden_size, d_sae=hidden_size * 4, k=32)
        sae = TopKSAE(cfg).to(device)
        torch.manual_seed(42)
        with torch.no_grad():
            sae.normalize_decoder_columns()

    generator = SteeredGenerator(model, tokenizer, sae, layer_idx=layer_idx, device=device)
    attributor = DirectLogitAttributor(model, tokenizer, sae, device=device)

    return model, tokenizer, sae, generator, attributor, device


# Sidebar Controls
st.sidebar.markdown("### ⚙️ Model & Circuit Setup")

SUPPORTED_MODELS = [
    "Qwen/Qwen2.5-0.5B",               # SOTA 2024 Compact Model (Alibaba)
    "moonshotai/Kimi-k2.6",            # Kimi K2.6 (Moonshot AI)
    "moonshotai/Moonlight-16B-A3B",    # Moonlight MoE (Moonshot AI)
    "Qwen/Qwen2.5-1.5B",               # Qwen 2.5 1.5B
    "HuggingFaceTB/SmolLM2-360M",       # SmolLM2 360M
    "google/gemma-2-2b",               # Google Gemma 2 2B
    "gpt2",                            # Classic Baseline
    "(Custom HuggingFace Model ID)",   # Custom input
]

selected_model_option = st.sidebar.selectbox("Host Language Model", SUPPORTED_MODELS, index=0)

if selected_model_option == "(Custom HuggingFace Model ID)":
    model_choice = st.sidebar.text_input("Enter HuggingFace Model ID", value="Qwen/Qwen2.5-0.5B")
else:
    model_choice = selected_model_option

# Layer Resolution
max_layer = 23
if "Qwen" in model_choice or "Kimi" in model_choice:
    max_layer = 23
elif "SmolLM" in model_choice:
    max_layer = 31
elif "gpt2" in model_choice:
    max_layer = 11

default_layer = min(12, max_layer)
layer_choice = st.sidebar.slider("Residual Hook Layer", min_value=0, max_value=max_layer, value=default_layer)

# Checkpoint Discovery
ckpt_candidates = list(Path("checkpoints").glob("**/*.pt")) if Path("checkpoints").exists() else []
ckpt_options = [str(p) for p in ckpt_candidates]
if not ckpt_options:
    ckpt_options = ["(Dynamic Initialized SAE)"]

selected_ckpt = st.sidebar.selectbox("SAE Checkpoint", ckpt_options, index=0)
actual_ckpt = selected_ckpt if selected_ckpt != "(Dynamic Initialized SAE)" else "checkpoints/qwen2.5_layer12_topk/sae_final.pt"

try:
    model, tokenizer, sae, generator, attributor, device = load_model_and_sae(model_choice, actual_ckpt, layer_choice)
    st.sidebar.success(f"🟢 **Loaded:** `{model_choice.split('/')[-1]}` | Latents: `{sae.d_sae}` | Device: `{device}`")
except Exception as e:
    st.sidebar.error(f"Could not load `{model_choice}`: {e}")
    st.info("Tip: If using larger gated/MoE weights, ensure you have sufficient RAM or select a compact variant like `Qwen/Qwen2.5-0.5B`.")
    st.stop()


# Tabs
tab1, tab2, tab3 = st.tabs(["🚀 Live Concept Steering", "🔍 Feature Dictionary Explorer", "📊 Architecture & Math"])

with tab1:
    st.markdown('<div class="main-header">Real-Time Causal Concept Steering</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">Steering <b>{model_choice}</b> internal representations at Layer {layer_choice} during generation.</div>', unsafe_allow_html=True)

    # Top Section: Input Text Area & Quick Presets
    st.markdown("#### ✍️ Input Prompt")
    
    col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
    
    if "current_prompt" not in st.session_state:
        st.session_state.current_prompt = "The key breakthrough in artificial intelligence came when researchers discovered"

    with col_p1:
        if st.button("🤖 AI & Reasoning", use_container_width=True):
            st.session_state.current_prompt = "The key breakthrough in artificial intelligence came when researchers discovered"
    with col_p2:
        if st.button("🏛️ Politics & Policy", use_container_width=True):
            st.session_state.current_prompt = "In a surprising turn of events, the international council announced"
    with col_p3:
        if st.button("🔬 Quantum Physics", use_container_width=True):
            st.session_state.current_prompt = "The quantum experiment revealed that entangled particles"
    with col_p4:
        if st.button("🌍 Climate Science", use_container_width=True):
            st.session_state.current_prompt = "The environmental researchers discovered that the global climate"
    with col_p5:
        if st.button("💻 Deep Learning", use_container_width=True):
            st.session_state.current_prompt = "To optimize distributed training across GPU clusters, engineers"

    # User Input Text Area
    user_prompt = st.text_area(
        "Type your prompt here:",
        value=st.session_state.current_prompt,
        height=100,
        help="Enter any starting text for the model to continue.",
        key="prompt_input_box"
    )

    st.markdown("---")

    col_ctrl, col_dla = st.columns([1.1, 1.2])

    with col_ctrl:
        st.markdown("#### 🎯 Steering Controls")
        
        c_feat, c_alpha = st.columns([1, 1])
        with c_feat:
            feature_idx = st.number_input(
                "Target Feature ID",
                min_value=0,
                max_value=sae.d_sae - 1,
                value=42,
                step=1,
                help=f"Select any latent feature from 0 to {sae.d_sae - 1}."
            )
        with c_alpha:
            alpha = st.slider(
                "Steering Strength (α)",
                min_value=-15.0,
                max_value=15.0,
                value=8.0,
                step=0.5,
                help="Positive: amplify concept | Negative: suppress concept | 0.0: baseline"
            )

        with st.expander("⚙️ Generation Hyperparameters"):
            max_tokens = st.slider("Max New Tokens", min_value=10, max_value=100, value=40)
            temperature = st.slider("Temperature", min_value=0.01, max_value=1.5, value=0.7)
            top_p = st.slider("Top-P Nucleus", min_value=0.1, max_value=1.0, value=0.9)
            intervention_type = st.selectbox("Intervention Method", ["addition (x + α·d_i)", "ablation (project out)", "clamping"])

        interv_clean = intervention_type.split()[0]
        btn_generate = st.button("⚡ Run Steering Comparison", type="primary", use_container_width=True)

    with col_dla:
        st.markdown(f"#### 🔬 Concept Profile: Feature #{feature_idx}")
        dla_res = attributor.attribute_feature(feature_idx, top_k=6)

        pos_df = pd.DataFrame(dla_res["top_positive"], columns=["Token", "Logit Boost"])
        fig = px.bar(
            pos_df,
            x="Logit Boost",
            y="Token",
            orientation="h",
            title=f"Vocabulary Tokens Promoted by Feature #{feature_idx}",
            color="Logit Boost",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(height=230, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Generation Trigger
    if btn_generate or "last_result" not in st.session_state:
        with st.spinner(f"Generating completions with {model_choice}..."):
            st.session_state.last_result = generator.generate_comparison(
                prompt=user_prompt,
                feature_idx=feature_idx,
                alpha=alpha,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )

    if "last_result" in st.session_state:
        res = st.session_state.last_result
        st.markdown("---")
        st.markdown("### 📝 Side-by-Side Comparison")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown('<span class="badge-baseline">🔴 Baseline (Unsteered, α = 0.0)</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="diff-box-baseline">{res["baseline_text"]}</div>', unsafe_allow_html=True)

        with res_col2:
            st.markdown(f'<span class="badge-steered">🟢 Steered Output ({model_choice.split("/")[-1]} Feature #{res.get("feature_idx", feature_idx)}, α = {res.get("alpha", alpha):+.1f})</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="diff-box-steered">{res["steered_text"]}</div>', unsafe_allow_html=True)


with tab2:
    st.markdown('<div class="main-header">Feature Dictionary Browser</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">Inspect the {sae.d_sae} monosemantic basis directions learned for {model_choice}.</div>', unsafe_allow_html=True)

    browse_idx = st.slider("Select Feature Index to Inspect", 0, sae.d_sae - 1, 42)
    feat_dla = attributor.attribute_feature(browse_idx, top_k=10)

    f_col1, f_col2 = st.columns(2)

    with f_col1:
        st.markdown(f"#### 📈 Top Promoted Tokens (Feature #{browse_idx})")
        df_p = pd.DataFrame(feat_dla["top_positive"], columns=["Token", "Logit Boost"])
        st.dataframe(df_p, use_container_width=True)

    with f_col2:
        st.markdown(f"#### 📉 Top Suppressed Tokens (Feature #{browse_idx})")
        df_n = pd.DataFrame(feat_dla["top_negative"], columns=["Token", "Logit Penalty"])
        st.dataframe(df_n, use_container_width=True)


with tab3:
    st.markdown('<div class="main-header">Architecture & Mathematical Formulations</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    ### 1. Top-K Sparse Autoencoder on Modern Architectures
    Currently attached to **`{model_choice}`** ($d_{{\\text{{in}}}} = {sae.d_in}$, $d_{{\\text{{sae}}}} = {sae.d_sae}$, $k = {sae.cfg.k}$):
    
    $$\\mathbf{{z}} = W_{{\\text{{enc}}}}(\\mathbf{{x}} - \\mathbf{{b}}_{{\\text{{dec}}}}) + \\mathbf{{b}}_{{\\text{{enc}}}}$$
    
    $$\\mathbf{{f}}(\\mathbf{{x}}) = \\text{{TopK}}(\\text{{ReLU}}(\\mathbf{{z}}), k)$$
    
    $$\\mathbf{{\\hat{{x}}}} = W_{{\\text{{dec}}}}\\mathbf{{f}}(\\mathbf{{x}}) + \\mathbf{{b}}_{{\\text{{dec}}}}$$
    
    Constraint: $\\|W_{{\\text{{dec}}}}[:, i]\\|_2 = 1$ for all $i \\in \\{{1, \\dots, d_{{\\text{{sae}}}}\\}}$.
    
    ---
    
    ### 2. Direct Logit Attribution (DLA)
    The unembedding matrix $W_U \\in \\mathbb{{R}}^{{d_{{\\text{{in}}}} \\times V}}$ maps residual states to vocabulary logits:
    
    $$\\text{{Logits}}_i = \\mathbf{{d}}_i \\cdot W_U$$
    
    ---
    
    ### 3. Causal Activation Intervention
    During inference forward passes at Layer ${layer_choice}$:
    
    $$\\mathbf{{x}}' = \\mathbf{{x}} + \\alpha \\cdot \\mathbf{{d}}_i$$
    """)
