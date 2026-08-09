"""
Streaming text sampler and tokenization pipeline.
Handles streaming online HuggingFace datasets with a diverse built-in fallback corpus.
"""

from typing import Iterator, List, Optional, Any
import torch

# High-diversity fallback corpus covering multiple semantic domains
BUILTIN_CORPUS = [
    "The transformer architecture relies on multi-head self-attention mechanisms to capture long-range contextual dependencies across tokens.",
    "Quantum computing leverages superposition and entanglement to solve specific computational complexity classes exponentially faster than classical Turing machines.",
    "In economics, price elasticity of demand measures the responsiveness of the quantity demanded to a change in the product price.",
    "The French Revolution began in 1789 with the Storming of the Bastille, overthrowing the monarchy and establishing a republic.",
    "Photosynthesis converts light energy, water, and carbon dioxide into glucose and oxygen inside the chloroplasts of plant cells.",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)",
    "Neuroplasticity refers to the brain's ability to reorganize itself by forming new neural connections throughout life in response to learning.",
    "The global climate system involves complex non-linear feedbacks between the atmosphere, oceans, cryosphere, and terrestrial biosphere.",
    "The Renaissance was a fervent period of European cultural, artistic, political and economic rebirth following the Middle Ages.",
    "General relativity describes gravitation as a geometric property of space and time, or spacetime curvature caused by mass-energy.",
    "Market microstructure studies the trading mechanisms that determine transaction prices, bid-ask spreads, and order book queue dynamics.",
    "Epistemology is the philosophical branch concerned with knowledge, belief justification, rationality, and skepticism.",
    "The immune system uses T-cells, B-cells, and antibodies to neutralize foreign pathogens and retain immunological memory.",
    "Sparse Autoencoders decompose dense representation vectors into monosemantic, interpretable latent feature directions.",
    "Deep learning optimization often navigates non-convex loss landscapes using stochastic gradient descent and adaptive learning rates.",
    "The Renaissance painter Leonardo da Vinci created masterpieces including the Mona Lisa and The Last Supper.",
    "Machine learning models must balance the bias-variance tradeoff to achieve generalization on unseen test distributions.",
    "Graph theory analyzes networks of vertices connected by edges, finding applications in routing, social networks, and chemistry.",
]


class TextSampler:
    """
    Yields batches of tokenized sequences for activation extraction.
    """

    def __init__(
        self,
        tokenizer: Any,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        seq_len: int = 128,
        batch_size: int = 16,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config

        if hasattr(self.tokenizer, "pad_token") and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._hf_dataset = self._init_dataset()

    def _init_dataset(self):
        try:
            from datasets import load_dataset
            dataset = load_dataset(self.dataset_name, self.dataset_config, split="train", streaming=True)
            return dataset
        except Exception:
            return None

    def _stream_text_chunks(self) -> Iterator[str]:
        if self._hf_dataset is not None:
            try:
                for sample in self._hf_dataset:
                    text = sample.get("text", "")
                    if len(text.strip()) > 50:
                        yield text
                return
            except Exception:
                pass

        # Fallback generator: repeat and tile built-in corpus
        idx = 0
        while True:
            yield BUILTIN_CORPUS[idx % len(BUILTIN_CORPUS)]
            idx += 1

    def stream_token_batches(self, device: torch.device) -> Iterator[torch.Tensor]:
        """
        Yields batches of token tensors of shape (batch_size, seq_len).
        """
        text_stream = self._stream_text_chunks()
        token_accumulator: List[int] = []

        while True:
            while len(token_accumulator) < self.batch_size * self.seq_len:
                text = next(text_stream)
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                token_accumulator.extend(tokens)

            # Slice exact batch
            batch_tokens = token_accumulator[: self.batch_size * self.seq_len]
            token_accumulator = token_accumulator[self.batch_size * self.seq_len :]

            tensor = torch.tensor(batch_tokens, dtype=torch.long, device=device)
            tensor = tensor.view(self.batch_size, self.seq_len)
            yield tensor
