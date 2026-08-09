"""
Automated Feature Semantic Labeler.
Generates concise, human-interpretable descriptions of learned SAE features by combining
direct logit projections and high-activation token contexts.
"""

from typing import Dict, Any, List
from .direct_logit_attribution import DirectLogitAttributor
from .feature_analyzer import FeatureAnalyzer


class FeatureAutoLabeler:
    """
    Synthesizes automated descriptions for latent features.
    """

    def __init__(self, attributor: DirectLogitAttributor, analyzer: FeatureAnalyzer):
        self.attributor = attributor
        self.analyzer = analyzer

    def generate_label(self, feature_idx: int, texts: List[str]) -> Dict[str, Any]:
        """
        Generates a structured semantic explanation for a specific feature.
        """
        dla_results = self.attributor.attribute_feature(feature_idx, top_k=5)
        contexts = self.analyzer.find_top_activating_contexts(texts, feature_idx, top_k=3)

        top_pos_tokens = [tok for tok, _ in dla_results["top_positive"][:4]]
        context_tokens = [c["target_token"] for c in contexts]

        # Heuristic concept synthesis
        summary_tokens = list(set(top_pos_tokens + context_tokens))[:5]
        if summary_tokens:
            semantic_summary = f"Tokens related to [{', '.join(summary_tokens)}]"
        else:
            semantic_summary = "Inactive / Undetermined feature"

        return {
            "feature_idx": feature_idx,
            "semantic_label": semantic_summary,
            "top_promoted_tokens": dla_results["top_positive"],
            "top_suppressed_tokens": dla_results["top_negative"],
            "top_activating_contexts": contexts,
        }
