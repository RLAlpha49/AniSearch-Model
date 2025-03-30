"""
Display utilities for the anime search application.

WARNING: This module should NOT import any ML frameworks like TensorFlow or PyTorch,
         as it's used for lightweight model listing without loading heavy dependencies.
"""

import os
from typing import Dict, Optional

from src.utils.constants import ALTERNATIVE_MODELS
from src.utils.error_handling import handle_exceptions

# Model save path
MODEL_SAVE_PATH = "model/fine-tuned/"


def format_score(score: float, normalize_scores: bool, model_name: str) -> str:
    """
    Format a score for display.

    Args:
        score: The raw score
        normalize_scores: Whether scores are normalized
        model_name: The model name for context

    Returns:
        Formatted score string
    """
    if normalize_scores or "ms-marco" in model_name.lower():
        # For MS Marco models or normalized scores, display as percentage
        return f"{score:.1%} relevance"
    else:
        # For other models, just show the raw score
        return f"score: {score:.3f}"


def list_fine_tuned_models() -> Dict[str, str]:
    """
    List available fine-tuned models.

    This is a lightweight version that doesn't import TensorFlow/PyTorch.

    Returns:
        Dictionary mapping model names to their paths
    """
    if not os.path.exists(MODEL_SAVE_PATH):
        return {}

    fine_tuned_models = {}
    for model_name in os.listdir(MODEL_SAVE_PATH):
        model_path = os.path.join(MODEL_SAVE_PATH, model_name)
        config_path = os.path.join(model_path, "config.json")

        if os.path.isdir(model_path) and os.path.exists(config_path):
            fine_tuned_models[model_name] = model_path

    return fine_tuned_models


@handle_exceptions(cli_mode=True, reraise=False)
def display_available_models(
    fine_tuned_models: Optional[Dict[str, str]] = None,
) -> None:
    """
    Display available models for searching/training.

    Args:
        fine_tuned_models: Dictionary of fine-tuned models to display
    """
    models = ALTERNATIVE_MODELS

    print("\nAvailable Pre-trained Cross-Encoder Models:")
    print("======================================")

    for category, model_dict in models.items():
        print(f"\n{category.upper()}:")
        for name, path in model_dict.items():
            print(f"  {name}: {path}")

    # Display fine-tuned models if provided
    if fine_tuned_models:
        print("\nAvailable Fine-tuned Models:")
        print("==========================")
        for name, path in fine_tuned_models.items():
            print(f"  {name}: {path}")

    print("\nUsage example:")
    print(
        '  python src/main.py search --type anime --query "Your query" '
        '--model "cross-encoder/ms-marco-MiniLM-L-6-v2"'
    )
    if fine_tuned_models:
        print("\nTo use a fine-tuned model:")
        print(
            '  python src/main.py search --type anime --query "Your query" '
            '--model "model/fine-tuned/your-model-name"'
        )
    print("\nModel selection guide:")
    print("- TinyBERT models: Smallest and fastest, good for low-resource environments")
    print("- MiniLM models: Good balance of performance and efficiency")
    print("- ELECTRA models: Higher accuracy but more computationally intensive")
    print("- MS Marco models: Optimized for information retrieval")
    print("- Fine-tuned models: Domain-specific models trained on anime/manga data")
