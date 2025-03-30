"""
Utility functions for displaying search results and model information.
"""

from typing import Dict, Optional
from src.utils.constants import ALTERNATIVE_MODELS
from src.utils.error_handling import handle_exceptions


@handle_exceptions(include_exc_info=True)
def format_score(score: float, normalize: bool, model_name: str = "") -> str:
    """
    Format a score for display based on whether it's normalized.

    Args:
        score: The score to format
        normalize: Whether the score is normalized between 0-1
        model_name: Name of the model (used to determine if it's a GooAQ model)

    Returns:
        A formatted string representation of the score
    """
    if normalize:
        # Check if this is a GooAQ model
        is_gooaq = "gooaq" in model_name.lower()

        if is_gooaq:
            # For GooAQ models, multiply by 10 instead of 100
            return f"Match: {score*10:.1f}%"
        # For other normalized scores (0-1), display as percentage
        return f"Match: {score*100:.1f}%"
    # For non-normalized scores, display the raw value
    return f"Score: {score:.2f}"


@handle_exceptions(cli_mode=True, reraise=False)
def display_available_models(
    fine_tuned_models: Optional[Dict[str, str]] = None,
) -> None:
    """
    Display available cross-encoder models.

    Args:
        fine_tuned_models: Dictionary of fine-tuned models to display
    """
    # Use the constant instead of importing BaseSearchModel which would load TensorFlow
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
