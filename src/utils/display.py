"""
# Display Utilities

Formatting and display functions for the anime/manga search application.

This module provides utilities for displaying model information, formatting search
results, and presenting information to users in a consistent and readable format.
It's designed to be lightweight and not import any heavy ML dependencies, allowing
it to be used for model listing without loading TensorFlow or PyTorch.

## Features

- Score formatting that adapts to different model types
- Model listing capabilities for both pre-trained and fine-tuned models
- User-friendly console display functions
- Error handling for display operations

## Usage Context

These utilities are primarily used in:

1. CLI output formatting for search results
2. Model listing for the '--list-models' CLI argument
3. Interactive search result display

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
    Format a model's relevance score for user-friendly display.

    This function formats search result scores differently based on the model type
    and normalization settings. It handles two main cases:

    1. MS Marco models or other models with normalized scores (0-1 range)
       These are displayed as percentages for intuitive interpretation
    2. Other models with unnormalized scores
       These are displayed as raw float values with fixed precision

    Args:
        score: The raw relevance score from the model, typically a float between
            0 and 1 for normalized models, or any float range for others.
        normalize_scores: Boolean flag indicating whether the scores are normalized
            to the 0-1 range. This affects the display format.
        model_name: Name of the model that produced the score, used to detect
            specific model types that require special formatting.

    Returns:
        str: A formatted string representation of the score, either as a percentage
            (e.g., "95.2% relevance") or as a raw score (e.g., "score: 0.952").

    Examples:
        ```python
        # Format a score from an MS Marco model
        formatted = format_score(0.952, True, "cross-encoder/ms-marco-MiniLM-L-6-v2")
        # Result: "95.2% relevance"

        # Format a score from a non-normalized model
        formatted = format_score(4.73, False, "cross-encoder/stsb-roberta-base")
        # Result: "score: 4.730"
        ```

    Notes:
        - MS Marco models are automatically detected by checking if "ms-marco" appears
          in the model name (case-insensitive)
        - Percentage format shows one decimal place (e.g., 95.2%)
        - Raw score format shows three decimal places (e.g., 4.730)
    """
    if normalize_scores or "ms-marco" in model_name.lower():
        # For MS Marco models or normalized scores, display as percentage
        return f"{score:.1%} relevance"
    # For other models, just show the raw score
    return f"score: {score:.3f}"


def list_fine_tuned_models() -> Dict[str, str]:
    """
    List all available fine-tuned models in the model directory.

    This function scans the fine-tuned model directory for valid model folders,
    identifying them by the presence of a config.json file. It returns a mapping
    of model names to their full paths, which can be used to load the models.

    This is a lightweight implementation that doesn't import heavy ML frameworks
    like TensorFlow or PyTorch, making it suitable for quick model listing without
    the overhead of loading these dependencies.

    Returns:
        Dict[str, str]: A dictionary where:
            - Keys are the model directory names (model identifiers)
            - Values are the full paths to the model directories

            Returns an empty dictionary if no models are found or if the
            model directory doesn't exist.

    Examples:
        ```python
        # Get a dictionary of available fine-tuned models
        models = list_fine_tuned_models()

        # Print the available models
        if models:
            print("Available fine-tuned models:")
            for name, path in models.items():
                print(f"- {name}: {path}")
        else:
            print("No fine-tuned models found")
        ```

    Notes:
        - Models are identified by the presence of a config.json file
        - The default search location is the MODEL_SAVE_PATH constant ("model/fine-tuned/")
        - This function does not validate that the models are functional or compatible
        - Use this function before attempting to load a fine-tuned model to check availability
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
    Display a formatted list of available models for searching or training.

    This function prints a comprehensive list of available models to the console,
    organized by category. It displays both pre-trained models from the constants
    and fine-tuned models if provided. The output includes usage examples and a
    guide to help users select appropriate models for their needs.

    The function is decorated with error handling to gracefully handle any exceptions
    that might occur during the display process.

    Args:
        fine_tuned_models: Optional dictionary mapping model names to their paths.
            If provided, these fine-tuned models will be displayed in a separate section.
            If None, only pre-trained models will be shown.

    Returns:
        None: This function prints information to the console but doesn't return any values.

    Example Output:
        ```
        Available Pre-trained Cross-Encoder Models:
        ======================================

        MS_MARCO_MODELS:
          ms-marco-MiniLM-L6-v2: cross-encoder/ms-marco-MiniLM-L6-v2
          ms-marco-TinyBERT-L6: cross-encoder/ms-marco-TinyBERT-L6
          ...

        Available Fine-tuned Models:
        ==========================
          anime-search-v1: model/fine-tuned/anime-search-v1
          ...

        Usage example:
          python src/main.py search --type anime --query "Your query" \
          --model "cross-encoder/ms-marco-MiniLM-L-6-v2"

        To use a fine-tuned model:
          python src/main.py search --type anime --query "Your query" \
          --model "model/fine-tuned/your-model-name"

        Model selection guide:
        - TinyBERT models: Smallest and fastest, good for low-resource environments
        - MiniLM models: Good balance of performance and efficiency
        ...
        ```

    Notes:
        - The function accesses ALTERNATIVE_MODELS from constants.py
        - The output includes usage examples tailored to the available models
        - The model selection guide helps users choose appropriate models
        - Error handling is provided by the @handle_exceptions decorator
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
