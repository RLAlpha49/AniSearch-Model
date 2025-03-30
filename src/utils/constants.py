"""
# Application Constants

Centralized configuration constants for the anime/manga search application.

This module defines all the global constants used throughout the application,
including default paths, model configurations, and search parameters. Centralizing
these values makes it easier to maintain consistent settings across the application
and simplifies configuration changes.

## Constants Categories

- **Dataset Paths**: Locations of the merged anime and manga datasets
- **Model Configuration**: Default model name, batch size, and result count
- **Alternative Models**: Dictionary of other models that can be used

## Usage Context

These constants are imported and used throughout the application:

1. Model initialization uses the model name and dataset paths
2. Search operations use the default number of results and batch size
3. Model listing commands use the alternative models dictionary

Using constants instead of hard-coded values improves maintainability and
ensures consistency across the application.
"""

from typing import Dict

# Default paths
ANIME_DATASET_PATH: str = "model/merged_anime_dataset.csv"
"""Path to the merged anime dataset CSV file."""

MANGA_DATASET_PATH: str = "model/merged_manga_dataset.csv"
"""Path to the merged manga dataset CSV file."""

MODEL_NAME: str = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"  # A good default cross-encoder model
)
"""
Default cross-encoder model used for search operations.

This model offers a good balance between performance and accuracy for text ranking tasks.
It's a MiniLM model with 6 layers, which makes it relatively lightweight while still
providing good search results.
"""

NUM_RESULTS: int = 5  # Default number of results to return
"""
Default number of search results to return.

This controls how many top matches are returned when performing a search operation.
Can be overridden via command-line arguments.
"""

DEFAULT_BATCH_SIZE: int = 256  # Default batch size for processing
"""
Default batch size for model inference and data processing operations.

Larger batch sizes generally provide better performance up to hardware limits.
This value may need adjustment based on available memory and processor capabilities.
"""

# Alternative cross-encoder models that can be used
ALTERNATIVE_MODELS: Dict[str, Dict[str, str]] = {
    # MS Marco models - Text Ranking
    "ms_marco_models": {
        "ms-marco-MiniLM-L2-v2": "cross-encoder/ms-marco-MiniLM-L2-v2",
        "ms-marco-MiniLM-L4-v2": "cross-encoder/ms-marco-MiniLM-L4-v2",
        "ms-marco-MiniLM-L6-v2": "cross-encoder/ms-marco-MiniLM-L6-v2",  # Default model
        "ms-marco-MiniLM-L12-v2": "cross-encoder/ms-marco-MiniLM-L12-v2",
        "ms-marco-TinyBERT-L2": "cross-encoder/ms-marco-TinyBERT-L2",
        "ms-marco-TinyBERT-L2-v2": "cross-encoder/ms-marco-TinyBERT-L2-v2",
        "ms-marco-TinyBERT-L4": "cross-encoder/ms-marco-TinyBERT-L4",
        "ms-marco-TinyBERT-L6": "cross-encoder/ms-marco-TinyBERT-L6",
        "ms-marco-electra-base": "cross-encoder/ms-marco-electra-base",
    }
}
"""
Dictionary of alternative cross-encoder models available for use.

Organized by category (currently only 'ms_marco_models'), this dictionary maps
friendly model names to their full HuggingFace model identifiers. These models 
can be selected via the --model command-line argument.

Model performance characteristics:
- TinyBERT models: Smallest and fastest, good for low-resource environments
- MiniLM models: Good balance of performance and efficiency
- ELECTRA models: Higher accuracy but more computationally intensive
"""
