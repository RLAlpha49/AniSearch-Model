"""
Utilities package for the anime search model application.

This module provides a collection of utility functions and constants that are used
throughout the application. It includes:

- Dataset paths for anime and manga
- Model name and number of results constants
- Batch size and alternative model configurations
"""

from src.utils.constants import (
    ANIME_DATASET_PATH,
    MANGA_DATASET_PATH,
    MODEL_NAME,
    NUM_RESULTS,
    DEFAULT_BATCH_SIZE,
    ALTERNATIVE_MODELS,
)
from src.utils.display import list_fine_tuned_models, display_available_models
from src.utils.error_handling import handle_exceptions
from src.utils.logging_config import setup_logging

__all__ = [
    "ANIME_DATASET_PATH",
    "MANGA_DATASET_PATH",
    "MODEL_NAME",
    "NUM_RESULTS",
    "DEFAULT_BATCH_SIZE",
    "ALTERNATIVE_MODELS",
    "list_fine_tuned_models",
    "display_available_models",
    "handle_exceptions",
    "setup_logging",
]
