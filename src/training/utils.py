"""
# Training Utilities

Utility functions for training and fine-tuning cross-encoder models.

This module provides specialized utility functions for training, evaluating, and
optimizing models in the anime/manga search application. It includes functionality
for text preprocessing, device management, and optimization settings that are
specifically tailored for cross-encoder model training.

## Features

- Random seed initialization for reproducible experiments
- Device detection and configuration for CPU/GPU training
- Efficient batch text truncation for handling large text pairs
- Data parsing utilities for handling list data from datasets
- Default training parameters for common scenarios

## Usage Context

These utilities are primarily used in:

1. Model fine-tuning workflows
2. Training script configuration
3. Dataset preprocessing for training

The functions work together to provide a consistent environment for model training
and help manage the complexities of preparing text data for transformer models.
"""

import ast
import logging
import math
import random
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from src.utils.error_handling import handle_exceptions

# Constants
MODEL_SAVE_PATH: str = "model/fine-tuned/"
"""Path where fine-tuned models are saved."""

# Default parameters
DEFAULT_EPOCHS: int = 3
"""
Default number of training epochs.

The model will iterate over the training data this many times. For cross-encoder
models, 3 epochs is often sufficient to get good performance while avoiding overfitting.
"""

DEFAULT_BATCH_SIZE: int = 16
"""
Default training batch size.

This batch size works well on most consumer GPUs with 8GB+ VRAM. Adjust based on 
available memory - larger batches generally provide more stable training but require
more memory.
"""

DEFAULT_EVAL_STEPS: int = 500
"""
Default number of steps between model evaluations during training.

Controls how frequently the model is evaluated on the validation set.
"""

DEFAULT_WARMUP_STEPS: int = 500
"""
Default number of learning rate warmup steps.

Learning rate starts at a low value and gradually increases to the full learning rate
over this many steps, which helps with training stability.
"""

DEFAULT_MAX_SAMPLES: int = 10000
"""
Default maximum number of training samples to use.

Limits the training dataset size to avoid excessive training times for large datasets.
Set to None to use the entire dataset.
"""

DEFAULT_LEARNING_RATE: float = 2e-6
"""
Default learning rate for fine-tuning.

A conservative learning rate that works well for most cross-encoder fine-tuning.
Smaller than typical learning rates for training from scratch to avoid disrupting
pre-trained weights.
"""

# Setup logger
logger = logging.getLogger(__name__)


def setup_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    This function sets consistent random seeds for all random number generators used
    in the training process, ensuring that experiments can be reproduced with the
    same randomization patterns. It sets seeds for:

    - Python's random module
    - NumPy's random number generator
    - PyTorch's CPU random number generator
    - PyTorch's GPU random number generators (if available)

    Args:
        seed: Integer value to use as the random seed. Default is 42, which is a
            common choice for reproducible machine learning experiments.

    Returns:
        None: This function doesn't return a value but sets global random states.

    Example:
        ```python
        # Initialize random seeds before training
        setup_random_seeds(42)

        # Now all random operations will be reproducible
        train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
        ```

    Notes:
        - Using the same seed guarantees the same random sequence across runs
        - Different hardware or PyTorch versions might still produce variations
        - For full reproducibility, also set deterministic algorithms in PyTorch
          configurations and control the environment more strictly
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: Optional[str] = None) -> str:
    """
    Determine the appropriate computing device for model training and inference.

    This function selects the best available device for running models, defaulting
    to CUDA (GPU) if available, and falling back to CPU if not. It also allows for
    explicitly specifying a device if needed.

    Args:
        device: Optional string specifying the device to use. If provided, this
            overrides the automatic detection. Valid values include 'cpu', 'cuda',
            'cuda:0', etc. Default is None, which triggers automatic detection.

    Returns:
        str: A string identifier for the device to use, compatible with PyTorch's
            device specification format (e.g., 'cuda', 'cpu', 'cuda:1').

    Example:
        ```python
        # Get the best available device
        device = get_device()

        # Move a model to the appropriate device
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        model.model = model.model.to(device)

        # Or explicitly specify a device
        device = get_device('cuda:1')  # Use second GPU if available
        ```

    Notes:
        - CUDA device is only returned if PyTorch can access CUDA
        - The function doesn't check for specific CUDA device availability beyond
          what torch.cuda.is_available() provides
        - For multi-GPU setups, you may want to explicitly specify a device
          or implement more sophisticated device selection logic
    """
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


@handle_exceptions(log_exceptions=True, include_exc_info=True)
def batch_truncate_text_pairs(
    text_pairs: List[Tuple[str, str]],
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    max_length: int = 512,
    batch_size: int = 128,
) -> List[Tuple[str, str]]:
    """
    Efficiently truncate multiple text pairs to fit within a specified token length.

    This function processes a large list of text pairs (query, text) and truncates
    them to fit within the model's maximum sequence length. It uses a batch processing
    approach for efficiency and preserves as much of the query (text_a) as possible,
    truncating the document (text_b) to fit the remaining space.

    The truncation process:
    1. Tokenizes all text_a entries to calculate their token lengths
    2. For each pair, reserves tokens for text_a plus special tokens
    3. Allocates remaining tokens for text_b and truncates as needed
    4. Performs validation checks on a sample of results to ensure compliance

    Args:
        text_pairs: List of tuples, each containing two strings (text_a, text_b).
            Typically, text_a is a query and text_b is a document or longer text.

        tokenizer: The tokenizer that will be used with the model. Must be a
            transformers PreTrainedTokenizer or PreTrainedTokenizerFast instance
            compatible with the target model.

        max_length: Maximum allowed sequence length in tokens (including all
            special tokens). Default is 512, which is common for many transformer
            models.

        batch_size: Number of text pairs to process in each batch. Higher values
            increase memory usage but improve processing speed. Default is 128.

    Returns:
        List[Tuple[str, str]]: A list of truncated text pairs, where each text_b
            has been truncated as needed to fit within the max_length constraint
            when combined with its text_a.

    Example:
        ```python
        from transformers import AutoTokenizer

        # Load a tokenizer
        tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # Sample text pairs (query, document)
        text_pairs = [
            ("short query", "very long document text that exceeds the limit..."),
            ("another query", "another document that's also quite long...")
        ]

        # Truncate to fit model's constraints
        truncated_pairs = batch_truncate_text_pairs(
            text_pairs=text_pairs,
            tokenizer=tokenizer,
            max_length=128,  # Short for example purposes
            batch_size=32
        )
        ```

    Notes:
        - The function prioritizes preserving text_a (usually the query) completely
        - Only text_b is truncated unless absolutely necessary
        - The function includes a double-check mechanism that samples some pairs
          to verify they actually fit within max_length
        - Very long text_a entries might result in empty text_b if there's no space left
        - The function uses the @handle_exceptions decorator for error handling
    """
    results = []
    total_batches = math.ceil(len(text_pairs) / batch_size)

    # First, tokenize all text_a entries to get their lengths
    logger.info("Pre-computing text_a token lengths")
    text_a_list = [pair[0] for pair in text_pairs]

    # Process queries in smaller batches to avoid memory issues
    text_a_lengths = []
    sub_batch_size = 1000  # Smaller batch size for tokenization

    for i in range(0, len(text_a_list), sub_batch_size):
        sub_batch = text_a_list[i : i + sub_batch_size]
        # Get token counts using direct encoding
        sub_lengths = [
            len(tokenizer.encode(text, add_special_tokens=True)) for text in sub_batch
        ]
        text_a_lengths.extend(sub_lengths)

    # Process in batches
    logger.info("Truncating text pairs in batches")
    for batch_idx in tqdm(range(total_batches), desc="Truncating pairs"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(text_pairs))
        batch_pairs = text_pairs[start_idx:end_idx]
        batch_a_lengths = text_a_lengths[start_idx:end_idx]

        # Calculate available tokens for each text_b in the batch
        special_tokens_count = 3  # Account for special tokens
        available_tokens_for_b = [
            max(0, max_length - length - special_tokens_count)
            for length in batch_a_lengths
        ]

        # Extract text_b entries for the batch
        batch_text_b = [pair[1] for pair in batch_pairs]

        # Process text_b truncation sequentially
        batch_truncated_b = []
        for i, (text_b, available_tokens) in enumerate(
            zip(batch_text_b, available_tokens_for_b)
        ):
            if available_tokens <= 0:
                batch_truncated_b.append("")
                continue

            # Tokenize and truncate
            truncated_b = tokenizer.encode(
                text_b,
                add_special_tokens=False,
                max_length=available_tokens,
                truncation=True,
            )

            # Decode back to text
            batch_truncated_b.append(
                tokenizer.decode(truncated_b, skip_special_tokens=True)
            )

        # Create truncated pairs for this batch
        batch_results = [
            (batch_pairs[i][0], batch_truncated_b[i]) for i in range(len(batch_pairs))
        ]

        # Double-check only a small sample (every 20th) to save time
        for i in range(0, len(batch_results), 20):
            if i >= len(batch_results):
                break

            text_a, text_b = batch_results[i]
            if not text_b:  # Skip empty text_b
                continue

            # Check the final length
            final_tokens = tokenizer.encode(
                text_a, text_b, add_special_tokens=True, truncation=False
            )
            final_length = len(final_tokens)

            # Emergency truncation if needed
            if final_length > max_length:
                available = max(
                    0,
                    max_length - batch_a_lengths[i] - special_tokens_count - 5,
                )
                if available <= 0:
                    batch_results[i] = (text_a, "")
                else:
                    truncated_b = tokenizer.encode(
                        batch_text_b[i],
                        add_special_tokens=False,
                        max_length=available,
                        truncation=True,
                    )
                    batch_results[i] = (
                        text_a,
                        tokenizer.decode(truncated_b, skip_special_tokens=True),
                    )

        results.extend(batch_results)

    return results


def parse_list_column(column_value: Any) -> List[str]:
    """
    Parse a list column from a dataset that may be stored as a string representation.

    This function handles various formats of list data that may come from CSV or
    DataFrame columns, converting them to Python lists. It handles:

    - String representations of lists like "[item1, item2, item3]"
    - Already parsed list objects
    - Single string values (converted to a single-item list)
    - None or NaN values (converted to empty list)

    Args:
        column_value: The value to parse, which could be a string representation of
            a list, an actual list object, a single string, or a missing value (None/NaN).

    Returns:
        List[str]: A list of strings parsed from the input. Returns an empty list
            for None or NaN values.

    Example:
        ```python
        # Parse string representation of a list
        genres = parse_list_column("['Action', 'Comedy', 'Drama']")
        # Result: ['Action', 'Comedy', 'Drama']

        # Parse a single string
        tags = parse_list_column("Shounen")
        # Result: ['Shounen']

        # Handle NaN values
        empty = parse_list_column(float('nan'))
        # Result: []
        ```

    Notes:
        - Uses ast.literal_eval to safely parse string representations of lists
        - Falls back to comma splitting if literal_eval fails
        - Handles missing values (None, NaN) by returning an empty list
        - Non-string, non-list inputs that aren't NaN will result in an empty list
    """
    if pd.isna(column_value):
        return []

    if isinstance(column_value, str):
        # Try to parse as literal if it looks like a list
        if column_value.startswith("[") and column_value.endswith("]"):
            try:
                return ast.literal_eval(column_value)
            except (ValueError, SyntaxError):
                # If parsing fails, split by comma
                return [item.strip() for item in column_value.strip("[]").split(",")]
        else:
            # If it's just a single string value
            return [column_value]
    elif isinstance(column_value, list):
        return column_value
    else:
        return []
