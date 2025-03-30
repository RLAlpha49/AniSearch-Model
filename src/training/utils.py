"""
Utility functions for training models.

This module provides utility functions for training and evaluating models.
"""

import ast
import logging
import math
import random
from typing import List, Optional, Tuple, Union

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from src.utils.error_handling import handle_exceptions

# Constants
MODEL_SAVE_PATH = "model/fine-tuned/"

# Default parameters
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 16
DEFAULT_EVAL_STEPS = 500
DEFAULT_WARMUP_STEPS = 500
DEFAULT_MAX_SAMPLES = 10000
DEFAULT_LEARNING_RATE = 2e-6

# Setup logger
logger = logging.getLogger(__name__)


def setup_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: Optional[str] = None) -> str:
    """
    Determine the device to use for computation.

    Args:
        device: Optional device specifier ('cpu', 'cuda', etc.)

    Returns:
        String device identifier
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
    Efficiently truncate multiple text pairs using batch processing.

    Args:
        text_pairs: List of (text_a, text_b) tuples
        tokenizer: Tokenizer to use for encoding
        max_length: Maximum sequence length
        batch_size: Size of batches for processing

    Returns:
        List of truncated (text_a, text_b) tuples
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


def parse_list_column(column_value) -> List[str]:
    """
    Parse a list column from a dataset that may be stored as a string.

    Args:
        column_value: Value from DataFrame, could be string representation of list
                      or already a list

    Returns:
        List of strings
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
