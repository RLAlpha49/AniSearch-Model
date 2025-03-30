"""
Base Search Model for anime/manga description searches using Cross-Encoders.

This module provides a base class with common functionality for searching
anime and manga based on descriptions.
"""

import os
import logging
import re
from typing import List, Dict, Optional, Any, Mapping

import numpy as np
import pandas as pd
import torch
import transformers
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from src.utils.constants import (
    MODEL_NAME,
    NUM_RESULTS,
    DEFAULT_BATCH_SIZE,
    ALTERNATIVE_MODELS,
)
from src.utils.error_handling import handle_exceptions

logger = logging.getLogger(__name__)


class BaseSearchModel:
    """
    Base class for search models using cross-encoders.

    This class provides common functionality for loading datasets and models,
    and performing semantic searches across entries.
    """

    def __init__(
        self,
        dataset_path: str,
        id_column: str,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        dataset_type: str = "base",
    ):
        """
        Initialize the base search model.

        Args:
            dataset_path: Path to the dataset file
            id_column: Name of the ID column in the dataset
            model_name: Name of the cross-encoder model to use
            device: Device to run the model on, either 'cpu', 'cuda', or None (auto-detect)
            dataset_type: Type of dataset ("anime" or "manga")
        """
        # Store the model name and dataset info for later use
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.id_col = id_column
        self.dataset_type = dataset_type

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load the cross-encoder model
        logger.info("Loading cross-encoder model: %s", model_name)
        self._load_model()

        # Load the dataset
        self._load_dataset()

    @handle_exceptions(log_exceptions=True, include_exc_info=True)
    def _load_model(self) -> None:
        """
        Load the cross-encoder model.

        This method handles the loading of the model and sets up the normalization
        behavior based on the model type.
        """
        try:
            # Check if this is an MS Marco model or fine-tuned from one
            # MS Marco models return logits by default
            is_ms_marco = "ms-marco" in self.model_name.lower()
            is_fine_tuned = os.path.isdir(self.model_name) and os.path.exists(
                os.path.join(self.model_name, "config.json")
            )

            # Special case for fine-tuned models
            if is_fine_tuned:
                logger.info(
                    "Loading fine-tuned model from local path: %s", self.model_name
                )
                self.model = CrossEncoder(self.model_name, device=self.device)

                # Check if this was fine-tuned from an MS Marco model
                if "ms-marco" in os.path.basename(self.model_name).lower():
                    logger.info(
                        "Fine-tuned MS Marco model detected, assuming normalized scores (0-1)"
                    )
                    self.normalize_scores = True
                else:
                    logger.info(
                        "Fine-tuned non-MS Marco model detected, using natural model outputs"
                    )
                    self.normalize_scores = False
            elif is_ms_marco:
                # Use sigmoid activation for MS Marco models to get scores between 0 and 1
                self.model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                    activation_fn=torch.nn.Sigmoid(),
                )
                logger.info(
                    "MS Marco model detected, using sigmoid activation "
                    "to normalize scores between 0 and 1"
                )
                self.normalize_scores = True
            else:
                # For other models (like NLI models), use their natural output scale
                self.model = CrossEncoder(self.model_name, device=self.device)
                logger.info("Non-MS Marco model detected, using natural model outputs")
                self.normalize_scores = False

        except ValueError as e:
            error_message = str(e)
            # Check specifically for unsupported model architecture errors
            if "Transformers does not recognize this architecture" in error_message:
                # Extract model type for better error messaging
                model_type_match = re.search(r"model type `([^`]+)`", error_message)
                model_type = (
                    model_type_match.group(1) if model_type_match else "unknown"
                )

                # Get transformers version for diagnostics
                try:
                    transformers_version = transformers.__version__
                    logger.error(
                        "Unsupported model architecture '%s' in model '%s'. "
                        "Your transformers version is %s. "
                        "Try updating with: pip install --upgrade transformers",
                        model_type,
                        self.model_name,
                        transformers_version,
                    )

                    # Recommend alternative models
                    logger.info(
                        "Consider using one of these supported models instead: "
                        "cross-encoder/ms-marco-MiniLM-L-6-v2, "
                        "cross-encoder/ms-marco-TinyBERT-L-6, "
                        "or cross-encoder/stsb-distilroberta-base"
                    )

                    raise ValueError(
                        f"Model architecture '{model_type}' is not supported by your transformers "
                        f"version ({transformers_version}). Please try updating transformers or "
                        f"use a supported model instead. See logs for recommendations."
                    ) from e
                except ImportError as exc:
                    # Fallback if transformers version can't be determined
                    logger.error(
                        "Unsupported model architecture in '%s'. "
                        "Try updating transformers: pip install --upgrade transformers",
                        self.model_name,
                    )
                    raise ValueError(
                        f"Unsupported model architecture in '{self.model_name}'"
                    ) from exc
            else:
                # For other value errors, just log and re-raise
                logger.error(
                    "Failed to load model '%s': %s", self.model_name, error_message
                )
                raise ValueError(
                    f"Could not load model '{self.model_name}'. Error: {error_message}"
                ) from e
        except Exception as e:
            # Handle other types of exceptions
            logger.error("Failed to load model '%s': %s", self.model_name, str(e))
            raise ValueError(
                f"Could not load model '{self.model_name}'. Error: {str(e)}"
            ) from e

    @handle_exceptions(log_exceptions=True, include_exc_info=True)
    def _load_dataset(self) -> None:
        """
        Load and prepare the dataset.

        Loads the merged dataset and extracts relevant columns for searching.
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}. "
                f"Run 'python src/merge_datasets.py' first."
            )

        logger.info("Loading dataset from %s", self.dataset_path)
        self.df = pd.read_csv(self.dataset_path)
        logger.info("Loaded %d entries", len(self.df))

        # Extract synopsis columns - these will be used for searching
        self.synopsis_cols = [
            col for col in self.df.columns if "synopsis" in col.lower()
        ]
        logger.info(
            "Found %d synopsis columns: %s", len(self.synopsis_cols), self.synopsis_cols
        )

        # Prepare the document corpus by combining all synopsis columns
        self.df["combined_synopsis"] = self.df.apply(
            lambda row: " ".join(
                [str(row[col]) for col in self.synopsis_cols if pd.notna(row[col])]
            ),
            axis=1,
        )

        # Remove entries with empty combined synopsis
        initial_count = len(self.df)
        self.df = self.df[self.df["combined_synopsis"].str.strip() != ""]
        logger.info(
            "Removed %d entries with empty synopsis", initial_count - len(self.df)
        )

        # Ensure essential columns exist
        required_cols = [self.id_col, "title", "combined_synopsis"]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    @handle_exceptions(log_exceptions=True, include_exc_info=True)
    def search(
        self,
        query: str,
        num_results: int = NUM_RESULTS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> List[Dict[str, Any]]:
        """
        Search for entries matching the provided description.

        Args:
            query: The search query or description to match
            num_results: Number of top results to return
            batch_size: Batch size for processing with the cross-encoder model

        Returns:
            List of dictionaries containing matched entries with scores
        """
        if not query.strip():
            raise ValueError("Search query cannot be empty")

        logger.info("Searching for: %s", query)

        # Prepare pairs for cross-encoder scoring
        all_synopses = self.df["combined_synopsis"].tolist()
        sentence_pairs = [(query, text) for text in all_synopses]

        # Calculate total number of pairs and batches
        total_pairs = len(sentence_pairs)
        if batch_size <= 0:
            batch_size = DEFAULT_BATCH_SIZE
            logger.warning(
                "Invalid batch size provided, using default: %d", DEFAULT_BATCH_SIZE
            )

        # Compute relevance scores in batches
        logger.info(
            "Computing relevance scores with cross-encoder (batch size: %d)", batch_size
        )
        scores = []

        # Use tqdm to display progress for all datasets
        with tqdm(
            total=total_pairs,
            desc="Scoring",
            disable=False,
        ) as pbar:
            for i in range(0, total_pairs, batch_size):
                batch = sentence_pairs[i : i + batch_size]
                # Disable progress_bar in the model's predict method to avoid multiple progress bars
                batch_scores = self.model.predict(batch, show_progress_bar=False)

                if isinstance(batch_scores, np.ndarray):
                    scores.extend(batch_scores.tolist())
                else:
                    scores.extend(batch_scores)

                pbar.update(len(batch))

        # Convert scores to numpy array
        scores = np.array(scores)

        # Get indices of top scores
        top_indices = scores.argsort()[-num_results:][::-1]

        # Prepare results
        results = []
        for idx in top_indices:
            entry = self.df.iloc[idx]
            synopsis = entry["combined_synopsis"]
            results.append(
                {
                    "id": entry[self.id_col],
                    "title": entry["title"],
                    "score": float(scores[idx]),
                    "synopsis": (
                        synopsis[:500] + "..." if len(synopsis) > 500 else synopsis
                    ),
                }
            )

        logger.info("Found %d matches", len(results))
        return results

    @staticmethod
    @handle_exceptions(log_exceptions=True, include_exc_info=True)
    def list_available_models() -> Mapping[str, Dict[str, str]]:
        """
        List available cross-encoder models that can be used with this search model.

        Returns:
            Dictionary of model categories and their corresponding model names
        """
        return ALTERNATIVE_MODELS

    @staticmethod
    @handle_exceptions(log_exceptions=True, include_exc_info=True)
    def list_fine_tuned_models() -> Dict[str, str]:
        """
        List available fine-tuned models in the model directory.

        Returns:
            Dictionary of model names and their paths
        """
        fine_tuned_models = {}
        model_dir = "model/fine-tuned"

        if not os.path.exists(model_dir):
            logger.warning("Fine-tuned model directory not found: %s", model_dir)
            return fine_tuned_models

        for model_name in os.listdir(model_dir):
            model_path = os.path.join(model_dir, model_name)
            if os.path.isdir(model_path) and os.path.exists(
                os.path.join(model_path, "config.json")
            ):
                fine_tuned_models[model_name] = model_path

        return fine_tuned_models
