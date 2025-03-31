"""
# Base Search Model

A foundational framework for semantic search of anime and manga using cross-encoder models.

This module provides a base class with common functionality for semantic search
across anime and manga datasets using cross-encoder models. It handles dataset loading,
model initialization, and efficient similarity scoring between user queries and content
descriptions.

## Features

- Flexible cross-encoder model loading with error handling
- Automatic device detection (CPU/CUDA)
- Efficient batch processing for large datasets
- Progress tracking for search operations
- Support for various model architectures with appropriate score normalization
- Utilities for listing available pre-trained and fine-tuned models

## Architecture

The `BaseSearchModel` serves as the foundation for specialized search models like
`AnimeSearchModel` and `MangaSearchModel`. It implements core functionality including:

1. Dataset loading and preprocessing
2. Cross-encoder model initialization
3. Semantic similarity computation
4. Result ranking and formatting

## Model Compatibility

The implementation supports various types of cross-encoder models:
- MS Marco-based models (with score normalization)
- Non-MS Marco models (using natural outputs)
- Locally fine-tuned models

## Error Handling

The implementation includes comprehensive error handling for common issues:
- Missing datasets
- Unsupported model architectures
- GPU/CPU compatibility issues
- Empty search queries
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
    Base class for cross-encoder powered semantic search models.

    This class provides the foundation for building specialized search models that use
    cross-encoder architectures to compute semantic similarity between user queries
    and content descriptions (synopses). It handles the common functionality such as
    dataset loading, model initialization, and search computation.

    The class is designed to be extended by specialized search models for different
    content types (e.g., anime, manga) that can implement additional domain-specific
    functionality while reusing the core search capabilities.

    Attributes:
        model_name (str): Name or path of the cross-encoder model being used
        dataset_path (str): Path to the dataset file
        id_col (str): Name of the ID column in the dataset
        dataset_type (str): Type of dataset ("anime" or "manga")
        device (str): Device being used for computation ('cpu', 'cuda', etc.)
        model (CrossEncoder): The loaded cross-encoder model
        df (pd.DataFrame): The loaded and preprocessed dataset
        synopsis_cols (List[str]): List of column names containing synopsis text
        normalize_scores (bool): Whether model scores need normalization
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
        Initialize the base search model with dataset and model configuration.

        This constructor sets up the search model by:

        1. Initializing configuration parameters
        2. Detecting or setting the compute device (CPU/CUDA)
        3. Loading the cross-encoder model
        4. Loading and preprocessing the dataset

        Args:
            dataset_path: Path to the dataset CSV file containing entries to search.
                The file should contain at minimum ID, title, and synopsis columns.
            id_column: Name of the column containing unique identifiers in the dataset.
                This will be used to reference specific entries in search results.
            model_name: Name or path of the cross-encoder model to use.
                Can be a Hugging Face model name or local path to a fine-tuned model.
                Defaults to the value specified in constants.MODEL_NAME.
            device: Device to run the model on ('cpu', 'cuda', 'cuda:0', etc.).
                If None, automatically selects the best available device.
            dataset_type: Type of dataset being loaded, used for logging.
                Common values are "anime" or "manga".

        Raises:
            FileNotFoundError: If the dataset file cannot be found
            ValueError: If the model_name is invalid or the model cannot be loaded

        Example:
            ```python
            # Create a basic search model with default settings
            basic_search = BaseSearchModel(
                dataset_path="data/merged_anime_dataset.csv",
                id_column="anime_id",
                dataset_type="anime"
            )

            # Create a search model with custom model and device
            custom_search = BaseSearchModel(
                dataset_path="data/merged_manga_dataset.csv",
                id_column="manga_id",
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                device="cuda:0",
                dataset_type="manga"
            )
            ```
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
        Load and initialize the cross-encoder model for semantic search.

        This method handles the loading of the cross-encoder model with appropriate
        configuration based on the model type. It determines whether score normalization
        is needed and handles various error conditions that might occur during loading.

        The method specifically accounts for:

        1. MS Marco models (requiring sigmoid activation for score normalization)
        2. Fine-tuned models (loaded from local paths)
        3. Standard cross-encoder models (using their natural output scale)

        Raises:
            ValueError: If the model fails to load due to:
                - Unsupported model architecture
                - Incompatible transformers version
                - Missing model files
                - Other initialization errors

        Notes:
            - MS Marco models return unbounded logits and need sigmoid normalization
            - Fine-tuned models are detected by looking for a config.json file
            - The method sets self.normalize_scores based on the model type
            - For unsupported model architectures, alternative models are suggested
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
        Load and preprocess the dataset for semantic search.

        This method loads the dataset from the specified path and prepares it for
        search operations by:

        1. Loading the dataset from CSV
        2. Identifying all synopsis columns
        3. Creating a combined synopsis field from all available synopsis columns
        4. Filtering out entries with empty synopses
        5. Validating that all required columns are present

        The preprocessing steps ensure that the dataset is ready for efficient
        semantic search with the cross-encoder model.

        Raises:
            FileNotFoundError: If the dataset file does not exist
            ValueError: If required columns are missing from the dataset

        Notes:
            - Synopsis columns are identified by having "synopsis" in their name
            - Multiple synopsis columns are combined to provide richer context
            - Entries with empty combined synopses are filtered out
            - Required columns include the ID column, title, and combined_synopsis
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
        Search for entries matching the provided description or query.

        This method performs semantic search across the dataset by computing similarity
        scores between the user query and all synopses in the dataset. It returns the
        top matches sorted by relevance score.

        The search process includes:

        1. Creating sentence pairs between the query and all synopses
        2. Computing relevance scores using the cross-encoder model in batches
        3. Sorting results by score and returning the top matches

        Args:
            query: The search query or description to match against synopses.
                This should be a descriptive text that captures the content
                the user is looking for.
            num_results: Number of top matches to return, sorted by relevance score.
                Defaults to the value specified in constants.NUM_RESULTS.
            batch_size: Number of sentence pairs to process at once with the model.
                Using batches helps manage memory usage with large datasets.
                Defaults to the value specified in constants.DEFAULT_BATCH_SIZE.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing:
                - id: The entry ID from the id_column specified during initialization
                - title: The entry title
                - score: The relevance score (higher is better)
                - synopsis: A preview of the entry synopsis (truncated to 500 chars)

                The list is sorted by score in descending order.

        Raises:
            ValueError: If the query is empty or consists only of whitespace

        Example:
            ```python
            # Initialize a search model
            search_model = BaseSearchModel(
                dataset_path="data/merged_anime_dataset.csv",
                id_column="anime_id"
            )

            # Search for content about time travel
            results = search_model.search(
                query="A story where characters travel through time and change history",
                num_results=5,
                batch_size=64
            )

            # Process the top results
            for result in results:
                print(f"{result['title']} (Score: {result['score']:.2f})")
                print(f"Synopsis: {result['synopsis'][:100]}...")
            ```
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
        scores: List[Any] = []

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
        scores_array = np.array(scores)

        # Get indices of top scores
        top_indices = scores_array.argsort()[-num_results:][::-1]

        # Prepare results
        results = []
        for idx in top_indices:
            entry = self.df.iloc[idx]
            synopsis = entry["combined_synopsis"]
            results.append(
                {
                    "id": entry[self.id_col],
                    "title": entry["title"],
                    "score": float(scores_array[idx]),
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
        List available pre-trained cross-encoder models categorized by type.

        This static method returns a dictionary of model categories and their
        corresponding model recommendations that can be used with the search system.
        These models are defined in the ALTERNATIVE_MODELS constant.

        Returns:
            Mapping[str, Dict[str, str]]: A dictionary where:
                - Keys are model categories (e.g., "Semantic Search", "Question Answering")
                - Values are dictionaries mapping model names to descriptions

        Example:
            ```python
            # Get a dictionary of available models by category
            available_models = BaseSearchModel.list_available_models()

            # Print model categories and models
            for category, models in available_models.items():
                print(f"\n{category}:")
                for model_name, description in models.items():
                    print(f"  - {model_name}: {description}")
            ```
        """
        return ALTERNATIVE_MODELS

    @staticmethod
    @handle_exceptions(log_exceptions=True, include_exc_info=True)
    def list_fine_tuned_models() -> Dict[str, str]:
        """
        List locally available fine-tuned models that can be used for search.

        This static method scans the fine-tuned model directory to find models
        that have been fine-tuned specifically for anime/manga search. It identifies
        valid models by checking for the presence of a config.json file.

        Returns:
            Dict[str, str]: A dictionary mapping:
                - Keys: Model directory names
                - Values: Full paths to the model directories

        Notes:
            - Searches in the "model/fine-tuned" directory by default
            - Only directories containing a config.json file are included
            - Returns an empty dictionary if no fine-tuned models are found

        Example:
            ```python
            # Get a dictionary of available fine-tuned models
            fine_tuned_models = BaseSearchModel.list_fine_tuned_models()

            if fine_tuned_models:
                print("Available fine-tuned models:")
                for name, path in fine_tuned_models.items():
                    print(f"- {name}: {path}")
            else:
                print("No fine-tuned models found.")
            ```
        """
        fine_tuned_models: Dict[str, str] = {}
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
