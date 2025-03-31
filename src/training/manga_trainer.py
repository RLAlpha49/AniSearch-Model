"""
# Manga Model Trainer

Specialized trainer class for fine-tuning cross-encoder models specifically for
manga search applications.

This module extends the base trainer framework with manga-specific functionality, including
manga-focused query variations and optimizations tailored for manga datasets. It simplifies
the process of creating specialized search models that better understand manga terminology,
titles, and content.

## Features

- Pre-configured to work with manga datasets
- Manga-specific query templates for training data generation
- Optional filtering of light novels from training data
- Specialized handling of manga terminology and search patterns
- Inherits all capabilities from the base trainer framework
- Simplified initialization with manga-specific defaults

## Usage Context

The manga trainer is designed for:

1. Creating specialized search models for manga content
2. Fine-tuning models to better understand manga-specific terminology
3. Generating realistic manga search query variations
4. Training models that provide more relevant results for manga searches
5. Optionally excluding light novels for manga-focused models

For general training capabilities, see the base trainer documentation.
"""

import logging
import random
from typing import List, Optional

import pandas as pd

from src.training.base_trainer import BaseModelTrainer
from src.training.utils import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_EVAL_STEPS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_WARMUP_STEPS,
)
from src.utils.constants import MODEL_NAME
from src.utils.error_handling import handle_exceptions

# Configure logging
logger = logging.getLogger(__name__)


class MangaModelTrainer(BaseModelTrainer):
    """
    Specialized trainer for fine-tuning cross-encoder models on manga datasets.

    This class extends the BaseModelTrainer with manga-specific functionality,
    simplifying the creation of search models optimized for manga content. It
    automatically configures the training process for manga datasets and provides
    manga-specific query generation for more robust training.

    The trainer creates relevant training examples using manga titles and synopses,
    and generates query variations that reflect how users typically search for manga
    content (e.g., "Looking for manga about...", "Manga similar to...").

    Attributes:
        dataset_type (str): Fixed to "manga" to specify this trainer works with manga datasets.
        include_light_novels (bool): Flag indicating whether light novels should be
            included in the training dataset. When False, light novels are filtered out.
        model_name (str): Name of the base cross-encoder model used for fine-tuning.
        epochs (int): Number of training epochs.
        batch_size (int): Number of examples processed in each training step.
        eval_steps (int): Number of steps between model evaluations.
        warmup_steps (int): Number of warmup steps for the learning rate scheduler.
        max_samples (int): Maximum number of training samples to use.
        learning_rate (float): Learning rate for the optimizer.
        eval_split (float): Fraction of data used for evaluation.
        seed (int): Random seed for reproducibility.
        device (str): Device used for training (cpu or cuda).
        df (pandas.DataFrame): The loaded manga dataset after preparation.

    Example:
        ```python
        # Initialize a trainer for manga model, excluding light novels
        trainer = MangaModelTrainer(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            epochs=5,
            batch_size=16,
            include_light_novels=False
        )

        # Train the model with MSE loss and linear scheduler
        model_path = trainer.train(loss_type="mse", scheduler="linear")
        print(f"Manga search model saved to: {model_path}")

        # Create labeled data for inspection
        trainer.create_and_save_labeled_data(
            output_file="manga_labeled_data.csv",
            n_samples=5000
        )
        ```

    Notes:
        - The trainer automatically uses the default manga dataset path unless specified
        - For best results, ensure your manga dataset contains adequate synopsis information
          and metadata like genres and themes
        - This class sets dataset_type="manga" in the parent class, focusing all operations
          on manga data
        - Light novels can be excluded from training to create more manga-specific models
    """

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        model_name: str = MODEL_NAME,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        eval_steps: int = DEFAULT_EVAL_STEPS,
        warmup_steps: int = DEFAULT_WARMUP_STEPS,
        max_samples: int = DEFAULT_MAX_SAMPLES,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        eval_split: float = 0.1,
        seed: int = 42,
        device: Optional[str] = None,
        dataset_path: Optional[str] = None,
        include_light_novels: bool = False,
    ):
        """
        Initialize the manga-specific trainer with configuration parameters.

        This constructor sets up the training environment specifically for manga data,
        passing "manga" as the dataset_type to the parent class. It configures all
        training parameters, loads the appropriate manga dataset, and optionally filters
        out light novels from the dataset.

        Args:
            model_name: The name or path of the base cross-encoder model to fine-tune.
                Can be a HuggingFace model identifier or a local path. Default is the
                value from MODEL_NAME constant.

            epochs: Number of complete passes through the training dataset. Higher values
                may improve performance but risk overfitting. Default is DEFAULT_EPOCHS (3).

            batch_size: Number of examples processed in each training step. Larger batches
                provide more stable gradients but require more memory. Default is
                DEFAULT_BATCH_SIZE (16).

            eval_steps: Number of training steps between model evaluations. If not specified,
                a reasonable value will be calculated based on dataset size. Default is
                DEFAULT_EVAL_STEPS (500).

            warmup_steps: Number of steps for learning rate warm-up. During warm-up, the
                learning rate gradually increases from 0 to the specified rate. Default
                is DEFAULT_WARMUP_STEPS (500).

            max_samples: Maximum number of training samples to use from the manga dataset.
                Useful for limiting training time or for testing. Set to None to use all
                available data. Default is DEFAULT_MAX_SAMPLES (10000).

            learning_rate: Learning rate for the optimizer. Controls how quickly model
                weights are updated during training. Default is DEFAULT_LEARNING_RATE (2e-6).

            eval_split: Fraction of data to use for evaluation instead of training.
                Must be between 0 and 1. Default is 0.1 (10% for evaluation).

            seed: Random seed for reproducibility. Ensures the same training/evaluation
                split and data sampling across runs. Default is 42.

            device: Device to use for training ('cpu', 'cuda', 'cuda:0', etc.). If None,
                automatically selects GPU if available, otherwise CPU. Default is None.

            dataset_path: Path to the manga dataset file. If None, uses the default
                manga dataset path. Default is None.

            include_light_novels: Whether to include light novels in the manga dataset.
                When False, entries identified as light novels based on their genres will
                be filtered out. Default is False.

        Notes:
            - This constructor passes "manga" as the dataset_type to the parent class
            - The method automatically creates the output directory if it doesn't exist
            - The output path is constructed from the model name and "manga"
            - After initialization, the manga dataset is prepared for training
            - If include_light_novels is False, light novels will be filtered from the dataset
        """
        super().__init__(
            dataset_type="manga",
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            eval_steps=eval_steps,
            warmup_steps=warmup_steps,
            max_samples=max_samples,
            learning_rate=learning_rate,
            eval_split=eval_split,
            seed=seed,
            device=device,
            dataset_path=dataset_path,
        )
        self.include_light_novels = include_light_novels
        logger.info(
            "Initialized MangaModelTrainer with include_light_novels=%s",
            include_light_novels,
        )

        # Filter light novels if necessary
        if not self.include_light_novels:
            self._filter_light_novels()

    @handle_exceptions(log_exceptions=True, include_exc_info=True)
    def _filter_light_novels(self) -> None:
        """
        Filter out light novels from the manga dataset based on genre information.

        This private method examines the 'genres' column in the dataset to identify and
        remove entries that are categorized as light novels. It handles both string and
        list representations of genres, and accounts for missing genre data.

        The method modifies the dataset in-place by removing entries identified as
        light novels, keeping only true manga entries for training.

        Returns:
            None: The method modifies self.df in-place.

        Raises:
            No exceptions are raised as the method is decorated with handle_exceptions,
            which logs any errors without interrupting execution.

        Notes:
            - Requires a 'genres' column in the dataset to function properly
            - If the 'genres' column is missing, a warning is logged and no filtering occurs
            - Identifies light novels by checking for 'light novel' or 'novel' in the genres
            - Works with both string and list representations of genre data
            - Logs the number of entries filtered and remaining for transparency
            - If no light novels are found, a log message is generated but no filtering occurs

        Example:
            ```python
            # This method is called automatically when include_light_novels=False
            trainer = MangaModelTrainer(include_light_novels=False)

            # The method can also be called manually if needed
            trainer._filter_light_novels()
            ```
        """
        if "genres" not in self.df.columns:
            logger.warning(
                "Cannot filter light novels: 'genres' column not found in dataset"
            )
            return

        # Create a function to check if a row contains light novel genres
        def is_light_novel(row):
            if pd.isna(row["genres"]):
                return False

            genres = row["genres"]
            if isinstance(genres, str):
                genres_lower = genres.lower()
                return "light novel" in genres_lower or "novel" in genres_lower
            if isinstance(genres, list):
                genres_lower = [g.lower() if isinstance(g, str) else "" for g in genres]
                return any("light novel" in g or "novel" in g for g in genres_lower)

            return False

        # Apply the filter
        light_novel_mask = self.df.apply(is_light_novel, axis=1)
        light_novel_count = light_novel_mask.sum()

        if light_novel_count > 0:
            self.df = self.df[~light_novel_mask]
            logger.info(
                "Filtered out %d light novels, leaving %d manga entries",
                light_novel_count,
                len(self.df),
            )
        else:
            logger.info("No light novels found in the dataset")

    @handle_exceptions(log_exceptions=True, include_exc_info=True)
    def create_query_variations(
        self, base_queries: List[str], n_variations: int = 7
    ) -> List[str]:
        """
        Create manga-specific variations of base queries to improve training robustness.

        This method overrides the parent class implementation to generate query variations
        specifically tailored for manga search, using templates that reflect how users
        typically search for manga content (e.g., "Looking for manga about...",
        "Manga similar to...").

        The variations help the model learn to recognize the same manga-related intent
        expressed in different ways, making it more robust to real-world search queries.

        Args:
            base_queries: List of original query strings (typically manga titles or
                descriptions) that will be used as the basis for generating variations.

            n_variations: Number of manga-specific variations to create for each base
                query. If this exceeds the number of available templates, all templates
                will be used. Default is 7.

        Returns:
            List[str]: A combined list containing both the original queries and their
                manga-specific variations. The length will be approximately
                len(base_queries) * (1 + n_variations), but may be less if n_variations
                exceeds the number of available templates.

        Example:
            ```python
            # Create variations of manga titles
            titles = ["One Piece", "Berserk", "Chainsaw Man"]
            trainer = MangaModelTrainer()
            variations = trainer.create_query_variations(titles, n_variations=3)

            # Print all variations
            for var in variations:
                print(var)
            # Example output:
            # One Piece
            # Looking for manga about One Piece
            # I want to read manga with One Piece
            # Find me manga where One Piece
            # Berserk
            # ...etc.
            ```

        Notes:
            - The method always includes the original queries in the returned list
            - Templates are selected randomly for each query
            - All templates include the word "manga" to help the model recognize
              manga-specific search patterns
            - This manga-specific implementation provides better training examples
              than the generic implementation in the parent class
            - The manga templates focus on reading rather than watching (compared to anime)
        """
        # Add manga-specific templates
        manga_templates = [
            "Looking for manga about {query}",
            "I want to read manga with {query}",
            "Find me manga where {query}",
            "Can you recommend manga that has {query}",
            "What manga is about {query}",
            "Manga similar to {query}",
            "{query} manga recommendation",
            "I'm looking for manga with {query}",
            "I'm searching for manga with {query}",
            "I'm trying to find manga with {query}",
            "What manga should I read if I like {query}",
            "Good manga about {query}",
        ]

        variations = []
        for query in base_queries:
            # Add the original query
            variations.append(query)

            # Select n_variations randomly from manga-specific templates
            n_to_use = min(n_variations, len(manga_templates))
            selected_templates = random.sample(manga_templates, n_to_use)
            for template in selected_templates:
                variations.append(template.format(query=query))

        return variations
