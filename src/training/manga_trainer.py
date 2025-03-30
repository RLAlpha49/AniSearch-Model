"""
Manga-specific trainer class for fine-tuning cross-encoder models.
"""

import logging
import random
from typing import List, Optional

import pandas as pd

from src.utils.constants import MODEL_NAME
from src.training.base_trainer import BaseModelTrainer
from src.training.utils import (
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EVAL_STEPS,
    DEFAULT_WARMUP_STEPS,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_LEARNING_RATE,
)
from src.utils.error_handling import handle_exceptions

# Configure logging
logger = logging.getLogger(__name__)


class MangaModelTrainer(BaseModelTrainer):
    """Class for training cross-encoder models specifically on manga datasets."""

    def __init__(
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
        Initialize the trainer with specified parameters for manga datasets.

        Args:
            model_name: Name of the base cross-encoder model to fine-tune
            epochs: Number of training epochs
            batch_size: Training batch size
            eval_steps: Steps between evaluations
            warmup_steps: Warmup steps for learning rate scheduler
            max_samples: Maximum number of training samples to use
            learning_rate: Learning rate for optimizer
            eval_split: Fraction of data to use for evaluation
            seed: Random seed for reproducibility
            device: Device to use for training ('cpu', 'cuda', or None for auto-detect)
            dataset_path: Path to the dataset file (if None, will use default)
            include_light_novels: Whether to include light novels in the manga dataset
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
        """Filter out light novels from the manga dataset."""
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
            elif isinstance(genres, list):
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
        Create manga-specific variations of base queries.

        Args:
            base_queries: List of original query strings
            n_variations: Number of variations to create per query

        Returns:
            List of query variations
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
