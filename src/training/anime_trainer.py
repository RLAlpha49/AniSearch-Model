"""
Anime-specific trainer class for fine-tuning cross-encoder models.
"""

import logging
import random
from typing import List, Optional

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


class AnimeModelTrainer(BaseModelTrainer):
    """Class for training cross-encoder models specifically on anime datasets."""

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
    ):
        """
        Initialize the trainer with specified parameters for anime datasets.

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
        """
        super().__init__(
            dataset_type="anime",
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
        logger.info("Initialized AnimeModelTrainer")

    @handle_exceptions(log_exceptions=True, include_exc_info=True)
    def create_query_variations(
        self, base_queries: List[str], n_variations: int = 7
    ) -> List[str]:
        """
        Create anime-specific variations of base queries.

        Args:
            base_queries: List of original query strings
            n_variations: Number of variations to create per query

        Returns:
            List of query variations
        """
        # Add anime-specific templates
        anime_templates = [
            "Looking for anime about {query}",
            "I want to watch anime with {query}",
            "Find me anime where {query}",
            "Can you recommend anime that has {query}",
            "What anime is about {query}",
            "Anime similar to {query}",
            "{query} anime recommendation",
            "I'm looking for anime with {query}",
            "I'm searching for anime with {query}",
            "I'm trying to find anime with {query}",
        ]

        variations = []
        for query in base_queries:
            # Add the original query
            variations.append(query)

            # Select n_variations randomly from anime-specific templates
            n_to_use = min(n_variations, len(anime_templates))
            selected_templates = random.sample(anime_templates, n_to_use)
            for template in selected_templates:
                variations.append(template.format(query=query))

        return variations
