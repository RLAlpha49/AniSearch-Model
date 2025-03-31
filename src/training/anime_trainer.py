"""
# Anime Model Trainer

Specialized trainer class for fine-tuning cross-encoder models specifically
for anime search applications.

This module extends the base trainer framework with anime-specific functionality, including
anime-focused query variations and optimizations tailored for anime datasets. It simplifies
the process of creating specialized search models that better understand anime terminology,
titles, and content.

## Features

- Pre-configured to work with anime datasets
- Anime-specific query templates for training data generation
- Specialized handling of anime terminology and search patterns
- Inherits all capabilities from the base trainer framework
- Simplified initialization with anime-specific defaults

## Usage Context

The anime trainer is designed for:

1. Creating specialized search models for anime content
2. Fine-tuning models to better understand anime-specific terminology
3. Generating realistic anime search query variations
4. Training models that provide more relevant results for anime searches

For general training capabilities, see the base trainer documentation.
"""

import logging
import random
from typing import List, Optional

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


class AnimeModelTrainer(BaseModelTrainer):
    """
    Specialized trainer for fine-tuning cross-encoder models on anime datasets.

    This class extends the BaseModelTrainer with anime-specific functionality,
    simplifying the creation of search models optimized for anime content. It
    automatically configures the training process for anime datasets and provides
    anime-specific query generation for more robust training.

    The trainer creates relevant training examples using anime titles and synopses,
    and generates query variations that reflect how users typically search for anime
    content (e.g., "Looking for anime about...", "Anime similar to...").

    Attributes:
        dataset_type (str): Fixed to "anime" to specify this trainer works with anime datasets.
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
        df (pandas.DataFrame): The loaded anime dataset after preparation.

    Example:
        ```python
        # Initialize a trainer for anime model
        trainer = AnimeModelTrainer(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            epochs=5,
            batch_size=16
        )

        # Train the model with MSE loss and linear scheduler
        model_path = trainer.train(loss_type="mse", scheduler="linear")
        print(f"Anime search model saved to: {model_path}")

        # Create labeled data for inspection
        trainer.create_and_save_labeled_data(
            output_file="anime_labeled_data.csv",
            n_samples=5000
        )
        ```

    Notes:
        - The trainer automatically uses the default anime dataset path unless specified
        - For best results, ensure your anime dataset contains adequate synopsis information
          and metadata like genres and themes
        - This class sets dataset_type="anime" in the parent class, focusing all operations
          on anime data
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
    ):
        """
        Initialize the anime-specific trainer with configuration parameters.

        This constructor sets up the training environment specifically for anime data,
        passing "anime" as the dataset_type to the parent class. It configures all
        training parameters and loads the appropriate anime dataset.

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

            max_samples: Maximum number of training samples to use from the anime dataset.
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

            dataset_path: Path to the anime dataset file. If None, uses the default
                anime dataset path. Default is None.

        Notes:
            - This constructor passes "anime" as the dataset_type to the parent class
            - The method automatically creates the output directory if it doesn't exist
            - The output path is constructed from the model name and "anime"
            - After initialization, the anime dataset is prepared for training
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
        Create anime-specific variations of base queries to improve training robustness.

        This method overrides the parent class implementation to generate query variations
        specifically tailored for anime search, using templates that reflect how users
        typically search for anime content (e.g., "Looking for anime about...",
        "Anime similar to...").

        The variations help the model learn to recognize the same anime-related intent
        expressed in different ways, making it more robust to real-world search queries.

        Args:
            base_queries: List of original query strings (typically anime titles or
                descriptions) that will be used as the basis for generating variations.

            n_variations: Number of anime-specific variations to create for each base
                query. If this exceeds the number of available templates, all templates
                will be used. Default is 7.

        Returns:
            List[str]: A combined list containing both the original queries and their
                anime-specific variations. The length will be approximately
                len(base_queries) * (1 + n_variations), but may be less if n_variations
                exceeds the number of available templates.

        Example:
            ```python
            # Create variations of anime titles
            titles = ["Naruto", "One Piece", "Attack on Titan"]
            trainer = AnimeModelTrainer()
            variations = trainer.create_query_variations(titles, n_variations=3)

            # Print all variations
            for var in variations:
                print(var)
            # Example output:
            # Naruto
            # Looking for anime about Naruto
            # I want to watch anime with Naruto
            # Find me anime where Naruto
            # One Piece
            # ...etc.
            ```

        Notes:
            - The method always includes the original queries in the returned list
            - Templates are selected randomly for each query
            - All templates include the word "anime" to help the model recognize
              anime-specific search patterns
            - This anime-specific implementation provides better training examples
              than the generic implementation in the parent class
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
