"""
Anime/Manga Cross-Encoder Model Training

This script provides functionality to fine-tune cross-encoder models for improved
anime/manga description matching. It implements training on labelled data and
can be used to create domain-specific models optimized for anime/manga content.

Usage:
    python train_model.py --type anime --model "cross-encoder/ms-marco-MiniLM-L-6-v2" --epochs 3
    python train_model.py --type manga --create-synthetic-data --epochs 5

The fine-tuned model will be saved to model/fine-tuned/{model_name}-{dataset_type}-finetuned
"""

# pylint: disable=too-many-lines

import os
import argparse
import ast
import logging
import math
import random
import sys
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from sentence_transformers import InputExample, LoggingHandler
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import (
    CrossEncoderCorrelationEvaluator,
)
from sentence_transformers.cross_encoder.losses import (
    BinaryCrossEntropyLoss,
    CrossEntropyLoss,
    LambdaLoss,
    ListMLELoss,
    PListMLELoss,
    ListNetLoss,
    MultipleNegativesRankingLoss,
    CachedMultipleNegativesRankingLoss,
    MSELoss,
    MarginMSELoss,
    RankNetLoss,
)
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import (
    CrossEncoderTrainingArguments,
)
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# pylint: disable=wrong-import-position
from src.search_model import (
    ANIME_DATASET_PATH,
    MANGA_DATASET_PATH,
    MODEL_NAME,
    ALTERNATIVE_MODELS,
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
logger = logging.getLogger(__name__)

# Default parameters
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 16
DEFAULT_EVAL_STEPS = 500
DEFAULT_WARMUP_STEPS = 500
DEFAULT_MAX_SAMPLES = 10000  # Max samples to use for training
DEFAULT_LEARNING_RATE = 2e-6
MODEL_SAVE_PATH = "model/fine-tuned/"


# Simple dataset class for training
class SimpleDataset(Dataset):
    """Simple dataset that just wraps a list of examples"""

    def __init__(self, examples: List[Any]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Any:
        return self.examples[idx]


# Wrapper class for CrossEncoderCorrelationEvaluator to match type expectations
class CEEvaluatorWrapper(SentenceEvaluator):
    """
    Wrapper around CrossEncoderCorrelationEvaluator that explicitly implements
    SentenceEvaluator
    """

    def __init__(self, evaluator: CrossEncoderCorrelationEvaluator):
        super().__init__()
        self.evaluator = evaluator

    def __call__(
        self,
        model: CrossEncoder,
        output_path: Optional[str] = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> float:
        """
        Call the wrapped evaluator

        Args:
            model: Model to evaluate
            output_path: Path where results will be saved
            epoch: Current epoch
            steps: Current steps

        Returns:
            Evaluation score
        """
        # Handle None case explicitly to satisfy type checking
        actual_output_path = output_path if output_path is not None else ""
        # The evaluator returns a dictionary with metrics, extract the main
        # metric (spearman correlation)
        result = self.evaluator(model, actual_output_path, epoch, steps)
        # Extract the main metric (spearman correlation) or return 0.0 if not available
        main_score = result.get("spearman", 0.0) if isinstance(result, dict) else 0.0
        return main_score


class AnimeModelTrainer:
    """Class for training cross-encoder models on anime/manga datasets."""

    def __init__(
        self,
        dataset_type: str = "anime",
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
    ):
        """
        Initialize the trainer with specified parameters.

        Args:
            dataset_type: Type of dataset to use ('anime' or 'manga')
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
        """
        self.dataset_type = dataset_type.lower()
        if self.dataset_type not in ["anime", "manga"]:
            raise ValueError("Dataset type must be either 'anime' or 'manga'")

        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.eval_steps = eval_steps
        self.warmup_steps = warmup_steps
        self.max_samples = max_samples
        self.learning_rate = learning_rate
        self.eval_split = eval_split
        self.seed = seed

        # Track whether eval_steps was explicitly set
        self.eval_steps_specified = eval_steps != DEFAULT_EVAL_STEPS

        # Track whether warmup_steps was explicitly set
        self.warmup_steps_specified = warmup_steps != DEFAULT_WARMUP_STEPS

        # Fix random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info("Using device: %s", self.device)

        # Set model save path
        if not os.path.exists(MODEL_SAVE_PATH):
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

        model_basename = os.path.basename(model_name.replace("/", "-"))
        self.output_path = os.path.join(
            MODEL_SAVE_PATH, f"{model_basename}-{dataset_type}-finetuned"
        )

        # Load dataset
        self.dataset_path = (
            ANIME_DATASET_PATH if dataset_type == "anime" else MANGA_DATASET_PATH
        )
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}. "
                f"Run 'python src/merge_datasets.py --type {dataset_type}' first."
            )

        logger.info("Loading dataset from: %s", self.dataset_path)
        self.df = pd.read_csv(self.dataset_path)
        logger.info("Loaded %d entries", len(self.df))

        # Prepare dataset for training
        self._prepare_dataset()

    def _prepare_dataset(self) -> None:
        """Prepare the dataset for training by extracting and combining synopses."""
        # Extract synopsis columns
        self.synopsis_cols = [
            col for col in self.df.columns if "synopsis" in col.lower()
        ]
        logger.info(
            "Found %d synopsis columns: %s", len(self.synopsis_cols), self.synopsis_cols
        )

        # Combine synopses and filter empty entries
        self.df["combined_synopsis"] = self.df.apply(
            lambda row: " ".join(
                [str(row[col]) for col in self.synopsis_cols if pd.notna(row[col])]
            ),
            axis=1,
        )

        # Keep only entries with non-empty synopses
        initial_count = len(self.df)
        self.df = self.df[self.df["combined_synopsis"].str.strip() != ""]
        logger.info(
            "Removed %d entries with empty synopsis", initial_count - len(self.df)
        )

        # Set id column based on dataset type
        self.id_col = f"{self.dataset_type}_id"

        # Ensure essential columns exist
        required_cols = [self.id_col, "title", "combined_synopsis"]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    def create_synthetic_training_data(self) -> List[InputExample]:
        """
        Create synthetic training data for cross-encoder training.

        This function creates positive and negative training pairs:
        - Positive pairs: Original synopsis with title as query
        - Negative pairs: Random synopses with unrelated titles

        Returns:
            List of InputExample objects for training
        """
        logger.info("Creating synthetic training data")

        # Limit dataset size if needed
        df_sample = self.df.sample(
            min(len(self.df), self.max_samples), random_state=self.seed
        )
        examples = []

        for idx, row in tqdm(
            df_sample.iterrows(), total=len(df_sample), desc="Creating training pairs"
        ):
            title = str(row["title"]) if not pd.isna(row["title"]) else ""
            synopsis = (
                str(row["combined_synopsis"])
                if not pd.isna(row["combined_synopsis"])
                else ""
            )

            # Skip entries with empty titles or synopses
            if not title or not synopsis:
                continue

            # Create positive pair (score 1.0)
            examples.append(InputExample(texts=[title, synopsis], label=1.0))

            # For each positive pair, create 3 negative pairs (score 0.0)
            # with random synopses from other entries
            for _ in range(3):
                negative_idx = random.choice(df_sample.index)
                while negative_idx == idx:  # Ensure different entry
                    negative_idx = random.choice(df_sample.index)

                negative_synopsis = (
                    str(df_sample.loc[negative_idx, "combined_synopsis"])
                    if not pd.isna(df_sample.loc[negative_idx, "combined_synopsis"])
                    else ""
                )
                if not negative_synopsis:
                    continue

                examples.append(
                    InputExample(texts=[title, negative_synopsis], label=0.0)
                )

        logger.info(
            "Created %d training examples: %d positive, %d negative",
            len(examples),
            len(examples) // 4,
            3 * len(examples) // 4,
        )

        # Shuffle examples
        random.shuffle(examples)
        return examples

    def create_training_data_from_labeled_file(
        self, labeled_file: str
    ) -> List[InputExample]:
        """
        Create training data from a labeled CSV file.

        The CSV should have columns: query, synopsis, score

        Args:
            labeled_file: Path to labeled data CSV file

        Returns:
            List of InputExample objects for training
        """
        logger.info("Loading labeled training data from: %s", labeled_file)

        if not os.path.exists(labeled_file):
            raise FileNotFoundError(f"Labeled data file not found: {labeled_file}")

        labeled_df = pd.read_csv(labeled_file)
        required_cols = ["query", "synopsis", "score"]
        missing_cols = [col for col in required_cols if col not in labeled_df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in labeled data: {missing_cols}"
            )

        examples = []
        for _, row in tqdm(
            labeled_df.iterrows(), total=len(labeled_df), desc="Loading labeled data"
        ):
            query = str(row["query"]) if not pd.isna(row["query"]) else ""
            synopsis = str(row["synopsis"]) if not pd.isna(row["synopsis"]) else ""

            # Skip entries with empty query or synopsis
            if not query or not synopsis:
                continue

            examples.append(
                InputExample(texts=[query, synopsis], label=float(row["score"]))
            )

        logger.info("Loaded %d labeled examples", len(examples))
        return examples

    def create_query_variations(
        self, base_queries: List[str], n_variations: int = 7
    ) -> List[str]:
        """
        Create variations of base queries for robust training.

        Args:
            base_queries: List of base queries to create variations from
            n_variations: Number of variations to create per base query

        Returns:
            List of query variations
        """
        variations = []
        for query in base_queries:
            # Add the original query
            variations.append(query)

            # Create variations like:
            # "Looking for anime about X"
            # "What anime has X"
            # "Find me a manga with X"
            templates = [
                f"Looking for {self.dataset_type} about {query}",
                f"I want to watch {self.dataset_type} with {query}",
                f"Find me {self.dataset_type} where {query}",
                f"Can you recommend {self.dataset_type} that has {query}",
                f"What {self.dataset_type} is about {query}",
                f"{self.dataset_type} similar to {query}",
                f"{query} {self.dataset_type} recommendation",
                f"I'm looking for {self.dataset_type} with {query}",
                f"I'm searching for {self.dataset_type} with {query}",
                f"I'm trying to find {self.dataset_type} with {query}",
            ]

            # Select n_variations randomly
            variations.extend(
                random.sample(templates, min(n_variations, len(templates)))
            )

        return variations

    def train(
        self,
        labeled_file: Optional[str] = None,
        loss_type: str = "mse",
        scheduler: str = "linear",
    ) -> str:
        """
        Train the cross-encoder model with the prepared dataset.

        Args:
            labeled_file: Optional path to labeled data CSV file
            loss_type: Type of loss function to use ('mse' or 'cosine')
            scheduler: Learning rate scheduler type ('linear', 'cosine', etc.)

        Returns:
            Path to the saved fine-tuned model
        """
        logger.info("Starting training with %s", self.model_name)

        # Prepare training data
        if labeled_file is not None:
            train_examples = self.create_training_data_from_labeled_file(labeled_file)
        else:
            train_examples = self.create_synthetic_training_data()

        # Split into train and evaluation sets
        train_size = int(len(train_examples) * (1 - self.eval_split))
        train_data = train_examples[:train_size]
        eval_data = train_examples[train_size:]

        logger.info(
            "Training on %d examples, evaluating on %d examples",
            len(train_data),
            len(eval_data),
        )

        # 1. Define the model
        logger.info("Initializing model: %s", self.model_name)
        model = CrossEncoder(
            self.model_name,
            num_labels=1,
            device=self.device,
            max_length=512,
        )

        # Get tokenizer to determine token counts
        tokenizer = model.tokenizer
        max_length = 512

        # Faster batch truncation of text pairs
        def batch_truncate_pairs(text_pairs, batch_size=128):
            """
            Efficiently truncate multiple text pairs using batch processing.

            Args:
                text_pairs: List of (text_a, text_b) tuples
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
                sub_tokenized = tokenizer(
                    sub_batch,
                    add_special_tokens=True,
                    padding=False,
                    truncation=False,
                    return_tensors=None,
                )
                # Get token counts using most compatible approach
                sub_lengths = []

                # Use manual counting approach regardless of output format
                # This avoids any issues with iteration over tokenizer outputs
                for i, text in enumerate(sub_batch):
                    # Direct encode method is most reliable
                    token_count = len(tokenizer.encode(text, add_special_tokens=True))
                    sub_lengths.append(token_count)

                # Log if we're using fallback method
                if not (
                    isinstance(sub_tokenized, dict)
                    and "input_ids" in sub_tokenized
                    and isinstance(sub_tokenized["input_ids"], list)
                ):
                    logger.warning(
                        "Fallback method used for tokenization - results may be less accurate"
                    )

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

                # Process text_b truncation sequentially (no threading)
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
                    (batch_pairs[i][0], batch_truncated_b[i])
                    for i in range(len(batch_pairs))
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

        # 2. Format data for training with optimized truncation
        logger.info("Preparing training and evaluation data with batch truncation")

        # Extract text pairs from examples
        train_pairs = [(example.texts[0], example.texts[1]) for example in train_data]
        train_labels_list = [example.label for example in train_data]

        # Process training data in batches
        truncated_train_pairs = batch_truncate_pairs(train_pairs)

        # Separate truncated pairs back into lists
        train_texts1 = [pair[0] for pair in truncated_train_pairs]
        train_texts2 = [pair[1] for pair in truncated_train_pairs]

        # Create Hugging Face datasets
        train_dataset = HFDataset.from_dict(
            {
                "sentence_A": train_texts1,
                "sentence_B": train_texts2,
                "labels": train_labels_list,
            }
        )

        # Process evaluation data in batches
        eval_pairs = [(example.texts[0], example.texts[1]) for example in eval_data]
        eval_labels_list = [example.label for example in eval_data]

        truncated_eval_pairs = batch_truncate_pairs(eval_pairs)

        # Separate truncated pairs back into lists
        eval_texts1 = [pair[0] for pair in truncated_eval_pairs]
        eval_texts2 = [pair[1] for pair in truncated_eval_pairs]

        eval_dataset = HFDataset.from_dict(
            {
                "sentence_A": eval_texts1,
                "sentence_B": eval_texts2,
                "labels": eval_labels_list,
            }
        )

        # 3. Define a loss function
        if loss_type == "binary_cross_entropy":
            loss = BinaryCrossEntropyLoss(model)
        elif loss_type == "cross_entropy":
            loss = CrossEntropyLoss(model)
        elif loss_type == "lambda":
            loss = LambdaLoss(model)
        elif loss_type == "list_mle":
            loss = ListMLELoss(model)
        elif loss_type == "p_list_mle":
            loss = PListMLELoss(model)
        elif loss_type == "list_net":
            loss = ListNetLoss(model)
        elif loss_type == "multiple_negatives_ranking":
            loss = MultipleNegativesRankingLoss(model)
        elif loss_type == "cached_multiple_negatives_ranking":
            loss = CachedMultipleNegativesRankingLoss(model)
        elif loss_type == "mse":
            loss = MSELoss(model)
        elif loss_type == "margin_mse":
            loss = MarginMSELoss(model)
        elif loss_type == "rank_net":
            loss = RankNetLoss(model)
        else:
            logger.info("Unknown loss type '%s', falling back to MSE loss", loss_type)
            loss = MSELoss(model)

        # Configure training parameters
        # Dynamically calculate warmup_steps based on batch size if not explicitly specified
        if not self.warmup_steps_specified:
            # Scale warmup steps based on batch size - keeping 500 as baseline for batch_size=16
            reference_batch_size = 16
            reference_warmup_steps = 500
            warmup_steps = max(
                100,
                int(reference_warmup_steps * (reference_batch_size / self.batch_size)),
            )
            logger.info("Calculated dynamic warmup_steps: %d", warmup_steps)
        else:
            warmup_steps = self.warmup_steps
            logger.info("Using user-specified warmup_steps: %d", warmup_steps)

        # Dynamically calculate eval_steps if not specified by user
        if not self.eval_steps_specified:
            # Scale eval steps based on batch size - keeping 500 as baseline for batch_size=16
            reference_batch_size = 16
            reference_eval_steps = 500
            eval_steps = max(
                100,
                int(reference_eval_steps * (reference_batch_size / self.batch_size)),
            )
            logger.info("Calculated dynamic eval_steps: %d", eval_steps)
        else:
            eval_steps = self.eval_steps
            logger.info("Using specified eval_steps: %d", eval_steps)

        # Create training arguments
        training_args = CrossEncoderTrainingArguments(
            output_dir=self.output_path,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            eval_strategy="steps",
            eval_steps=eval_steps,
            warmup_steps=warmup_steps,
            learning_rate=self.learning_rate,
            weight_decay=0.05,  # L2 regularization
            lr_scheduler_type=scheduler,
            save_strategy="steps",
            save_steps=eval_steps,
            logging_steps=100,
            load_best_model_at_end=True,
            auto_find_batch_size=True,
        )

        logger.info(
            "Training for %d epochs with batch size %d", self.epochs, self.batch_size
        )
        logger.info(
            "Learning rate: %f, warmup steps: %d", self.learning_rate, warmup_steps
        )
        logger.info("Using scheduler: %s", scheduler)

        # 4. Create a trainer & train
        trainer = CrossEncoderTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
            args=training_args,
        )

        # Train the model
        trainer.train()

        # 5. Save the trained model
        os.makedirs(self.output_path, exist_ok=True)
        model.save_pretrained(self.output_path)
        logger.info("Training completed. Model saved to: %s", self.output_path)

        return self.output_path

    def _parse_list_column(self, column_value):
        """
        Parse a list column from the dataset that may be stored as a string.

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
                    return [
                        item.strip() for item in column_value.strip("[]").split(",")
                    ]
            else:
                # If it's just a single string value
                return [column_value]
        elif isinstance(column_value, list):
            return column_value
        else:
            return []

    def _calculate_similarity_score(self, row1, row2):
        """
        Calculate similarity score between two entries based on genres and themes.

        Args:
            row1: First DataFrame row
            row2: Second DataFrame row

        Returns:
            Float score between 0.0 and 0.8
        """
        # Parse genres and themes from both rows
        genres1 = set(self._parse_list_column(row1.get("genres", [])))
        genres2 = set(self._parse_list_column(row2.get("genres", [])))

        themes1 = set(self._parse_list_column(row1.get("themes", [])))
        themes2 = set(self._parse_list_column(row2.get("themes", [])))

        # If no genres or themes are available, return 0
        if not (genres1 or genres2 or themes1 or themes2):
            return 0.0

        # Calculate Jaccard similarity for genres and themes
        genre_similarity = 0.0
        if genres1 and genres2:
            genre_similarity = len(genres1.intersection(genres2)) / len(
                genres1.union(genres2)
            )

        theme_similarity = 0.0
        if themes1 and themes2:
            theme_similarity = len(themes1.intersection(themes2)) / len(
                themes1.union(themes2)
            )

        # Combine similarities with more weight on themes (more specific than genres)
        if genres1 or genres2:
            if themes1 or themes2:
                # If both are available, weighted average
                similarity = (0.4 * genre_similarity) + (0.6 * theme_similarity)
            else:
                # Only genres
                similarity = (
                    genre_similarity * 0.7
                )  # Reduce max score if only genres match
        else:
            # Only themes
            similarity = theme_similarity * 0.8  # Themes alone are a good signal

        # Scale to 0.0-0.8 range (perfect matches are still 1.0, these are partial matches)
        return min(0.8, similarity)

    def create_and_save_labeled_data(
        self,
        output_file: str,
        n_samples: int = 10000,
        include_partial_matches: bool = True,
    ) -> None:
        """
        Create and save synthetic labeled data to a CSV file.

        This can be useful for inspecting the training data or further modification.

        Args:
            output_file: Path to save the labeled data CSV
            n_samples: Number of labeled samples to create
            include_partial_matches: Whether to include partial matches based on genres/themes
        """
        logger.info("Creating %d labeled examples for inspection", n_samples)

        # Ensure output_file has a proper path
        if output_file.startswith("/"):
            output_file = output_file.lstrip("/")

        if not output_file.endswith(".csv"):
            output_file = f"{output_file}.csv"

        output_dir = os.path.dirname(output_file)
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
                logger.info("Created output directory: %s", output_dir)
            except PermissionError:
                logger.error(
                    "Permission denied when creating directory: %s", output_dir
                )
                logger.info("Trying to save to current directory instead")
                output_file = os.path.basename(output_file)

        # Limit dataset size
        df_sample = self.df.sample(min(len(self.df), n_samples), random_state=self.seed)

        # Create data
        data = []
        for idx, row in tqdm(
            df_sample.iterrows(), total=len(df_sample), desc="Creating labeled data"
        ):
            title = str(row["title"]) if not pd.isna(row["title"]) else ""
            synopsis = (
                str(row["combined_synopsis"])
                if not pd.isna(row["combined_synopsis"])
                else ""
            )

            # Skip entries with empty titles or synopses
            if not title or not synopsis:
                continue

            # 1. Add positive example (perfect match)
            data.append(
                {
                    "query": title,
                    "synopsis": synopsis,
                    "score": 1.0,
                    "example_type": "positive",
                }
            )

            # 2. Add similarity-based matches with varying scores
            if include_partial_matches and (
                "genres" in self.df.columns or "themes" in self.df.columns
            ):
                # Sample a larger pool of entries to evaluate for similarity
                sample_indices = random.sample(
                    list(set(df_sample.index) - {idx}),
                    min(
                        50, len(df_sample) - 1
                    ),  # Increased from 20 to 50 for better coverage
                )

                # Calculate similarity for all sampled entries
                similarity_scores = []
                for sample_idx in sample_indices:
                    sample_row = df_sample.loc[sample_idx]
                    score = self._calculate_similarity_score(row, sample_row)
                    similarity_scores.append((sample_idx, score))

                # Sort by similarity score
                similarity_scores.sort(key=lambda x: x[1])

                # Select entries across the similarity spectrum
                # We'll pick examples from different similarity bands to ensure good coverage
                selected_scores = []

                # Very low similarity (0.0-0.2) - replaces completely random negatives
                very_low = [s for s in similarity_scores if s[1] < 0.2]
                if very_low:
                    selected_scores.extend(
                        random.sample(very_low, min(2, len(very_low)))
                    )

                # Low similarity (0.2-0.4)
                low = [s for s in similarity_scores if 0.2 <= s[1] < 0.4]
                if low:
                    selected_scores.extend(random.sample(low, min(2, len(low))))

                # Medium similarity (0.4-0.6)
                medium = [s for s in similarity_scores if 0.4 <= s[1] < 0.6]
                if medium:
                    selected_scores.extend(random.sample(medium, min(2, len(medium))))

                # High similarity (0.6-0.8)
                high = [s for s in similarity_scores if 0.6 <= s[1] <= 0.8]
                if high:
                    selected_scores.extend(random.sample(high, min(2, len(high))))

                # Add all selected examples
                for sample_idx, score in selected_scores:
                    sample_row = df_sample.loc[sample_idx]
                    sample_synopsis = (
                        str(sample_row["combined_synopsis"])
                        if not pd.isna(sample_row["combined_synopsis"])
                        else ""
                    )

                    if not sample_synopsis:
                        continue

                    # Round score to nearest 0.1 for cleaner values
                    rounded_score = round(score * 10) / 10

                    # Set example type based on score range
                    if rounded_score < 0.2:
                        example_type = "very_low_similarity"
                    elif rounded_score < 0.4:
                        example_type = "low_similarity"
                    elif rounded_score < 0.6:
                        example_type = "medium_similarity"
                    else:
                        example_type = "high_similarity"

                    data.append(
                        {
                            "query": title,
                            "synopsis": sample_synopsis,
                            "score": rounded_score,
                            "example_type": example_type,
                        }
                    )

            # 3. Add query variations for some entries
            if random.random() < 0.5:  # 50% chance
                query_variations = self.create_query_variations([title])
                for variation in query_variations[1:]:  # Skip the original
                    data.append(
                        {
                            "query": variation,
                            "synopsis": synopsis,
                            "score": 1.0,
                            "example_type": "variation_positive",
                        }
                    )

        # Create DataFrame
        labeled_df = pd.DataFrame(data)

        # Print distribution of scores
        score_counts = labeled_df["score"].value_counts().sort_index()
        logger.info("Score distribution:")
        for score, count in score_counts.items():
            logger.info(
                "  Score %.1f: %d examples (%.1f%%)",
                score,
                count,
                100 * count / len(labeled_df),
            )

        # Try to save the file
        try:
            labeled_df.to_csv(output_file, index=False)
            logger.info("Saved %d labeled examples to %s", len(labeled_df), output_file)
        except PermissionError:
            fallback_file = f"labeled_data_{self.dataset_type}.csv"
            logger.error("Permission denied when writing to %s", output_file)
            logger.info("Saving to %s instead", fallback_file)
            try:
                labeled_df.to_csv(fallback_file, index=False)
                logger.info(
                    "Saved %d labeled examples to %s", len(labeled_df), fallback_file
                )
            except Exception as e:
                logger.error("Failed to save labeled data: %s", str(e))
                raise


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train and fine-tune cross-encoder models for anime/manga search"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["anime", "manga"],
        required=True,
        help="Type of dataset to use: 'anime' or 'manga'",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"Base cross-encoder model to fine-tune (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Training batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=DEFAULT_EVAL_STEPS,
        help=f"Steps between evaluations (default: {DEFAULT_EVAL_STEPS})",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"Maximum number of training samples (default: {DEFAULT_MAX_SAMPLES})",
    )
    parser.add_argument(
        "--labeled-data", type=str, help="Path to labeled data CSV file (if available)"
    )
    parser.add_argument(
        "--create-labeled-data",
        type=str,
        help="Create and save synthetic labeled data to the specified CSV path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available base models for fine-tuning",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=[
            "binary_cross_entropy",
            "cross_entropy",
            "lambda",
            "list_mle",
            "p_list_mle",
            "list_net",
            "multiple_negatives_ranking",
            "cached_multiple_negatives_ranking",
            "mse",
            "margin_mse",
            "rank_net",
        ],
        default="mse",
        help="Loss function to use for training",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
            "inverse_sqrt",
            "reduce_lr_on_plateau",
            "cosine_with_min_lr",
            "warmup_stable_decay",
        ],
        default="linear",
        help="Learning rate scheduler",
    )

    return parser.parse_args()


def list_available_models() -> None:
    """Display available models for fine-tuning."""
    print("\nAvailable Base Models for Fine-Tuning:")
    print("====================================")

    # Flatten nested dictionary for display
    flat_models = {}
    for category, models in ALTERNATIVE_MODELS.items():
        print(f"\n{category.upper()}:")
        for name, path in models.items():
            print(f"  {name}: {path}")
            flat_models[name] = path

    print("\nUsage example:")
    print(
        "  python src/train_model.py --type anime ",
        '--model "cross-encoder/ms-marco-MiniLM-L-6-v2" --epochs 3',
    )
    print("\nModel selection guide:")
    print("- TinyBERT models: Smallest and fastest, good for low-resource environments")
    print("- MiniLM models: Good balance of performance and efficiency")
    print("- ELECTRA models: Higher accuracy but more computationally intensive")
    print("- MS Marco models: Optimized for information retrieval")
    print("- Other models: Specialized for specific NLP tasks (NLI, QA, etc.)")


def main() -> None:
    """
    Main function to run the model training.
    """
    args = parse_args()

    # Display available models if requested
    if args.list_models:
        list_available_models()
        return

    try:
        # Initialize trainer
        trainer = AnimeModelTrainer(
            dataset_type=args.type,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_steps=args.eval_steps,
            max_samples=args.max_samples,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )

        # Create and save labeled data if requested
        if args.create_labeled_data:
            trainer.create_and_save_labeled_data(args.create_labeled_data)
            logger.info(
                "Labeled data created and saved to: %s", args.create_labeled_data
            )
            return

        # Train model
        output_path = trainer.train(
            labeled_file=args.labeled_data,
            loss_type=args.loss,
            scheduler=args.scheduler,
        )

        print("\n" + "=" * 50)
        print("Training completed successfully!")
        print(f"Fine-tuned model saved to: {output_path}")
        print("To use this model for search:")
        print(
            f"  python src/search_model.py --type {args.type} ",
            '--model "{output_path}" --query "Your query"',
        )
        print("=" * 50)

    except Exception as e:
        logger.error("Error during training: %s", str(e), exc_info=True)
        raise


if __name__ == "__main__":
    main()
