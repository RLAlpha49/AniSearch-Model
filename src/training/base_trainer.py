"""
Base trainer class for fine-tuning cross-encoder models for anime/manga search.
"""

import os
import logging
import random
from typing import List, Optional

import pandas as pd
from datasets import Dataset as HFDataset
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
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
from tqdm import tqdm
import transformers

from src.utils.constants import MODEL_NAME
from src.training.utils import (
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EVAL_STEPS,
    DEFAULT_WARMUP_STEPS,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_LEARNING_RATE,
    MODEL_SAVE_PATH,
    batch_truncate_text_pairs,
    get_device,
    parse_list_column,
    setup_random_seeds,
)
from src.models.anime_search_model import ANIME_DATASET_PATH
from src.models.manga_search_model import MANGA_DATASET_PATH
from src.utils.error_handling import handle_exceptions

# Configure logging
logger = logging.getLogger(__name__)


class BaseModelTrainer:
    """Base class for training cross-encoder models on anime/manga datasets."""

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
        dataset_path: Optional[str] = None,
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
            dataset_path: Path to the dataset file (if None, will use default based on dataset_type)
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
        setup_random_seeds(seed)

        # Set device
        self.device = get_device(device)
        logger.info("Using device: %s", self.device)

        # Set model save path
        if not os.path.exists(MODEL_SAVE_PATH):
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

        model_basename = os.path.basename(model_name.replace("/", "-"))
        self.output_path = os.path.join(
            MODEL_SAVE_PATH, f"{model_basename}-{dataset_type}-finetuned"
        )

        # Load dataset
        if dataset_path is None:
            # Use default dataset path if none is provided

            self.dataset_path = (
                ANIME_DATASET_PATH if dataset_type == "anime" else MANGA_DATASET_PATH
            )
        else:
            self.dataset_path = dataset_path

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

    @handle_exceptions(log_exceptions=True, include_exc_info=True)
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

    @handle_exceptions(log_exceptions=True, include_exc_info=True)
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

    @handle_exceptions(log_exceptions=True, include_exc_info=True)
    def create_training_data_from_labeled_file(
        self, labeled_file: str
    ) -> List[InputExample]:
        """
        Create training data from a labeled CSV file.

        Args:
            labeled_file: Path to the labeled CSV file

        Returns:
            List of InputExample objects for training
        """
        logger.info("Loading labeled data from %s", labeled_file)
        if not os.path.exists(labeled_file):
            raise FileNotFoundError(f"Labeled file not found: {labeled_file}")

        df = pd.read_csv(labeled_file)
        logger.info("Loaded %d labeled examples", len(df))

        # Ensure required columns exist
        required_cols = ["query", "text", "score"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in labeled data: {missing_cols}"
            )

        # Convert to InputExample format
        examples = []
        for _, row in df.iterrows():
            examples.append(
                InputExample(
                    texts=[
                        str(row["query"]) if not pd.isna(row["query"]) else "",
                        str(row["text"]) if not pd.isna(row["text"]) else "",
                    ],
                    label=float(row["score"]),
                )
            )

        logger.info("Created %d training examples from labeled data", len(examples))
        return examples

    @handle_exceptions(log_exceptions=True, include_exc_info=True)
    def create_query_variations(
        self, base_queries: List[str], n_variations: int = 7
    ) -> List[str]:
        """
        Create variations of base queries to increase training data diversity.

        Args:
            base_queries: List of original query strings
            n_variations: Number of variations to create per query

        Returns:
            List of query variations
        """
        templates = [
            "I'm looking for {query}",
            "Can you recommend {query}?",
            "Find me {query}",
            "I want to watch {query}",
            "Suggest {query}",
            "I need {query}",
            "Something like {query}",
            "Similar to {query}",
            "{query} or similar",
            "Looking for {query}",
            "Need recommendations for {query}",
            "What's similar to {query}?",
            "I enjoyed {query}, what else?",
        ]

        variations = []
        for query in base_queries:
            # Add the original query
            variations.append(query)

            # Add variations based on templates
            n_to_use = min(n_variations, len(templates))
            selected_templates = random.sample(templates, n_to_use)
            for template in selected_templates:
                variations.append(template.format(query=query))

        return variations

    @handle_exceptions(log_exceptions=True, include_exc_info=True)
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
            loss_type: Type of loss function to use ('mse' or other supported types)
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

        # Try to disable tokenizer warnings
        transformers.logging.set_verbosity_error()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Initialize the model with env vars to encourage fast tokenizer
        os.environ["USE_FAST_TOKENIZER"] = "true"

        # Initialize the model
        logger.info("Initializing model: %s", self.model_name)
        model = CrossEncoder(
            self.model_name,
            num_labels=1,
            device=self.device,
            max_length=512,
        )

        # Get the tokenizer
        tokenizer = model.tokenizer
        max_length = 512

        # Manually truncate text pairs to avoid tokenizer warnings
        logger.info("Truncating training examples to fit max_length")
        # Extract text pairs from examples
        train_pairs = [(example.texts[0], example.texts[1]) for example in train_data]
        train_labels = [example.label for example in train_data]

        # Process training data in batches
        truncated_train_pairs = batch_truncate_text_pairs(
            train_pairs, tokenizer, max_length=max_length
        )

        # Create truncated examples
        truncated_train_data = []
        for i, (text_a, text_b) in enumerate(truncated_train_pairs):
            truncated_train_data.append(
                InputExample(texts=[text_a, text_b], label=train_labels[i])
            )

        # Process evaluation data in batches
        eval_pairs = [(example.texts[0], example.texts[1]) for example in eval_data]
        eval_labels = [example.label for example in eval_data]

        truncated_eval_pairs = batch_truncate_text_pairs(
            eval_pairs, tokenizer, max_length=max_length
        )

        # Create truncated examples
        truncated_eval_data = []
        for i, (text_a, text_b) in enumerate(truncated_eval_pairs):
            truncated_eval_data.append(
                InputExample(texts=[text_a, text_b], label=eval_labels[i])
            )

        # Prepare datasets for the CrossEncoderTrainer
        train_texts1 = [example.texts[0] for example in truncated_train_data]
        train_texts2 = [example.texts[1] for example in truncated_train_data]
        train_labels = [example.label for example in truncated_train_data]

        eval_texts1 = [example.texts[0] for example in truncated_eval_data]
        eval_texts2 = [example.texts[1] for example in truncated_eval_data]
        eval_labels = [example.label for example in truncated_eval_data]

        # Create HuggingFace datasets
        train_hf_dataset = HFDataset.from_dict(
            {
                "sentence_A": train_texts1,
                "sentence_B": train_texts2,
                "labels": train_labels,
            }
        )

        eval_hf_dataset = HFDataset.from_dict(
            {
                "sentence_A": eval_texts1,
                "sentence_B": eval_texts2,
                "labels": eval_labels,
            }
        )

        # Set warm-up steps based on epochs and dataset size
        if not self.warmup_steps_specified:
            # Calculate steps per epoch (approx)
            steps_per_epoch = max(1, len(truncated_train_data) // self.batch_size)
            # Use 10% of total steps but ensure at least 100 steps
            total_steps = steps_per_epoch * self.epochs
            self.warmup_steps = max(100, int(total_steps * 0.1))
            logger.info(
                "Using %d warm-up steps (approx. %d%% of total steps)",
                self.warmup_steps,
                int(100 * self.warmup_steps / total_steps),
            )

        # Set evaluation steps if not specified
        if not self.eval_steps_specified:
            # Calculate reasonable evaluation frequency - evaluate ~5 times per epoch
            steps_per_epoch = max(1, len(truncated_train_data) // self.batch_size)
            self.eval_steps = max(100, steps_per_epoch // 5)
            logger.info(
                "Using %d evaluation steps (approx. 5 times per epoch)", self.eval_steps
            )

        # Set up loss function
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

        # Create training arguments
        training_args = CrossEncoderTrainingArguments(
            output_dir=self.output_path,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            eval_strategy="steps",
            eval_steps=self.eval_steps,
            warmup_steps=self.warmup_steps,
            learning_rate=self.learning_rate,
            weight_decay=0.05,  # L2 regularization
            lr_scheduler_type=scheduler,
            save_strategy="steps",
            save_steps=self.eval_steps,
            logging_steps=100,
            load_best_model_at_end=True,
            auto_find_batch_size=True,
            disable_tqdm=False,
        )

        # Initialize trainer
        trainer = CrossEncoderTrainer(
            model=model,
            args=training_args,
            train_dataset=train_hf_dataset,
            eval_dataset=eval_hf_dataset,
            loss=loss,
        )

        # Train the model
        logger.info(
            "Training with: epochs=%d, batch_size=%d, warmup_steps=%d, eval_steps=%d",
            self.epochs,
            self.batch_size,
            self.warmup_steps,
            self.eval_steps,
        )
        trainer.train()

        # Save the model
        logger.info("Saving fine-tuned model to %s", self.output_path)
        model.save(self.output_path)
        logger.info("Training completed successfully!")

        return self.output_path

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
        genres1 = set(parse_list_column(row1.get("genres", [])))
        genres2 = set(parse_list_column(row2.get("genres", [])))

        themes1 = set(parse_list_column(row1.get("themes", [])))
        themes2 = set(parse_list_column(row2.get("themes", [])))

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
                    "text": synopsis,
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
                            "text": sample_synopsis,
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
                            "text": synopsis,
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
