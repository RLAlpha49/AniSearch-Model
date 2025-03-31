"""
# Base Model Trainer

Comprehensive framework for fine-tuning cross-encoder models for anime and manga search.

This module provides a base trainer class that implements the core functionality needed
for fine-tuning cross-encoder models on anime and manga datasets. It handles the entire
training pipeline, from dataset preparation and example generation to model training
and evaluation.

## Features

- Configurable training parameters (epochs, batch size, learning rate, etc.)
- Synthetic training data generation with positive and negative examples
- Support for custom labeled datasets
- Smart truncation of text pairs to fit model token limits
- Automatic device selection (CPU/GPU)
- Multiple loss function options
- Query variation generation for improved robustness
- Proper train/evaluation splitting
- Reproducible results through consistent random seed handling

## Usage Context

The base trainer is used for:

1. Fine-tuning pre-trained cross-encoder models on anime/manga data
2. Creating specialized models that better understand anime/manga terminology
3. Generating labeled training data for inspection or custom training
4. Developing models that provide more relevant search results for specific content types

The trainer uses SentenceTransformers' CrossEncoder implementation and integrates
with HuggingFace's training utilities for efficient fine-tuning.
"""

# pylint: disable=too-many-lines

import os
import logging
import random
from typing import List, Optional, Any

import pandas as pd
from datasets import Dataset as HFDataset
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.losses import (  # pylint: disable=no-name-in-module,import-error
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
from sentence_transformers.cross_encoder.trainer import (  # pylint: disable=no-name-in-module,import-error
    CrossEncoderTrainer,
)
from sentence_transformers.cross_encoder.training_args import (  # pylint: disable=no-name-in-module,import-error
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


class BaseModelTrainer:  # pylint: disable=too-many-instance-attributes
    """
    Base trainer class for fine-tuning cross-encoder models on anime/manga datasets.

    This class provides the core functionality for training cross-encoder models
    on anime and manga datasets. It handles dataset preparation, synthetic training
    data generation, model configuration, and training execution. The trainer supports
    various training parameters and loss functions, allowing for flexible model tuning.

    The trainer creates training examples by pairing titles (queries) with synopses
    (documents), generating both positive pairs (matching title-synopsis) and negative
    pairs (title with unrelated synopsis). It can also generate variations of queries
    to improve model robustness.

    Attributes:
        dataset_type (str): Type of dataset ('anime' or 'manga')
        model_name (str): Name or path of the base model to fine-tune
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
        eval_steps (int): Number of steps between evaluations
        warmup_steps (int): Number of warmup steps for learning rate scheduler
        max_samples (int): Maximum number of training samples to use
        learning_rate (float): Learning rate for the optimizer
        eval_split (float): Fraction of data to use for evaluation
        seed (int): Random seed for reproducibility
        device (str): Device to use for training ('cpu', 'cuda', etc.)
        dataset_path (str): Path to the dataset file
        df (pd.DataFrame): The loaded dataset
        output_path (str): Path where the fine-tuned model will be saved
        synopsis_cols (List[str]): Columns containing synopsis information
        id_col (str): Column containing the ID for anime/manga entries

    Example:
        ```python
        # Initialize a trainer for anime dataset
        trainer = BaseModelTrainer(
            dataset_type="anime",
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            epochs=3,
            batch_size=16
        )

        # Train the model
        output_path = trainer.train(loss_type="mse")

        # Create labeled data for inspection
        trainer.create_and_save_labeled_data(
            output_file="labeled_anime_data.csv",
            n_samples=5000
        )
        ```

    Notes:
        - The trainer requires merged datasets to be available. If not found, it will
          suggest running the merge_datasets.py script first.
        - For best results, ensure the dataset contains adequate synopsis information
          and relevant metadata like genres and themes.
        - The trainer automatically handles text truncation to fit within model token
          limits, prioritizing the query (title) over the document (synopsis).
    """

    def __init__(  # pylint: disable=too-many-arguments
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
        Initialize the trainer with configuration parameters for model fine-tuning.

        This constructor sets up the training environment, loads the appropriate dataset,
        and prepares internal state for the training process. It validates inputs,
        sets up random seeds for reproducibility, configures the device, and establishes
        the model output path.

        Args:
            dataset_type: The type of dataset to use for training. Must be either 'anime'
                or 'manga'. This determines which dataset is loaded and how certain
                processing steps are performed. Default is 'anime'.

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

            max_samples: Maximum number of training samples to use from the dataset. Useful
                for limiting training time or for testing. Set to None to use all available
                data. Default is DEFAULT_MAX_SAMPLES (10000).

            learning_rate: Learning rate for the optimizer. Controls how quickly model
                weights are updated during training. Default is DEFAULT_LEARNING_RATE (2e-6).

            eval_split: Fraction of data to use for evaluation instead of training. Must
                be between 0 and 1. Default is 0.1 (10% for evaluation).

            seed: Random seed for reproducibility. Ensures the same training/evaluation
                split and data sampling across runs. Default is 42.

            device: Device to use for training ('cpu', 'cuda', 'cuda:0', etc.). If None,
                automatically selects GPU if available, otherwise CPU. Default is None.

            dataset_path: Path to the dataset file. If None, uses the default path based
                on dataset_type. Default is None.

        Raises:
            ValueError: If dataset_type is not 'anime' or 'manga'
            FileNotFoundError: If the dataset file doesn't exist

        Notes:
            - The method automatically creates the output directory if it doesn't exist
            - The output path is constructed from the model name and dataset type
            - After initialization, the dataset is prepared by calling _prepare_dataset()
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
        """
        Prepare the dataset for training by extracting and combining synopsis data.

        This method performs the following preparation steps:

        1. Identifies all columns containing synopsis information
        2. Combines multiple synopsis columns into a single 'combined_synopsis' column
        3. Filters out entries with empty synopses
        4. Validates that all required columns exist in the dataset

        The method modifies the DataFrame in-place, adding a 'combined_synopsis' column
        and potentially reducing the number of rows if empty synopses are found.

        Returns:
            None: The method modifies the self.df attribute in-place rather than
                returning a value.

        Raises:
            ValueError: If required columns are missing from the dataset

        Notes:
            - The method uses a case-insensitive search to find all synopsis columns
            - Synopsis columns are combined with spaces between each synopsis
            - Empty or whitespace-only synopses are removed to avoid training on
              unhelpful examples
            - The method is decorated with handle_exceptions for error handling
        """
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
        Create synthetic training data pairs for cross-encoder model fine-tuning.

        This method generates a balanced dataset of positive and negative examples:

        - **Positive examples**: Pairs of titles with their matching synopses (label 1.0)
        - **Negative examples**: Pairs of titles with randomly selected unrelated
          synopses (label 0.0)

        For each positive example, the method creates 3 negative examples, resulting
        in a 1:3 ratio of positive to negative examples. This ratio helps the model
        learn to distinguish relevant from irrelevant content.

        The method samples up to max_samples entries from the dataset and applies
        randomization with the configured seed for reproducibility. Examples with
        empty titles or synopses are skipped.

        Returns:
            List[InputExample]: A list of InputExample objects ready for training,
                where each example contains:
                - texts[0]: A title (query)
                - texts[1]: A synopsis (document)
                - label: 1.0 for positive pairs, 0.0 for negative pairs

        Example:
            ```python
            # Create synthetic training data
            trainer = BaseModelTrainer(dataset_type="anime")
            examples = trainer.create_synthetic_training_data()

            # Examine the first few examples
            for i, example in enumerate(examples[:5]):
                print(f"Example {i}:")
                print(f"  Query: {example.texts[0][:50]}...")
                print(f"  Document: {example.texts[1][:50]}...")
                print(f"  Label: {example.label}")
            ```

        Notes:
            - The method is decorated with handle_exceptions for error handling
            - Results are shuffled before returning to randomize the training order
            - If max_samples is smaller than the dataset size, a random subset is used
            - The 1:3 positive-to-negative ratio is a common practice in information
              retrieval tasks to handle the natural imbalance of relevant vs. irrelevant
              documents
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
        Create training data from a pre-labeled CSV file instead of synthetic generation.

        This method allows for using custom or human-labeled data for training. The
        labeled file should be a CSV containing at least three columns:

        - **query**: The search query or title
        - **text**: The document or synopsis text
        - **score**: A numerical score/label (typically 0-1) indicating relevance

        Using labeled data gives more control over the training examples and can
        incorporate domain expertise about what constitutes good matches. It's especially
        useful for fine-grained relevance levels beyond just binary classification.

        Args:
            labeled_file: Path to the CSV file containing labeled examples. The file
                must include 'query', 'text', and 'score' columns.

        Returns:
            List[InputExample]: A list of InputExample objects created from the labeled
                file, where each example contains:
                - texts[0]: The query from the 'query' column
                - texts[1]: The document from the 'text' column
                - label: The float score from the 'score' column

        Raises:
            FileNotFoundError: If the labeled_file doesn't exist
            ValueError: If the required columns are missing from the file

        Example:
            ```python
            # Create training data from labeled file
            trainer = BaseModelTrainer(dataset_type="anime")
            examples = trainer.create_training_data_from_labeled_file(
                "path/to/labeled_data.csv"
            )

            # Print distribution of scores
            score_counts = {}
            for example in examples:
                score = example.label
                score_counts[score] = score_counts.get(score, 0) + 1

            for score, count in sorted(score_counts.items()):
                print(f"Score {score}: {count} examples")
            ```

        Notes:
            - The method is decorated with handle_exceptions for error handling
            - No shuffling is performed as the labeled file may already have a
              specific order
            - Empty values in the CSV are converted to empty strings
            - The scores are converted to float values
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
        Create natural language variations of base queries to enhance training robustness.

        This method generates conversational and alternative phrasings of the base queries
        to help the model recognize the same intent expressed in different ways. For each
        base query, it creates variations using templates like "I'm looking for {query}"
        or "Find me {query}".

        Query variations are important for training more robust models that can handle
        real-world search inputs, which often contain conversational phrases and different
        formulations of the same information need.

        Args:
            base_queries: List of original query strings (typically anime/manga titles)
                that will be used as the basis for generating variations.

            n_variations: Number of variations to create for each base query. The actual
                number may be less if there aren't enough templates. Default is 7.

        Returns:
            List[str]: A combined list containing both the original queries and their
                variations. The length will be approximately len(base_queries) * (1 + n_variations),
                but may be less if n_variations exceeds the number of available templates.

        Example:
            ```python
            # Create variations of anime titles
            titles = ["Naruto", "One Piece", "Attack on Titan"]
            trainer = BaseModelTrainer(dataset_type="anime")
            variations = trainer.create_query_variations(titles, n_variations=3)

            # Print all variations
            for var in variations:
                print(var)
            # Example output:
            # Naruto
            # I'm looking for Naruto
            # Can you recommend Naruto?
            # Find me Naruto
            # One Piece
            # ...etc.
            ```

        Notes:
            - The method always includes the original queries in the returned list
            - Templates are selected randomly for each query
            - The method is designed for English language variations
            - The method is decorated with handle_exceptions for error handling
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

        This method executes the full training pipeline:

        1. Prepares training data (synthetic or from labeled file)
        2. Splits data into training and evaluation sets
        3. Truncates text pairs to fit model token limits
        4. Configures the model, loss function, and training arguments
        5. Executes the training process
        6. Saves the fine-tuned model

        The method supports various loss functions and learning rate schedulers to
        optimize different aspects of model performance. It automatically handles
        device placement, batching, and evaluation during training.

        Args:
            labeled_file: Optional path to a pre-labeled CSV file containing training
                examples. If provided, uses this file instead of generating synthetic
                data. Default is None (generate synthetic data).

            loss_type: Type of loss function to use for training. Supported options:
                - 'mse' (default): Mean Squared Error loss
                - 'binary_cross_entropy': Binary Cross Entropy loss
                - 'cross_entropy': Cross Entropy loss
                - 'lambda': LambdaLoss for LambdaRank-style learning to rank
                - 'list_mle', 'p_list_mle': ListMLE/PListMLE losses for listwise learning
                - 'list_net': ListNet loss for listwise learning
                - 'multiple_negatives_ranking': Multiple Negatives Ranking loss
                - 'cached_multiple_negatives_ranking': Cached version of MNR loss
                - 'margin_mse': Margin MSE loss
                - 'rank_net': RankNet loss for pairwise learning

            scheduler: Learning rate scheduler type. Options include:
                - 'linear' (default): Linear decay from initial value to 0
                - 'cosine': Cosine decay schedule
                - 'cosine_with_restarts': Cosine decay with periodic restarts
                - 'polynomial': Polynomial decay
                - 'constant': Constant learning rate
                - 'constant_with_warmup': Constant learning rate after warmup

        Returns:
            str: Path to the saved fine-tuned model, which can be loaded later for
                inference or additional training.

        Example:
            ```python
            # Train a model with default settings
            trainer = BaseModelTrainer(
                dataset_type="anime",
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                epochs=3
            )
            model_path = trainer.train()
            print(f"Model saved to: {model_path}")

            # Train with custom loss and scheduler
            trainer2 = BaseModelTrainer(dataset_type="manga")
            model_path = trainer2.train(
                loss_type="binary_cross_entropy",
                scheduler="cosine"
            )
            ```

        Notes:
            - The method automatically calculates reasonable evaluation and warmup steps
              if they weren't explicitly specified during initialization
            - Training progress is logged using tqdm progress bars and the logger
            - The model with the best evaluation performance is automatically saved
            - The method is decorated with handle_exceptions for error handling
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
            loss: Any = BinaryCrossEntropyLoss(model)
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
        Calculate a similarity score between two dataset entries based on genres and themes.

        This internal method computes a content-based similarity score by comparing the
        genres and themes between two anime/manga entries. It uses the Jaccard similarity
        coefficient (intersection over union) for both genres and themes, then combines
        them with appropriate weighting.

        The method prioritizes theme matching over genre matching since themes tend to be
        more specific indicators of content similarity. The final score is capped at 0.8
        to reserve perfect scores (1.0) for exact matches.

        Args:
            row1: First DataFrame row containing genre and theme information
            row2: Second DataFrame row to compare against

        Returns:
            float: A similarity score between 0.0 and 0.8, where:
                - 0.0 indicates no similarity in genres or themes
                - 0.1-0.3 indicates minimal genre/theme overlap
                - 0.4-0.6 indicates moderate content similarity
                - 0.7-0.8 indicates high content similarity

        Notes:
            - The method handles missing genre/theme data gracefully
            - Themes are weighted more heavily (0.6) than genres (0.4) when both exist
            - If only genres match, the maximum possible score is 0.7
            - If only themes match, the maximum possible score is 0.8
            - The method uses parse_list_column to handle various list formats
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
        Create and save synthetic labeled data to a CSV file for inspection or custom training.

        This method generates a rich dataset of labeled examples with various levels of
        relevance between queries and documents. Unlike the synthetic training data used
        directly for training (which uses binary labels), this method creates examples with
        graded relevance scores between 0.0 and 1.0, capturing partial matches based on
        content similarity.

        The generated CSV file includes:

        - **Perfect matches**: Title paired with its own synopsis (score 1.0)
        - **Partial matches**: Title paired with synopses of similar content based on
          genres and themes (scores 0.1-0.8)
        - **Query variations**: Conversational variations of titles (e.g., "Looking for X")
          paired with matching synopses (score 1.0)

        Args:
            output_file: Path to save the labeled data CSV file. If the directory doesn't
                exist, it will be created. If writing fails due to permissions, the file
                will be saved to the current directory.

            n_samples: Number of base entries to sample from the dataset for creating
                labeled examples. The actual number of examples in the output will be
                larger due to variations and partial matches. Default is 10000.

            include_partial_matches: Whether to include examples with partial relevance
                based on genre/theme similarity. When True, the dataset will include
                examples with scores between 0.1 and 0.8. When False, only perfect
                matches (1.0) and variations will be included. Default is True.

        Returns:
            None: The method saves the labeled data to a file but doesn't return a value.

        Example:
            ```python
            # Create labeled data with default settings
            trainer = BaseModelTrainer(dataset_type="anime")
            trainer.create_and_save_labeled_data("data/labeled_anime.csv")

            # Create a smaller dataset without partial matches
            trainer.create_and_save_labeled_data(
                "data/simple_labeled_anime.csv",
                n_samples=5000,
                include_partial_matches=False
            )
            ```

        Notes:
            - The output CSV includes an 'example_type' column indicating the type of each
              example (positive, variation_positive, or similarity-based)
            - Similarity-based scores are rounded to the nearest 0.1 for cleaner values
            - Query variations are added to approximately 50% of the titles
            - The method handles permission errors by falling back to the current directory
            - The method logs a distribution of scores in the final dataset
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
