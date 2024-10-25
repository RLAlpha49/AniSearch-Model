"""
Training module for fine-tuning SentenceTransformer models on anime/manga synopsis data.

This module provides functionality for training sentence transformer models to understand
semantic similarities between anime/manga synopses. It handles the complete training
pipeline, including:

1. Data Processing:
   - Loading anime/manga datasets
   - Managing genres and themes
   - Generating embeddings for categories

2. Pair Generation:
   - Positive pairs: Same-entry synopses with high similarity
   - Partial positive pairs: Different-entry synopses with moderate similarity
   - Negative pairs: Different-entry synopses with low similarity

3. Model Training:
   - Fine-tuning pre-trained sentence transformers
   - Custom loss function support (cosine, cosent, angle)
   - Validation and evaluation during training
   - Checkpoint saving and model persistence

4. Resource Management:
   - GPU memory management with garbage collection
   - Multiprocessing for pair generation
   - Efficient data loading with DataLoader

Configuration:
    The module accepts various command line arguments to configure:
    - Model architecture and pre-trained weights
    - Training hyperparameters (learning rate, batch size, epochs)
    - Data processing settings (pair counts, thresholds)
    - Resource allocation (workers, GPU usage)
    - Input/output paths and data types

Usage:
    python train.py [arguments]

    For full list of arguments, use: python train.py --help

Notes:
    - Supports resuming training from saved pair files
    - Uses cosine similarity for evaluation
    - Handles both anime and manga datasets with specific genre/theme sets
    - Custom transformer option available for modified architectures
"""

import argparse
import os
import logging
from multiprocessing import cpu_count
import gc
from typing import Tuple, List, Dict, Optional
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from numpy.typing import NDArray

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf  # pylint: disable=wrong-import-position

# Set TensorFlow's logging level to ERROR
tf.get_logger().setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer, InputExample  # pylint: disable=wrong-import-position # noqa: E402

from training.data.pair_generation import (  # pylint: disable=wrong-import-position import-error no-name-in-module # noqa: E402
    create_positive_pairs,
    create_partial_positive_pairs,
    create_negative_pairs,
)
from training.common.data_utils import (  # pylint: disable=wrong-import-position import-error no-name-in-module # noqa: E402
    get_genres_and_themes,
)
from training.models.training import (  # pylint: disable=wrong-import-position import-error no-name-in-module # noqa: E402
    create_model,
    create_evaluator,
    get_loss_function,
)


# Function to create positive and negative pairs
def create_pairs(
    df: pd.DataFrame,
    max_negative_per_row: int,
    max_partial_positive_per_row: int,
    category_to_embedding: Dict[str, NDArray[np.float64]],
    partial_threshold: float = 0.5,
    positive_pairs_file: Optional[str] = None,
    partial_positive_pairs_file: Optional[str] = None,
    negative_pairs_file: Optional[str] = None,
    use_saved_pairs: bool = False,
    num_workers: int = cpu_count() // 4,
) -> Tuple[List[InputExample], List[InputExample], List[InputExample]]:
    """
    Create positive, partial positive, and negative pairs from the dataframe.

    This function handles the generation of three types of synopsis pairs:
    1. Positive pairs: From same entries with high similarity
    2. Partial positive pairs: From different entries with moderate similarity
    3. Negative pairs: From different entries with low similarity

    Args:
        df: DataFrame containing anime/manga data with synopses
        max_negative_per_row: Maximum negative pairs to generate per entry
        max_partial_positive_per_row: Maximum partial positive pairs per entry
        category_to_embedding: Dictionary mapping categories to their vector embeddings
        partial_threshold: Similarity threshold for partial positive pairs (default: 0.5)
        positive_pairs_file: Optional path to save/load positive pairs
        partial_positive_pairs_file: Optional path to save/load partial positive pairs
        negative_pairs_file: Optional path to save/load negative pairs
        use_saved_pairs: Whether to load existing pairs if available
        num_workers: Number of parallel workers for pair generation

    Returns:
        Tuple containing:
        - List[InputExample]: Positive pairs
        - List[InputExample]: Partial positive pairs
        - List[InputExample]: Negative pairs

    Notes:
        - Uses sentence-t5-xl model for encoding during pair generation
        - Performs garbage collection after each pair type generation
        - Saves generated pairs to files if paths are provided
    """
    synopses_columns: List[str] = [
        col for col in df.columns if "synopsis" in col.lower()
    ]

    # Load a pre-trained Sentence Transformer model for encoding
    encoder_model: SentenceTransformer = SentenceTransformer("sentence-t5-xl")

    positive_pairs: List[InputExample] = []
    if (
        positive_pairs_file is None
        or not os.path.exists(positive_pairs_file)
        or not use_saved_pairs
    ):
        positive_pairs = create_positive_pairs(
            df, synopses_columns, encoder_model, positive_pairs_file
        )
        gc.collect()
        torch.cuda.empty_cache()

    partial_positive_pairs: List[InputExample] = []
    if (
        partial_positive_pairs_file is None
        or not os.path.exists(partial_positive_pairs_file)
        or not use_saved_pairs
    ):
        partial_positive_pairs = create_partial_positive_pairs(
            df,
            synopses_columns,
            partial_threshold,
            max_partial_positive_per_row,
            partial_positive_pairs_file,
            num_workers,
            category_to_embedding,
        )
        gc.collect()
        torch.cuda.empty_cache()

    negative_pairs: List[InputExample] = []
    if (
        negative_pairs_file is None
        or not os.path.exists(negative_pairs_file)
        or not use_saved_pairs
    ):
        negative_pairs = create_negative_pairs(
            df,
            synopses_columns,
            partial_threshold,
            max_negative_per_row,
            negative_pairs_file,
            num_workers,
            category_to_embedding,
        )
        gc.collect()
        torch.cuda.empty_cache()

    return positive_pairs, partial_positive_pairs, negative_pairs


# Function to get the pairs
def get_pairs(
    df: pd.DataFrame,
    use_saved_pairs: bool,
    saved_pairs_directory: str,
    max_negative_per_row: int,
    max_partial_positive_per_row: int,
    num_workers: int,
    data_type: str,
    category_to_embedding: Dict[str, NDArray[np.float64]],
) -> List[InputExample]:
    """
    Retrieve or generate all training pairs for model training.

    This function handles loading existing pairs from files or generating new ones
    as needed. It manages three types of pairs: positive, partial positive, and
    negative pairs.

    Args:
        df: DataFrame containing the anime/manga data
        use_saved_pairs: Whether to attempt loading existing pairs
        saved_pairs_directory: Base directory for saved pair files
        max_negative_per_row: Maximum negative pairs per entry
        max_partial_positive_per_row: Maximum partial positive pairs per entry
        num_workers: Number of parallel workers for generation
        data_type: Type of data ('anime' or 'manga')
        category_to_embedding: Dictionary mapping categories to embeddings

    Returns:
        List[InputExample]: Combined list of all pair types for training

    Notes:
        - Automatically creates directory structure for pair files
        - Falls back to generation if loading fails or files missing
        - Combines all pair types into a single training set
        - Maintains consistent file naming based on data_type
    """
    positive_pairs: List[InputExample] = []
    partial_positive_pairs: List[InputExample] = []
    negative_pairs: List[InputExample] = []

    # Define file paths
    positive_pairs_file: str = os.path.join(
        saved_pairs_directory, "pairs", data_type, "positive_pairs.csv"
    )
    partial_positive_pairs_file: str = os.path.join(
        saved_pairs_directory, "pairs", data_type, "partial_positive_pairs.csv"
    )
    negative_pairs_file: str = os.path.join(
        saved_pairs_directory, "pairs", data_type, "negative_pairs.csv"
    )

    if use_saved_pairs:
        if os.path.exists(positive_pairs_file):
            print(f"Loading positive pairs from {positive_pairs_file}")
            positive_pairs_df: pd.DataFrame = pd.read_csv(positive_pairs_file)
            positive_pairs = [
                InputExample(texts=[row["text_a"], row["text_b"]], label=row["label"])
                for _, row in positive_pairs_df.iterrows()
            ]

        if os.path.exists(partial_positive_pairs_file):
            print(f"Loading partial positive pairs from {partial_positive_pairs_file}")
            partial_positive_pairs_df: pd.DataFrame = pd.read_csv(
                partial_positive_pairs_file
            )
            partial_positive_pairs = [
                InputExample(texts=[row["text_a"], row["text_b"]], label=row["label"])
                for _, row in partial_positive_pairs_df.iterrows()
            ]

        if os.path.exists(negative_pairs_file):
            print(f"Loading negative pairs from {negative_pairs_file}")
            negative_pairs_df: pd.DataFrame = pd.read_csv(negative_pairs_file)
            negative_pairs = [
                InputExample(texts=[row["text_a"], row["text_b"]], label=row["label"])
                for _, row in negative_pairs_df.iterrows()
            ]

    if (
        not use_saved_pairs
        or not positive_pairs
        or not partial_positive_pairs
        or not negative_pairs
    ):
        print("Generating pairs")
        (
            generated_positive_pairs,
            generated_partial_positive_pairs,
            generated_negative_pairs,
        ) = create_pairs(
            df,
            max_negative_per_row=max_negative_per_row,
            max_partial_positive_per_row=max_partial_positive_per_row,
            category_to_embedding=category_to_embedding,
            partial_threshold=0.5,
            positive_pairs_file=positive_pairs_file,
            partial_positive_pairs_file=partial_positive_pairs_file,
            negative_pairs_file=negative_pairs_file,
            use_saved_pairs=use_saved_pairs,
            num_workers=num_workers,
        )

        if not positive_pairs:
            positive_pairs = generated_positive_pairs
        if not partial_positive_pairs:
            partial_positive_pairs = generated_partial_positive_pairs
        if not negative_pairs:
            negative_pairs = generated_negative_pairs

    pairs: List[InputExample] = positive_pairs + partial_positive_pairs + negative_pairs
    return pairs


def main() -> None:
    """
    Main training function for fine-tuning SentenceTransformer models.

    This function:
    1. Parses command line arguments for training configuration
    2. Sets up model paths and data loading
    3. Generates or loads training pairs
    4. Initializes and configures the model
    5. Sets up training parameters and loss functions
    6. Executes the training loop
    7. Saves the final model

    Command line arguments control all aspects of training including:
    - Model selection and architecture
    - Training hyperparameters
    - Data processing settings
    - Resource allocation
    - Input/output paths

    The function handles the complete training pipeline from data preparation
    through model saving, with appropriate logging and error handling.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a SentenceTransformer model.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-t5-base",
        help="Name of the model to train. Default is 'sentence-t5-base'.",
    )
    parser.add_argument(
        "--use_saved_pairs",
        action="store_true",
        help="Whether to use saved pairs. Default is False.",
    )
    parser.add_argument(
        "--saved_pairs_directory",
        type=str,
        default="model",
        help="Directory to save/load pairs. Default is 'model'.",
    )
    parser.add_argument(
        "--max_negative_per_row",
        type=int,
        default=5,
        help="Maximum number of negative pairs to generate per row. Default is 5.",
    )
    parser.add_argument(
        "--max_partial_positive_per_row",
        type=int,
        default=5,
        help="Maximum number of partial positive pairs to generate per row. Default is 5.",
    )
    parser.add_argument(
        "--output_model_path",
        type=str,
        default="model/fine_tuned_sbert_model",
        help="Path to save the fine-tuned model. Default is 'model/fine_tuned_sbert_model'.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for the optimizer. Default is 2e-5 (0.00002).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=3,
        help="Batch size for training. Default is 3.",
    )
    parser.add_argument(
        "--evaluations_per_epoch",
        type=int,
        default=20,
        help="Number of evaluations per epoch. Default is 20.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of epochs for training. Default is 2.",
    )
    parser.add_argument(
        "--loss_function",
        type=str,
        choices=["cosine", "cosent", "angle"],
        default="cosine",
        help="Loss function to use: 'cosine', 'cosent', or 'angle'. Default is 'cosine'.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=cpu_count() // 4,
        help="Number of workers for multiprocessing. Default is (cpu_count() // 4).",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["anime", "manga"],
        default="anime",
        help="Type of data to train on: 'anime' or 'manga'. Default is 'anime'.",
    )
    parser.add_argument(
        "--use_custom_transformer",
        action="store_true",
        help="Whether to use the custom transformer with GELU activation.",
    )
    args = parser.parse_args()

    # Set the output model path based on data_type
    output_model_path: str = args.output_model_path
    if "anime" in args.output_model_path and args.data_type == "manga":
        output_model_path = output_model_path.replace("anime", "manga")
    elif "manga" in args.output_model_path and args.data_type == "anime":
        output_model_path = output_model_path.replace("manga", "anime")
    elif args.data_type not in args.output_model_path.lower():
        output_model_path = f"{output_model_path}_{args.data_type}"

    # Update the argument with the new output_model_path
    args.output_model_path = output_model_path

    all_genres: set
    all_themes: set
    all_genres, all_themes = get_genres_and_themes(args.data_type)

    # Load a pre-trained Sentence Transformer model for encoding
    encoder_model: SentenceTransformer = SentenceTransformer("sentence-t5-large")

    # Generate embeddings for all categories
    all_categories: List[str] = list(all_genres) + list(all_themes)
    category_embeddings: NDArray[np.float64] = encoder_model.encode(
        all_categories, convert_to_tensor=False
    )

    # Create a mapping from category to its embedding
    category_to_embedding: Dict[str, NDArray[np.float64]] = {
        category: embedding
        for category, embedding in zip(all_categories, category_embeddings)
    }

    # Load your dataset based on data_type
    dataset_path: str = f"model/merged_{args.data_type}_dataset.csv"
    logging.info("Loading dataset from %s", dataset_path)
    df: pd.DataFrame = pd.read_csv(dataset_path)

    # Get the pairs
    pairs: List[InputExample] = get_pairs(
        df,
        use_saved_pairs=args.use_saved_pairs,
        saved_pairs_directory=args.saved_pairs_directory,
        max_negative_per_row=args.max_negative_per_row,
        max_partial_positive_per_row=args.max_partial_positive_per_row,
        num_workers=args.num_workers,
        data_type=args.data_type,
        category_to_embedding=category_to_embedding,
    )

    # Split the pairs into training and validation sets
    train_pairs, val_pairs = train_test_split(pairs, test_size=0.1)

    # Load the SBERT model
    model: SentenceTransformer = create_model(
        "sentence-transformers/" + args.model_name,
        use_custom_transformer=True if args.use_custom_transformer else False,
        max_seq_length=843,
    )
    print(model)

    # Access the Transformer model
    transformer = model._first_module().auto_model  # pylint: disable=protected-access
    print(transformer.encoder.block)

    # Create the evaluator
    evaluator = create_evaluator(val_pairs)

    # Create a DataLoader
    print("Creating DataLoader")
    train_dataloader: DataLoader = DataLoader(
        train_pairs, shuffle=True, batch_size=args.batch_size
    )

    # Calculate the number of batches per epoch
    num_batches_per_epoch: int = len(train_dataloader)

    # Calculate evaluation steps
    evaluation_steps: int = num_batches_per_epoch // args.evaluations_per_epoch

    # Get the loss function
    train_loss = get_loss_function(args.loss_function, model)

    # Fine-tuning the model
    print("Fine-tuning the model")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        evaluation_steps=evaluation_steps,
        output_path=args.output_model_path,
        warmup_steps=evaluation_steps // 2,
        optimizer_params={"lr": args.learning_rate},
    )

    # Save the model
    model.save(args.output_model_path)


if __name__ == "__main__":
    main()
