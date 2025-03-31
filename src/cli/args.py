"""
# Command-Line Argument Parser

A robust argument parsing system for the anime/manga search and training commands.

This module provides command-line argument parsing functionality for the application,
supporting both search and training operations for anime and manga datasets. It defines
the available arguments, their defaults, help text, and handles validation of argument
combinations.

## Features

- Subcommand structure with 'search' and 'train' commands
- Comprehensive argument validation to prevent invalid combinations
- Support for interactive and batch search modes
- Extensive configuration options for model training
- Listing capabilities for pre-trained and fine-tuned models

## Command Structure

The CLI supports two main commands:

1. **search**: Search for anime/manga using natural language descriptions
   ```bash
   python -m src.main search --type anime --query "A story about robots in a post-apocalyptic world"
   ```

2. **train**: Train custom cross-encoder models on anime/manga datasets
   ```bash
   python -m src.main train --type anime --epochs 5 --batch-size 32
   ```

## Default Behavior

For backward compatibility, if no command is specified, the system defaults to 'search' mode.
"""

import argparse
import sys

from src.utils.constants import DEFAULT_BATCH_SIZE, MODEL_NAME, NUM_RESULTS


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    Validate command line arguments for consistency and correctness.

    This function ensures that argument combinations are valid and compatible.
    It performs checks such as:

    1. Ensuring the 'type' argument is provided when required
    2. Validating that certain arguments are only used with appropriate dataset types
       (e.g., --include-light-novels only with manga datasets)

    Args:
        args: The parsed command line arguments to validate.
            This is typically the output from argparse.ArgumentParser.parse_args().

    Returns:
        argparse.Namespace: The validated arguments, unchanged if validation passes.

    Raises:
        SystemExit: If validation fails, the program exits with an error message.
            The error message is printed to stderr before exiting.

    Notes:
        - This function is called automatically by parse_args()
        - Some validation rules are specific to certain commands or dataset types
        - The function provides user-friendly error messages to help diagnose issues

    Example:
        ```python
        parser = argparse.ArgumentParser()
        # Add arguments...
        args = parser.parse_args()
        validated_args = validate_args(args)
        # Use validated_args safely...
        ```
    """
    # Check if type is required but missing
    if not (hasattr(args, "list_models") and args.list_models) and not (
        hasattr(args, "list_fine_tuned") and args.list_fine_tuned
    ):
        if not hasattr(args, "type") or not args.type:
            print(
                "Error: --type is required when not listing models",
                file=sys.stderr,
            )
            sys.exit(1)

    # Validate that --include-light-novels is only used with manga dataset type
    if (
        hasattr(args, "include_light_novels")
        and args.include_light_novels
        and hasattr(args, "type")
        and args.type
        and args.type.lower() != "manga"
    ):
        print(
            "Error: --include-light-novels can only be used with --type manga",
            file=sys.stderr,
        )
        sys.exit(1)

    return args


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the anime/manga search and training system.

    This function sets up the argument parser with two main subcommands:

    1. 'search': For finding anime/manga using natural language descriptions
    2. 'train': For training custom cross-encoder models on the datasets

    Each subcommand has its own set of options and arguments. The function handles
    backward compatibility by defaulting to 'search' if no command is provided.

    Returns:
        argparse.Namespace: The parsed and validated command line arguments.
            Contains all the options and arguments specified by the user,
            or their default values if not explicitly provided.

    Notes:
        - For backward compatibility, if no command is provided, 'search' is assumed
        - The 'search' command supports interactive and batch modes
        - The 'train' command has extensive options for model training
        - All arguments are validated by calling validate_args() internally

    Argument Structure:

    ### Common Arguments:
        --type: Dataset type ('anime' or 'manga')
        --model: Model name or path
        --batch-size: Batch size for processing
        --list-models: List available pre-trained models
        --list-fine-tuned: List available fine-tuned models

    ### Search Command Arguments:
        --query: Description to search for
        --results: Number of results to return
        --interactive (-i): Enter interactive mode
        --include-light-novels: Include light novels in manga results

    ### Train Command Arguments:
        --epochs: Number of training epochs
        --eval-steps: Steps between evaluations
        --learning-rate: Learning rate for training
        --max-samples: Maximum number of training samples
        --labeled-data: Path to labeled data CSV
        --create-labeled-data: Create synthetic labeled data
        --seed: Random seed for reproducibility
        --loss: Loss function for training
        --scheduler: Learning rate scheduler

    Example:
        ```python
        # Parse arguments and use them
        args = parse_args()

        if args.command == 'search':
            # Handle search command
            if args.interactive:
                # Run interactive mode
                pass
            else:
                # Run single query search
                pass

        elif args.command == 'train':
            # Handle train command
            pass
        ```
    """
    parser = argparse.ArgumentParser(
        description="Anime/Manga Description Search Model using Cross-Encoders"
    )

    # Add subparsers for search and train commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Create the parser for the "search" command
    search_parser = subparsers.add_parser(
        "search", help="Search for anime/manga by description"
    )
    search_parser.add_argument(
        "--type",
        type=str,
        choices=["anime", "manga"],
        required=False,
        help="Type of dataset to search ('anime' or 'manga')",
    )
    search_parser.add_argument(
        "--query", type=str, help="Description query to search for"
    )
    search_parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"Name of the model to use (default: {MODEL_NAME})",
    )
    search_parser.add_argument(
        "--results",
        type=int,
        default=NUM_RESULTS,
        help=f"Number of results to return (default: {NUM_RESULTS})",
    )
    search_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for processing (default: {DEFAULT_BATCH_SIZE})",
    )
    search_parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Enter interactive mode for continuous queries",
    )
    search_parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available pre-trained cross-encoder models and exit",
    )
    search_parser.add_argument(
        "--list-fine-tuned",
        action="store_true",
        help="List available fine-tuned along with pre-trained models and exit",
    )
    search_parser.add_argument(
        "--include-light-novels",
        action="store_true",
        help="Include light novels in manga search results (manga only)",
    )

    # Create the parser for the "train" command
    train_parser = subparsers.add_parser(
        "train", help="Train a custom cross-encoder model"
    )
    train_parser.add_argument(
        "--type",
        type=str,
        choices=["anime", "manga"],
        required=False,
        help="Type of dataset to train on ('anime' or 'manga')",
    )
    train_parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"Base cross-encoder model to fine-tune (default: {MODEL_NAME})",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16)",
    )
    train_parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Steps between evaluations (default: 500)",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-6,
        help="Learning rate (default: 2e-6)",
    )
    train_parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Maximum number of training samples (default: 10000)",
    )
    train_parser.add_argument(
        "--labeled-data", type=str, help="Path to labeled data CSV file (if available)"
    )
    train_parser.add_argument(
        "--create-labeled-data",
        type=str,
        help="Create and save synthetic labeled data to the specified CSV path",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    train_parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available base models for fine-tuning",
    )
    train_parser.add_argument(
        "--list-fine-tuned",
        action="store_true",
        help="List available fine-tuned along with pre-trained models and exit",
    )
    train_parser.add_argument(
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
    train_parser.add_argument(
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
    train_parser.add_argument(
        "--include-light-novels",
        action="store_true",
        default=True,
        help="Include light novels in manga training data (manga only)",
    )

    # Set default command to "search" for backward compatibility
    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1][0] == "-"):
        sys.argv.insert(1, "search")

    args = parser.parse_args()
    return validate_args(args)
