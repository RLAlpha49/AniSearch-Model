"""
Anime/Manga Description Search Model using Cross-Encoders

This script implements a cross-encoder model to match user-provided descriptions with
anime/manga entries in the merged dataset. It enables semantic search capabilities
by computing relevance scores between queries and entries in the dataset.

The application has two main modes:

1. **Search Mode**: For finding anime/manga that match a description
2. **Training Mode**: For fine-tuning cross-encoder models on anime/manga data

Usage:
    ```bash
    # Search mode:
    python src/main.py search --type anime --query "A story about pirates searching for treasure"
    python src/main.py search --type manga --query "A story about a boy who becomes a hero"
    python src/main.py search --type anime --interactive  # For interactive mode
    python src/main.py search --type manga --query "Fantasy adventure" --include-light-novels

    # Training mode:
    python src/main.py train --type anime --model "cross-encoder/ms-marco-MiniLM-L-6-v2" --epochs 3
    python src/main.py train --type manga --model "cross-encoder/ms-marco-MiniLM-L-6-v2" --epochs 3
    python src/main.py train --type anime --create-labeled-data "data/labeled_anime.csv"
    ```

The script will return the top matching anime/manga titles based on the query.

Attributes:
    logger: Logger instance for the main module
"""

# pylint: disable=import-outside-toplevel

import logging
import os
import sys
from typing import Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pylint: disable=wrong-import-position
from src.cli.args import parse_args
from src.utils.logging_config import setup_logging

# Get a logger for this module
logger = logging.getLogger(__name__)


def handle_model_listing(args: Any) -> None:
    """
    Handle listing models without loading any ML frameworks.

    This is a lightweight function that doesn't import TensorFlow or PyTorch.
    It displays available pre-trained models for search, and optionally
    fine-tuned models if requested.

    Args:
        args: Command-line arguments namespace containing at least the following:

            - list_fine_tuned (bool): Whether to include fine-tuned models in the listing

    Notes:
        This function calls sys.exit(0) after displaying the models to prevent
        loading heavy ML frameworks unnecessarily.

    Example:
        ```python
        # List all pre-trained models
        handle_model_listing(args_with_list_models=True)

        # List both pre-trained and fine-tuned models
        handle_model_listing(args_with_list_fine_tuned=True)
        ```
    """
    from src.utils.display import display_available_models, list_fine_tuned_models

    if args.list_fine_tuned:
        display_available_models(fine_tuned_models=list_fine_tuned_models())
    else:
        display_available_models()

    # Exit after displaying models to prevent any further imports
    sys.exit(0)


def get_search_model(
    dataset_type: str,
    model_name: str,
    device: Optional[str] = None,
    include_light_novels: bool = False,
) -> Any:
    """
    Create and return the appropriate search model based on dataset type.

    This factory function initializes either an AnimeSearchModel or
    MangaSearchModel based on the specified dataset type. For manga
    models, it optionally includes light novels in the dataset.

    Args:
        dataset_type: The type of dataset to search. Must be either 'anime' or 'manga'.
        model_name: The name of the model to use, either a pre-trained model name
            or path to a fine-tuned model.
        device: Device to run the model on ('cpu', 'cuda', 'cuda:0', etc.).
            If None, automatically selects the best available device.
        include_light_novels: Whether to include light novels in manga search results.
            Only applicable when dataset_type is 'manga'. Defaults to False.

    Returns:
        An instance of either AnimeSearchModel or MangaSearchModel.

    Raises:
        ValueError: If dataset_type is not 'anime' or 'manga'.

    Example:
        ```python
        # Create an anime search model
        anime_model = get_search_model('anime', 'cross-encoder/ms-marco-MiniLM-L-6-v2')

        # Create a manga search model with light novels included
        manga_model = get_search_model('manga', 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                                      include_light_novels=True)
        ```
    """
    from src.models.anime_search_model import AnimeSearchModel
    from src.models.manga_search_model import MangaSearchModel

    if dataset_type.lower() == "anime":
        return AnimeSearchModel(model_name=model_name, device=device)
    if dataset_type.lower() == "manga":
        return MangaSearchModel(
            model_name=model_name,
            device=device,
            include_light_novels=include_light_novels,
        )
    raise ValueError(
        f"Invalid dataset type: {dataset_type}. Must be 'anime' or 'manga'."
    )


def handle_search_command(args: Any) -> None:
    """
    Handle the search command functionality.

    This function processes the search command, initializing the appropriate search model
    and executing either interactive search or one-time query search based on arguments.

    The function supports:

    - Interactive mode for continuous querying
    - One-time query with formatted results
    - Customizable number of results and batch size

    Args:
        args: Parsed command-line arguments for search, containing at least:

            - type (str): Dataset type ('anime' or 'manga')
            - model (str): Model name or path
            - include_light_novels (bool): Whether to include light novels (manga only)
            - interactive (bool): Whether to run in interactive mode
            - query (str, optional): The search query text
            - results (int): Number of results to return
            - batch_size (int): Batch size for processing

    Raises:
        Various exceptions might be raised but are handled by the decorator:

            - ValueError: For invalid arguments
            - ImportError: For missing dependencies
            - RuntimeError: For execution failures

    Example:
        ```python
        # Handle a search for anime with a specific query
        args = ArgNamespace(type='anime', model='default_model', query='pirates',
                           interactive=False, results=5, batch_size=32,
                           include_light_novels=False)
        handle_search_command(args)
        ```
    """
    from src.cli.interactive import interactive_mode
    from src.utils.display import format_score
    from src.utils.error_handling import handle_exceptions

    @handle_exceptions(log_exceptions=True, include_exc_info=True, reraise=True)
    def execute_search() -> None:
        """
        Execute the search operation with error handling.

        This inner function is decorated with error handling and performs
        the actual search operation, either in interactive mode or for a
        one-time query.
        """
        # Initialize the appropriate search model based on dataset type
        search_model = get_search_model(
            args.type, args.model, include_light_novels=args.include_light_novels
        )

        if args.interactive:
            interactive_mode(search_model, args.results, args.batch_size)
        elif args.query:
            results = search_model.search(
                args.query,
                args.results,
                args.batch_size,
            )

            print(f"\nTop {len(results)} matches for '{args.query}':")
            for i, result in enumerate(results):
                synopsis_excerpt = (
                    result["synopsis"][:300] + "..."
                    if len(result["synopsis"]) > 300
                    else result["synopsis"]
                )
                score_display = format_score(
                    result["score"],
                    search_model.normalize_scores,
                    search_model.model_name,
                )
                print(f"\n{i + 1}. {result['title']} ({score_display})")
                print(f"   ID: {result['id']}")
                print(f"   Synopsis excerpt: {synopsis_excerpt}")
        else:
            print("Error: Either --query or --interactive must be specified")

    execute_search()


def handle_train_command(args: Any) -> None:
    """
    Handle the train command functionality.

    This function processes the training command, initializing the appropriate model trainer
    and executing either model training or labeled data creation based on arguments.

    The function supports:

    - Training anime or manga models with customizable parameters
    - Creating labeled data without training
    - Using custom labeled data for training
    - Including light novels in manga training

    Args:
        args: Parsed command-line arguments for training, containing at least:

            - type (str): Dataset type ('anime' or 'manga')
            - model (str): Base model name for fine-tuning
            - epochs (int): Number of training epochs
            - batch_size (int): Training batch size
            - eval_steps (int): Steps between evaluations
            - max_samples (int): Maximum number of training samples
            - learning_rate (float): Learning rate for optimizer
            - seed (int): Random seed for reproducibility
            - include_light_novels (bool): Whether to include light novels (manga only)
            - create_labeled_data (str, optional): Path to save labeled data
            - labeled_data (str, optional): Path to custom labeled data
            - loss (str): Loss type ('mse' or 'cosine')
            - scheduler (str): Learning rate scheduler type

    Raises:
        Various exceptions might be raised but are handled by the decorator:

            - ValueError: For invalid arguments
            - ImportError: For missing dependencies
            - RuntimeError: For execution failures
            - FileNotFoundError: For missing files or directories

    Example:
        ```python
        # Handle training an anime model with specified parameters
        args = ArgNamespace(type='anime', model='cross-encoder/ms-marco-MiniLM-L-6-v2',
                           epochs=3, batch_size=16, eval_steps=250, max_samples=10000,
                           learning_rate=2e-5, seed=42, include_light_novels=False,
                           create_labeled_data=None, labeled_data=None, loss='mse',
                           scheduler='linear')
        handle_train_command(args)
        ```
    """
    from src.training.anime_trainer import AnimeModelTrainer
    from src.training.base_trainer import BaseModelTrainer
    from src.training.manga_trainer import MangaModelTrainer
    from src.utils.error_handling import handle_exceptions

    @handle_exceptions(log_exceptions=True, include_exc_info=True, reraise=True)
    def execute_training() -> None:
        """
        Execute the training operation with error handling.

        This inner function is decorated with error handling and performs
        the actual training operation, either for creating labeled data
        or for model training.
        """
        # Initialize appropriate trainer based on dataset type
        trainer: BaseModelTrainer

        if args.type == "anime":
            trainer = AnimeModelTrainer(
                model_name=args.model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                eval_steps=args.eval_steps,
                max_samples=args.max_samples,
                learning_rate=args.learning_rate,
                seed=args.seed,
            )
        else:  # args.type == "manga"
            trainer = MangaModelTrainer(
                model_name=args.model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                eval_steps=args.eval_steps,
                max_samples=args.max_samples,
                learning_rate=args.learning_rate,
                seed=args.seed,
                include_light_novels=args.include_light_novels,
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
            f"  python src/main.py search --type {args.type}",
            f'--model "{output_path}" --query "Your query"',
        )
        print("=" * 50)

    # Execute training with error handling
    execute_training()


def main() -> None:
    """
    Main function to run the search model or trainer.

    This is the entry point for the application. It:

    1. Sets up logging
    2. Parses command-line arguments
    3. Dispatches to the appropriate handler based on command

    The function handles three primary commands:

    - `search`: For finding anime/manga matching a description
    - `train`: For training custom models on anime/manga data
    - Implicit model listing via --list-models or --list-fine-tuned flags

    Returns:
        None

    Raises:
        SystemExit: With code 1 if an unknown command is provided

    Example:
        ```python
        if __name__ == "__main__":
            main()
        ```
    """
    # Configure logging
    setup_logging()

    # Parse command-line arguments
    args = parse_args()

    # Check for model listing arguments first, before any heavy imports
    if (
        hasattr(args, "list_models")
        and args.list_models
        or hasattr(args, "list_fine_tuned")
        and args.list_fine_tuned
    ):
        handle_model_listing(args)
        return  # This won't be reached as handle_model_listing calls sys.exit

    # Handle based on command
    if args.command == "search":
        handle_search_command(args)
    elif args.command == "train":
        handle_train_command(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
