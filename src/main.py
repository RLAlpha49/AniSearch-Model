"""
Anime/Manga Description Search Model using Cross-Encoders

This script implements a cross-encoder model to match user-provided descriptions with
anime/manga entries in the merged dataset. It enables semantic search capabilities
by computing relevance scores between queries and entries in the dataset.

Usage:
    # Search mode:
    python src/main.py search --type anime --query "An adventure about pirates searching for treasure"
    python src/main.py search --type manga --query "A story about a boy who becomes a hero"
    python src/main.py search --type anime --interactive  # For interactive mode
    python src/main.py search --type manga --query "Fantasy adventure" --include-light-novels  # Include light novels

    # Training mode:
    python src/main.py train --type anime --model "cross-encoder/ms-marco-MiniLM-L-6-v2" --epochs 3
    python src/main.py train --type manga --model "cross-encoder/ms-marco-MiniLM-L-6-v2" --epochs 3
    python src/main.py train --type anime --create-labeled-data "data/labeled_anime.csv"

The script will return the top matching anime/manga titles based on the query.
"""

# pylint: disable=import-outside-toplevel

import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pylint: disable=wrong-import-position
from src.cli.args import parse_args
from src.utils.logging_config import setup_logging

# Get a logger for this module
logger = logging.getLogger(__name__)


def get_search_model(
    dataset_type: str, model_name: str, include_light_novels: bool = False
):
    """
    Create and return the appropriate search model based on dataset type.

    Args:
        dataset_type: The type of dataset to search ('anime' or 'manga')
        model_name: The name of the model to use
        include_light_novels: Whether to include light novels in manga search results

    Returns:
        An instance of the appropriate search model
    """
    from src.models.anime_search_model import AnimeSearchModel
    from src.models.manga_search_model import MangaSearchModel

    if dataset_type.lower() == "anime":
        return AnimeSearchModel(model_name=model_name)
    if dataset_type.lower() == "manga":
        return MangaSearchModel(
            model_name=model_name, include_light_novels=include_light_novels
        )
    raise ValueError(
        f"Invalid dataset type: {dataset_type}. Must be 'anime' or 'manga'."
    )


def display_models(args):
    """Display available pre-trained cross-encoder models."""
    from src.utils.display import display_available_models
    from src.models.base_search_model import BaseSearchModel

    if args.list_fine_tuned:
        display_available_models(
            fine_tuned_models=BaseSearchModel.list_fine_tuned_models()
        )
    else:
        display_available_models()


def handle_search_command(args) -> None:
    """
    Handle the search command functionality.

    Args:
        args: Parsed command-line arguments for search
    """
    from src.cli.interactive import interactive_mode
    from src.utils.display import format_score

    # Display available models if requested
    if args.list_models or args.list_fine_tuned:
        display_models(args)
        return

    try:
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
                print(f"\n{i+1}. {result['title']} ({score_display})")
                print(f"   ID: {result['id']}")
                print(f"   Synopsis excerpt: {synopsis_excerpt}")
        else:
            print(
                "Error: Either --query, --interactive, --list-models, or --list-fine-tuned must be specified"
            )
    except Exception as e:
        logger.error("Error during search: %s", str(e))
        raise


def handle_train_command(args) -> None:
    """
    Handle the train command functionality.

    Args:
        args: Parsed command-line arguments for training
    """
    from src.training.utils import list_available_models
    from src.training.anime_trainer import AnimeModelTrainer
    from src.training.manga_trainer import MangaModelTrainer

    # Display available models if requested
    if args.list_models:
        list_available_models()
        return

    try:
        # Initialize appropriate trainer based on dataset type
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

    except Exception as e:
        logger.error("Error during training: %s", str(e), exc_info=True)
        raise


def main() -> None:
    """
    Main function to run the search model or trainer.
    """
    # Configure logging
    setup_logging()

    # Parse command-line arguments
    args = parse_args()

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
