"""
Anime/Manga Description Search Model using Cross-Encoders

This script implements a cross-encoder model to match user-provided descriptions with
anime/manga entries in the merged dataset. It enables semantic search capabilities
by computing relevance scores between queries and entries in the dataset.

Usage:
    python search_model.py --type anime --query "An adventure about pirates searching for treasure"
    python search_model.py --type manga --query "A story about a boy who becomes a hero"
    python search_model.py --type anime --interactive  # For interactive mode

The script will return the top matching anime/manga titles based on the query.
"""

import os
import argparse
import logging
from typing import List, Dict, Optional, Any, Mapping

import numpy as np
import pandas as pd
import torch
import re
import transformers
from sentence_transformers import CrossEncoder
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Default paths
ANIME_DATASET_PATH = "model/merged_anime_dataset.csv"
MANGA_DATASET_PATH = "model/merged_manga_dataset.csv"
MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"  # A good default cross-encoder model
)
NUM_RESULTS = 5  # Default number of results to return
DEFAULT_BATCH_SIZE = 256  # Default batch size for processing

# Alternative cross-encoder models that can be used
ALTERNATIVE_MODELS = {
    # MS Marco models - Text Ranking
    "ms_marco_models": {
        "ms-marco-MiniLM-L2-v2": "cross-encoder/ms-marco-MiniLM-L2-v2",
        "ms-marco-MiniLM-L4-v2": "cross-encoder/ms-marco-MiniLM-L4-v2",
        "ms-marco-MiniLM-L6-v2": "cross-encoder/ms-marco-MiniLM-L6-v2",  # Default model
        "ms-marco-MiniLM-L12-v2": "cross-encoder/ms-marco-MiniLM-L12-v2",
        "ms-marco-TinyBERT-L2": "cross-encoder/ms-marco-TinyBERT-L2",
        "ms-marco-TinyBERT-L2-v2": "cross-encoder/ms-marco-TinyBERT-L2-v2",
        "ms-marco-TinyBERT-L4": "cross-encoder/ms-marco-TinyBERT-L4",
        "ms-marco-TinyBERT-L6": "cross-encoder/ms-marco-TinyBERT-L6",
        "ms-marco-electra-base": "cross-encoder/ms-marco-electra-base",
    }
}


class AnimeSearchModel:
    """
    A search model using cross-encoders to find anime/manga based on descriptions.

    This class loads a merged dataset of anime or manga entries and uses a cross-encoder
    model to compute relevance scores between a query and entries in the dataset.
    """

    def __init__(
        self,
        dataset_type: str = "anime",
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
    ):
        """
        Initialize the search model.

        Args:
            dataset_type: Type of dataset to use, either 'anime' or 'manga'
            model_name: Name of the cross-encoder model to use
            device: Device to run the model on, either 'cpu', 'cuda', or None (auto-detect)
        """
        self.dataset_type = dataset_type.lower()
        if self.dataset_type not in ["anime", "manga"]:
            raise ValueError("Dataset type must be either 'anime' or 'manga'")

        # Store the model name for later use
        self.model_name = model_name

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load the cross-encoder model
        logger.info("Loading cross-encoder model: %s", model_name)
        try:
            # Check if this is an MS Marco model or fine-tuned from one
            # MS Marco models return logits by default
            is_ms_marco = "ms-marco" in model_name.lower()
            is_fine_tuned = os.path.isdir(model_name) and os.path.exists(
                os.path.join(model_name, "config.json")
            )

            # Special case for fine-tuned models
            if is_fine_tuned:
                logger.info("Loading fine-tuned model from local path: %s", model_name)
                self.model = CrossEncoder(model_name, device=self.device)

                # Check if this was fine-tuned from an MS Marco model
                if "ms-marco" in os.path.basename(model_name).lower():
                    logger.info(
                        "Fine-tuned MS Marco model detected, assuming normalized scores (0-1)"
                    )
                    self.normalize_scores = True
                else:
                    logger.info(
                        "Fine-tuned non-MS Marco model detected, using natural model outputs"
                    )
                    self.normalize_scores = False
            elif is_ms_marco:
                # Use sigmoid activation for MS Marco models to get scores between 0 and 1
                self.model = CrossEncoder(
                    model_name,
                    device=self.device,
                    activation_fn=torch.nn.Sigmoid(),
                )
                logger.info(
                    "MS Marco model detected, using sigmoid activation to normalize scores between 0 and 1"
                )
                self.normalize_scores = True
            else:
                # For other models (like NLI models), use their natural output scale
                self.model = CrossEncoder(model_name, device=self.device)
                logger.info("Non-MS Marco model detected, using natural model outputs")
                self.normalize_scores = False

        except ValueError as e:
            error_message = str(e)
            # Check specifically for unsupported model architecture errors
            if "Transformers does not recognize this architecture" in error_message:
                # Extract model type for better error messaging
                model_type_match = re.search(r"model type `([^`]+)`", error_message)
                model_type = (
                    model_type_match.group(1) if model_type_match else "unknown"
                )

                # Get transformers version for diagnostics
                try:
                    transformers_version = transformers.__version__
                    logger.error(
                        "Unsupported model architecture '%s' in model '%s'. "
                        "Your transformers version is %s. "
                        "Try updating with: pip install --upgrade transformers",
                        model_type,
                        model_name,
                        transformers_version,
                    )

                    # Recommend alternative models
                    logger.info(
                        "Consider using one of these supported models instead: "
                        "cross-encoder/ms-marco-MiniLM-L-6-v2, "
                        "cross-encoder/ms-marco-TinyBERT-L-6, "
                        "or cross-encoder/stsb-distilroberta-base"
                    )

                    raise ValueError(
                        f"Model architecture '{model_type}' is not supported by your transformers "
                        f"version ({transformers_version}). Please try updating transformers or "
                        f"use a supported model instead. See logs for recommendations."
                    ) from e
                except ImportError as exc:
                    # Fallback if transformers version can't be determined
                    logger.error(
                        "Unsupported model architecture in '%s'. "
                        "Try updating transformers: pip install --upgrade transformers",
                        model_name,
                    )
                    raise ValueError(
                        f"Unsupported model architecture in '{model_name}'"
                    ) from exc
            else:
                # For other value errors, just log and re-raise
                logger.error("Failed to load model '%s': %s", model_name, error_message)
                raise ValueError(
                    f"Could not load model '{model_name}'. Error: {error_message}"
                ) from e
        except Exception as e:
            # Handle other types of exceptions
            logger.error("Failed to load model '%s': %s", model_name, str(e))
            raise ValueError(
                f"Could not load model '{model_name}'. Error: {str(e)}"
            ) from e

        # Load dataset
        self.dataset_path = (
            ANIME_DATASET_PATH if self.dataset_type == "anime" else MANGA_DATASET_PATH
        )
        self._load_dataset()

    def _load_dataset(self) -> None:
        """
        Load and prepare the anime/manga dataset.

        Loads the merged dataset and extracts relevant columns for searching.
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}. "
                f"Run 'python src/merge_datasets.py --type {self.dataset_type}' first."
            )

        logger.info("Loading %s dataset from %s", self.dataset_type, self.dataset_path)
        self.df = pd.read_csv(self.dataset_path)
        logger.info("Loaded %d entries", len(self.df))

        # Extract synopsis columns - these will be used for searching
        self.synopsis_cols = [
            col for col in self.df.columns if "synopsis" in col.lower()
        ]
        logger.info(
            "Found %d synopsis columns: %s", len(self.synopsis_cols), self.synopsis_cols
        )

        # Prepare the document corpus by combining all synopsis columns
        self.df["combined_synopsis"] = self.df.apply(
            lambda row: " ".join(
                [str(row[col]) for col in self.synopsis_cols if pd.notna(row[col])]
            ),
            axis=1,
        )

        # Remove entries with empty combined synopsis
        initial_count = len(self.df)
        self.df = self.df[self.df["combined_synopsis"].str.strip() != ""]
        logger.info(
            "Removed %d entries with empty synopsis", initial_count - len(self.df)
        )

        # Required columns: ID, title, and synopsis
        self.id_col = f"{self.dataset_type}_id"

        # Ensure essential columns exist
        required_cols = [self.id_col, "title", "combined_synopsis"]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    def search(
        self,
        query: str,
        num_results: int = NUM_RESULTS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_progress_bar: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search for anime/manga matching the provided description.

        Args:
            query: The search query or description to match
            num_results: Number of top results to return
            batch_size: Batch size for processing with the cross-encoder model
            show_progress_bar: Whether to show a progress bar for larger datasets

        Returns:
            List of dictionaries containing matched entries with scores
        """
        if not query.strip():
            raise ValueError("Search query cannot be empty")

        logger.info("Searching for: %s", query)

        # Prepare pairs for cross-encoder scoring
        all_synopses = self.df["combined_synopsis"].tolist()
        sentence_pairs = [(query, text) for text in all_synopses]

        # Calculate total number of pairs and batches
        total_pairs = len(sentence_pairs)
        if batch_size <= 0:
            batch_size = DEFAULT_BATCH_SIZE
            logger.warning(
                "Invalid batch size provided, using default: %d", DEFAULT_BATCH_SIZE
            )

        # Compute relevance scores in batches
        logger.info(
            "Computing relevance scores with cross-encoder (batch size: %d)", batch_size
        )
        scores = []

        # Use tqdm to display progress for larger datasets
        with tqdm(
            total=total_pairs,
            desc="Scoring",
            disable=not show_progress_bar or total_pairs < 1000,
        ) as pbar:
            for i in range(0, total_pairs, batch_size):
                batch = sentence_pairs[i : i + batch_size]
                # Disable progress_bar in the model's predict method to avoid multiple progress bars
                batch_scores = self.model.predict(batch, show_progress_bar=False)

                if isinstance(batch_scores, np.ndarray):
                    scores.extend(batch_scores.tolist())
                else:
                    scores.extend(batch_scores)

                pbar.update(len(batch))

        # Convert scores to numpy array
        scores = np.array(scores)

        # Get indices of top scores
        top_indices = scores.argsort()[-num_results:][::-1]

        # Prepare results
        results = []
        for idx in top_indices:
            entry = self.df.iloc[idx]
            synopsis = entry["combined_synopsis"]
            results.append(
                {
                    "id": entry[self.id_col],
                    "title": entry["title"],
                    "score": float(scores[idx]),
                    "synopsis": (
                        synopsis[:500] + "..." if len(synopsis) > 500 else synopsis
                    ),
                }
            )

        logger.info("Found %d matches", len(results))
        return results

    @staticmethod
    def list_available_models() -> Mapping[str, Dict[str, str]]:
        """
        List available cross-encoder models that can be used with this search model.

        Returns:
            Dictionary of model categories and their corresponding model names
        """
        return ALTERNATIVE_MODELS

    @staticmethod
    def list_fine_tuned_models() -> Dict[str, str]:
        """
        List available fine-tuned models in the model directory.

        Returns:
            Dictionary of model names and their paths
        """
        fine_tuned_models = {}
        model_dir = "model/fine-tuned"

        if not os.path.exists(model_dir):
            logger.warning("Fine-tuned model directory not found: %s", model_dir)
            return fine_tuned_models

        for model_name in os.listdir(model_dir):
            model_path = os.path.join(model_dir, model_name)
            if os.path.isdir(model_path) and os.path.exists(
                os.path.join(model_path, "config.json")
            ):
                fine_tuned_models[model_name] = model_path

        return fine_tuned_models


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Search for anime/manga based on description using cross-encoder model"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["anime", "manga"],
        required=True,
        help="Type of dataset to search: 'anime' or 'manga'",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Description or query to search for",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"Cross-encoder model to use (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--results",
        type=int,
        default=NUM_RESULTS,
        help=f"Number of results to return (default: {NUM_RESULTS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for processing (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar display",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available cross-encoder models",
    )
    parser.add_argument(
        "--list-fine-tuned",
        action="store_true",
        help="List available fine-tuned models",
    )
    return parser.parse_args()


def format_score(score: float, normalize: bool, model_name: str = "") -> str:
    """
    Format a score for display based on whether it's normalized.

    Args:
        score: The score to format
        normalize: Whether the score is normalized between 0-1
        model_name: Name of the model (used to determine if it's a GooAQ model)

    Returns:
        A formatted string representation of the score
    """
    if normalize:
        # Check if this is a GooAQ model
        is_gooaq = "gooaq" in model_name.lower()

        if is_gooaq:
            # For GooAQ models, multiply by 10 instead of 100
            return f"Match: {score*10:.1f}%"
        # For other normalized scores (0-1), display as percentage
        return f"Match: {score*100:.1f}%"
    # For non-normalized scores, display the raw value
    return f"Score: {score:.2f}"


def interactive_mode(
    search_model: AnimeSearchModel,
    num_results: int,
    batch_size: int,
    show_progress_bar: bool = True,
) -> None:
    """
    Run the search model in interactive mode.

    Args:
        search_model: Initialized search model
        num_results: Number of results to return per query
        batch_size: Batch size for processing
        show_progress_bar: Whether to show progress bar during search
    """
    print(
        f"\nAnime/Manga Search Model - Interactive Mode ({search_model.dataset_type})"
    )
    print("Type 'exit' or 'quit' to end the session\n")

    while True:
        query = input("\nEnter a description to search for: ")
        if query.lower() in ["exit", "quit"]:
            break

        if not query.strip():
            print("Please enter a valid description")
            continue

        try:
            results = search_model.search(
                query,
                num_results=num_results,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
            )
            print("\nTop matches:")
            for i, result in enumerate(results):
                synopsis_excerpt = (
                    result["synopsis"][:200] + "..."
                    if len(result["synopsis"]) > 200
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
        except Exception as e:
            print(f"Error: {str(e)}")


def display_available_models() -> None:
    """Display available cross-encoder models."""
    models = AnimeSearchModel.list_available_models()
    print("\nAvailable Pre-trained Cross-Encoder Models:")
    print("======================================")

    for category, model_dict in models.items():
        print(f"\n{category.upper()}:")
        for name, path in model_dict.items():
            print(f"  {name}: {path}")

    # Display fine-tuned models
    fine_tuned_models = AnimeSearchModel.list_fine_tuned_models()
    if fine_tuned_models:
        print("\nAvailable Fine-tuned Models:")
        print("==========================")
        for name, path in fine_tuned_models.items():
            print(f"  {name}: {path}")

    print("\nUsage example:")
    print(
        '  python src/search_model.py --type anime --query "Your query" --model "cross-encoder/ms-marco-MiniLM-L-6-v2"'
    )
    print("\nTo use a fine-tuned model:")
    print(
        '  python src/search_model.py --type anime --query "Your query" --model "model/fine-tuned/your-model-name"'
    )
    print("\nModel selection guide:")
    print("- TinyBERT models: Smallest and fastest, good for low-resource environments")
    print("- MiniLM models: Good balance of performance and efficiency")
    print("- ELECTRA models: Higher accuracy but more computationally intensive")
    print("- MS Marco models: Optimized for information retrieval")
    print("- Fine-tuned models: Domain-specific models trained on anime/manga data")


def main() -> None:
    """
    Main function to run the search model.
    """
    args = parse_args()

    # Display available models if requested
    if args.list_models:
        display_available_models()
        return

    # Display fine-tuned models only if requested
    if args.list_fine_tuned:
        fine_tuned_models = AnimeSearchModel.list_fine_tuned_models()
        if not fine_tuned_models:
            print("\nNo fine-tuned models available.")
            print("To create fine-tuned models, run:")
            print(
                '  python src/train_model.py --type anime --model "cross-encoder/ms-marco-MiniLM-L-6-v2" --epochs 3'
            )
        else:
            print("\nAvailable Fine-tuned Models:")
            print("==========================")
            for name, path in fine_tuned_models.items():
                print(f"  {name}: {path}")
        return

    try:
        # Initialize search model
        search_model = AnimeSearchModel(
            dataset_type=args.type,
            model_name=args.model,
        )

        if args.interactive:
            interactive_mode(
                search_model, args.results, args.batch_size, not args.no_progress
            )
        elif args.query:
            results = search_model.search(
                args.query,
                args.results,
                args.batch_size,
                show_progress_bar=not args.no_progress,
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
        logger.error("Error: %s", str(e))
        raise


if __name__ == "__main__":
    main()
