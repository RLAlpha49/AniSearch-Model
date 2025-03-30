"""
Interactive search mode for the anime search model.
"""

from src.models.base_search_model import BaseSearchModel
from src.utils.display import format_score


def interactive_mode(
    search_model: BaseSearchModel,
    num_results: int,
    batch_size: int,
) -> None:
    """
    Run the search model in interactive mode.

    Args:
        search_model: Initialized search model
        num_results: Number of results to return per query
        batch_size: Batch size for processing
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
