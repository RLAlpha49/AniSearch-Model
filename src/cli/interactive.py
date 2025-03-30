"""
# Interactive Search Mode

A command-line interface for interactive semantic search of anime and manga.

This module provides an interactive command-line interface for searching anime and manga
using semantic search models. It allows users to enter natural language descriptions
and displays formatted search results with titles, IDs, and synopsis excerpts.

## Features

- Interactive query input with continuous search capability
- Formatted display of search results with relevance scores
- Error handling with graceful error recovery
- Customizable number of results and processing batch size
- Support for both anime and manga search models

## Usage

The interactive mode can be started from the main CLI with:

```bash
# For anime search
python -m src.main search --interactive

# For manga search
python -m src.main search --type manga --interactive
```

Once in interactive mode, users can enter descriptive queries and receive
ranked results until they choose to exit the session.
"""

from src.models.base_search_model import BaseSearchModel
from src.utils.display import format_score
from src.utils.error_handling import handle_exceptions


@handle_exceptions(cli_mode=True, reraise=False)
def search_and_display_results(
    search_model: BaseSearchModel,
    query: str,
    num_results: int,
    batch_size: int,
) -> None:
    """
    Execute a search query and display formatted results to the console.

    This function performs a semantic search using the provided search model and
    displays the results in a user-friendly format. It formats each result with
    a title, ID, normalized score, and synopsis excerpt.

    The function is decorated with error handling to gracefully handle any exceptions
    that might occur during the search process or result formatting.

    Args:
        search_model: An initialized search model instance that will perform the search.
            This can be any class that extends BaseSearchModel (like AnimeSearchModel
            or MangaSearchModel).
        query: The text description or query to search for. This should be a natural
            language description of the content the user is looking for.
        num_results: The maximum number of results to return and display.
            This determines how many top matches will be shown to the user.
        batch_size: The batch size to use when processing search pairs with the model.
            Larger batch sizes can improve performance but require more memory.

    Returns:
        None: This function prints results to the console but does not return any values.

    Notes:
        - The function truncates synopsis excerpts to 200 characters for display
        - Scores are formatted differently based on the model type using the format_score utility
        - Results are numbered starting from 1 for user-friendly display
        - Error handling is provided by the @handle_exceptions decorator
    """
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


def interactive_mode(
    search_model: BaseSearchModel,
    num_results: int,
    batch_size: int,
) -> None:
    """
    Run the search model in an interactive command-line session.

    This function starts an interactive search session where users can repeatedly
    enter queries and see results until they choose to exit. It handles the input
    loop, input validation, search execution, and session termination.

    The interactive session continues until the user types 'exit' or 'quit'.
    Each valid query is processed by the search_and_display_results function,
    which executes the search and formats the output.

    Args:
        search_model: An initialized search model instance that will perform the searches.
            This determines whether anime or manga content will be searched.
        num_results: The maximum number of results to return for each query.
            This value is passed to search_and_display_results for each search.
        batch_size: The batch size to use when processing search pairs with the model.
            This value is passed to search_and_display_results for each search.

    Returns:
        None: This function runs an interactive session but doesn't return any values.

    Example:
        ```python
        from src.models.anime_search_model import AnimeSearchModel

        # Initialize a search model
        model = AnimeSearchModel()

        # Start interactive mode with 5 results per query
        interactive_mode(model, num_results=5, batch_size=32)
        ```

    Notes:
        - Empty queries are validated and rejected with a message
        - The function displays which type of dataset is being searched (anime/manga)
        - Error handling for the search process is managed by search_and_display_results
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

        search_and_display_results(search_model, query, num_results, batch_size)
