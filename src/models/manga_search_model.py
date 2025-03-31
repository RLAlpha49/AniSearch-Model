"""
# Manga Search Model

A specialized cross-encoder model for semantic search of manga based on user descriptions.

This module implements a cross-encoder model specifically designed for manga searches.
It matches user-provided textual descriptions or queries with manga entries in the
merged dataset, computing semantic similarity scores to find the most relevant matches.

## Features

- Semantic search using state-of-the-art cross-encoder models
- Optional filtering to exclude light novels
- Batched processing for efficient memory usage with large datasets
- Progress tracking during search operations
- Customizable result count and batch sizes

## Implementation

The implementation extends the `BaseSearchModel` class, adding manga-specific
functionality such as light novel filtering and optimized search parameters.

## Usage Example

```python
from src.models.manga_search_model import MangaSearchModel

# Initialize the model
manga_search = MangaSearchModel(include_light_novels=False)

# Search for manga matching a description
results = manga_search.search(
    query="A story about a boy who becomes the strongest hero",
    num_results=5
)

# Process the results
for result in results:
    print(f"{result['title']} (Score: {result['score']:.2f})")
    print(f"Synopsis: {result['synopsis'][:100]}...")
```
"""

import logging
import os
from typing import List, Dict, Optional, Any, Union
from tqdm import tqdm
import numpy as np

from src.models.base_search_model import BaseSearchModel
from src.utils.constants import (
    MANGA_DATASET_PATH,
    MODEL_NAME,
    NUM_RESULTS,
    DEFAULT_BATCH_SIZE,
)

logger = logging.getLogger(__name__)


class MangaSearchModel(BaseSearchModel):
    """
    A specialized search model for finding manga based on textual descriptions.

    This class extends BaseSearchModel to provide manga-specific search functionality.
    It loads a comprehensive dataset of manga entries and uses a cross-encoder
    model to compute semantic similarity between user queries and manga synopses,
    returning the most relevant matches.

    The model provides additional functionality over the base class:

    - Optional filtering of light novels
    - Customized search parameters for manga content
    - Batch processing for efficient memory usage
    - Progress tracking during search operations

    Attributes:
        include_light_novels (bool): Whether to include light novels in search results
        df (pd.DataFrame): The loaded manga dataset
        id_col (str): Column name for the manga ID in the dataset
        model (CrossEncoder): The cross-encoder model used for scoring
        device (str): The device being used ('cpu', 'cuda', etc.)
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        include_light_novels: bool = False,
    ):
        """
        Initialize the manga search model with the specified parameters.

        This constructor sets up the manga search model by loading the manga dataset
        and initializing the cross-encoder model. It also configures whether light
        novels should be included in search results.

        Args:
            model_name: Name or path of the cross-encoder model to use.
                Defaults to the value specified in constants.MODEL_NAME.
            device: Device to run the model on ('cpu', 'cuda', 'cuda:0', etc.).
                If None, automatically selects the best available device.
            include_light_novels: Whether to include light novels in search results.
                When False, entries with type 'light_novel' will be filtered out.
                Defaults to False.

        Raises:
            FileNotFoundError: If the manga dataset cannot be found
            ValueError: If the model_name is invalid or the model cannot be loaded

        Example:
            ```python
            # Basic initialization with default settings
            manga_model = MangaSearchModel()

            # Initialize with custom model and including light novels
            custom_model = MangaSearchModel(
                model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
                device="cuda",
                include_light_novels=True
            )
            ```
        """
        logger.info("Initializing MangaSearchModel")
        super().__init__(
            dataset_path=MANGA_DATASET_PATH,
            id_column="manga_id",
            model_name=model_name,
            device=device,
            dataset_type="manga",
        )
        self.include_light_novels = include_light_novels
        logger.info(
            "Light novels will %sbe included in search results",
            "" if include_light_novels else "not ",
        )

    def _get_search_dataframe(self):
        """
        Get the filtered dataframe to be used for search operations.

        This internal method applies any necessary filters to the manga dataset
        before search operations. If light novels are set to be excluded, it
        filters out entries with type 'light_novel'.

        Returns:
            pd.DataFrame: The filtered dataframe ready for search operations.
                If no filtering is needed, returns the original dataframe.

        Notes:
            - This is an internal method used by the search method
            - The filtering is only applied if self.include_light_novels is False
              and the dataframe contains a 'type' column
        """
        if not self.include_light_novels and "type" in self.df.columns:
            # Filter out light novels
            logger.info("Filtering out light novels from search results")
            filtered_df = self.df[self.df["type"].str.lower() != "light_novel"]
            logger.info(
                "Filtered out %d light novel entries", len(self.df) - len(filtered_df)
            )
            return filtered_df

        # Return the original dataframe if no filtering is needed
        return self.df

    def search(
        self,
        query: str,
        num_results: int = NUM_RESULTS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> List[Dict[str, Any]]:
        """
        Search for manga entries matching the provided description or query.

        This method computes semantic similarity scores between the provided query
        and all manga synopses in the dataset (after optional filtering), returning
        the top matches sorted by relevance.

        The search process includes:

        1. Optional filtering of the dataset (e.g., removing light novels)
        2. Creating sentence pairs between the query and all manga synopses
        3. Computing relevance scores using the cross-encoder model in batches
        4. Sorting results by score and returning the top matches

        Args:
            query: The search query or description to match against manga synopses.
                This should be a descriptive text that captures the manga content
                the user is looking for.
            num_results: Number of top matches to return, sorted by relevance score.
                Defaults to the value specified in constants.NUM_RESULTS.
            batch_size: Number of sentence pairs to process at once with the model.
                Using batches helps manage memory usage with large datasets.
                Defaults to the value specified in constants.DEFAULT_BATCH_SIZE.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing:
                - id (int): The manga ID
                - title (str): The manga title
                - score (float): The relevance score (higher is better)
                - synopsis (str): A preview of the manga synopsis (truncated to 500 chars)

                The list is sorted by score in descending order.

        Raises:
            ValueError: If the query is empty or consists only of whitespace

        Example:
            ```python
            # Search for manga about time travel
            results = manga_model.search(
                query="A story about characters who can travel through time",
                num_results=3,
                batch_size=64
            )

            # Process the top results
            for result in results:
                print(f"{result['title']} (Score: {result['score']:.2f})")
            ```
        """
        if not query.strip():
            raise ValueError("Search query cannot be empty")

        logger.info("Searching for: %s", query)

        # Get the appropriate dataframe for search
        search_df = self._get_search_dataframe()

        # Prepare pairs for cross-encoder scoring
        all_synopses = search_df["combined_synopsis"].tolist()
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
        scores: List[Any] = []

        with tqdm(
            total=total_pairs,
            desc="Scoring",
            disable=False,
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

        scores_array = np.array(scores)

        # Get indices of top scores
        top_indices = scores_array.argsort()[-num_results:][::-1]

        # Prepare results
        results = []
        for idx in top_indices:
            entry = search_df.iloc[idx]
            synopsis = entry["combined_synopsis"]
            results.append(
                {
                    "id": entry[self.id_col],
                    "title": entry["title"],
                    "score": float(scores_array[idx]),
                    "synopsis": (
                        synopsis[:500] + "..." if len(synopsis) > 500 else synopsis
                    ),
                }
            )

        logger.info("Found %d matches", len(results))
        return results
