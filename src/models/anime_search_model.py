"""
# Anime Search Model

A specialized cross-encoder model for semantic search of anime based on user descriptions.

This module implements a cross-encoder model specifically designed for anime searches.
It matches user-provided textual descriptions or queries with anime entries in the
merged dataset, computing semantic similarity scores to find the most relevant matches.

## Features

- Semantic search using state-of-the-art cross-encoder models
- Efficient processing of large anime datasets
- Customizable result count and device selection
- Integration with the BaseSearchModel framework for consistent behavior

## Implementation

The implementation extends the `BaseSearchModel` class, providing anime-specific
configuration and optimized search parameters.

## Usage Example

```python
from src.models.anime_search_model import AnimeSearchModel

# Initialize the model
anime_search = AnimeSearchModel()

# Search for anime matching a description
results = anime_search.search(
    query="A post-apocalyptic world where humans fight against giant creatures",
    num_results=5
)

# Process the results
for result in results:
    print(f"{result['title']} (Score: {result['score']:.2f})")
    print(f"Synopsis: {result['synopsis'][:100]}...")
```
"""

import logging
from typing import Optional

from src.models.base_search_model import BaseSearchModel
from src.utils.constants import ANIME_DATASET_PATH, MODEL_NAME

logger = logging.getLogger(__name__)


class AnimeSearchModel(BaseSearchModel):
    """
    A specialized search model for finding anime based on textual descriptions.

    This class extends BaseSearchModel to provide anime-specific search functionality.
    It loads a comprehensive dataset of anime entries and uses a cross-encoder
    model to compute semantic similarity between user queries and anime synopses,
    returning the most relevant matches.

    The model uses the merged anime dataset to provide search capabilities across
    a wide range of anime titles with rich metadata and synopses information.

    Attributes:
        df (pd.DataFrame): The loaded anime dataset
        id_col (str): Column name for the anime ID in the dataset
        model (CrossEncoder): The cross-encoder model used for scoring
        device (str): The device being used ('cpu', 'cuda', etc.)
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
    ):
        """
        Initialize the anime search model with the specified parameters.

        This constructor sets up the anime search model by loading the anime dataset
        and initializing the cross-encoder model.

        Args:
            model_name: Name or path of the cross-encoder model to use.
                Defaults to the value specified in constants.MODEL_NAME.
            device: Device to run the model on ('cpu', 'cuda', 'cuda:0', etc.).
                If None, automatically selects the best available device.

        Raises:
            FileNotFoundError: If the anime dataset cannot be found
            ValueError: If the model_name is invalid or the model cannot be loaded

        Example:
            ```python
            # Basic initialization with default settings
            anime_model = AnimeSearchModel()

            # Initialize with custom model and specific device
            custom_model = AnimeSearchModel(
                model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
                device="cuda"
            )
            ```
        """
        logger.info("Initializing AnimeSearchModel")
        super().__init__(
            dataset_path=ANIME_DATASET_PATH,
            id_column="anime_id",
            model_name=model_name,
            device=device,
            dataset_type="anime",
        )
