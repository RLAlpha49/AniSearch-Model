"""
Manga Search Model using Cross-Encoders.

This module implements a cross-encoder model specifically for manga searches,
matching user-provided descriptions with manga entries in the merged dataset.
"""

import logging
from typing import List, Dict, Optional, Any


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
    A search model using cross-encoders to find manga based on descriptions.

    This class loads a merged dataset of manga entries and uses a cross-encoder
    model to compute relevance scores between a query and entries in the dataset.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        include_light_novels: bool = False,
    ):
        """
        Initialize the manga search model.

        Args:
            model_name: Name of the cross-encoder model to use
            device: Device to run the model on, either 'cpu', 'cuda', or None (auto-detect)
            include_light_novels: Whether to include light novels in search results (default: False)
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

    def search(
        self,
        query: str,
        num_results: int = NUM_RESULTS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> List[Dict[str, Any]]:
        """
        Search for manga matching the provided description.

        This method extends the base search by optionally filtering out light novels.

        Args:
            query: The search query or description to match
            num_results: Number of top results to return
            batch_size: Batch size for processing with the cross-encoder model

        Returns:
            List of dictionaries containing matched entries with scores
        """
        if not self.include_light_novels and "type" in self.df.columns:
            # Filter out light novels if requested
            logger.info("Filtering out light novels from search results")
            original_df = self.df.copy()
            self.df = self.df[self.df["type"].str.lower() != "light_novel"]
            logger.info(
                "Filtered out %d light novel entries", len(original_df) - len(self.df)
            )

        try:
            # Use the base class search method
            results = super().search(
                query=query,
                num_results=num_results,
                batch_size=batch_size,
            )
            return results
        finally:
            # Restore the original dataframe if it was filtered
            if not self.include_light_novels and "type" in self.df.columns:
                self.df = original_df
                logger.info("Restored original dataframe with light novels")
