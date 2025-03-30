"""
Manga Search Model using Cross-Encoders.

This module implements a cross-encoder model specifically for manga searches,
matching user-provided descriptions with manga entries in the merged dataset.
"""

import logging
from typing import List, Dict, Optional, Any
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

    def _get_search_dataframe(self):
        """
        Get the dataframe to be used for search, applying any necessary filters.

        Returns:
            DataFrame with any required filters applied for search
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
        Search for manga matching the provided description.

        This method extends the base search by optionally filtering out light novels.

        Args:
            query: The search query or description to match
            num_results: Number of top results to return
            batch_size: Batch size for processing with the cross-encoder model

        Returns:
            List of dictionaries containing matched entries with scores
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
        scores = []

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

        scores = np.array(scores)

        # Get indices of top scores
        top_indices = scores.argsort()[-num_results:][::-1]

        # Prepare results
        results = []
        for idx in top_indices:
            entry = search_df.iloc[idx]
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
