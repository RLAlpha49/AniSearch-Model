"""
Anime Search Model using Cross-Encoders.

This module implements a cross-encoder model specifically for anime searches,
matching user-provided descriptions with anime entries in the merged dataset.
"""

import logging
from typing import Optional

from src.models.base_search_model import BaseSearchModel
from src.utils.constants import ANIME_DATASET_PATH, MODEL_NAME

logger = logging.getLogger(__name__)


class AnimeSearchModel(BaseSearchModel):
    """
    A search model using cross-encoders to find anime based on descriptions.

    This class loads a merged dataset of anime entries and uses a cross-encoder
    model to compute relevance scores between a query and entries in the dataset.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
    ):
        """
        Initialize the anime search model.

        Args:
            model_name: Name of the cross-encoder model to use
            device: Device to run the model on, either 'cpu', 'cuda', or None (auto-detect)
        """
        logger.info("Initializing AnimeSearchModel")
        super().__init__(
            dataset_path=ANIME_DATASET_PATH,
            id_column="anime_id",
            model_name=model_name,
            device=device,
            dataset_type="anime",
        )
