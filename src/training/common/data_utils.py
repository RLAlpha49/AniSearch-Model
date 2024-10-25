"""
Utility module for data handling and management in the anime/manga recommendation system.

This module provides utility functions for:
1. Data persistence: Saving and loading training pairs
2. Genre and theme management: Maintaining predefined sets of genres and themes
   for both anime and manga datasets

The module supports two main data types:
- Anime: Contains specific genres and themes for anime content
- Manga: Contains specific genres and themes for manga content

Functions:
    save_pairs_to_csv: Save training pairs (text pairs and their similarity labels) to CSV
    get_genres_and_themes: Retrieve predefined sets of genres and themes based on data type

The genre and theme sets are carefully curated for each data type and are used
for calculating semantic similarities between different entries in the dataset.
These sets are essential for the training process and maintaining consistency
in the recommendation system.

Note:
    The genres and themes are maintained as static sets within the module.
    Updates to these sets should be done with careful consideration of the
    impact on the overall recommendation system.
"""

import os
from typing import Dict, List, Optional, Set, Tuple, Union, Literal
import pandas as pd
from sentence_transformers import InputExample

DataType = Literal["anime", "manga"]


def save_pairs_to_csv(pairs: List[InputExample], filename: Optional[str]) -> None:
    """
    Save pairs of texts and their similarity labels to a CSV file.

    This function takes a list of InputExample objects containing text pairs and their
    similarity labels and saves them to a CSV file. It creates the output directory
    if it doesn't exist.

    Args:
        pairs (List[InputExample]): List of text pairs and their similarity labels
        filename (Optional[str]): Path to save the CSV file

    Returns:
        None

    Raises:
        TypeError: If filename parameter is None
        OSError: If directory creation fails or file cannot be written
    """
    if filename is None:
        raise TypeError("Filename cannot be None")

    # Ensure the directory exists
    directory: str = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    data: Dict[str, List[Union[str, float]]] = {
        "text_a": [pair.texts[0] for pair in pairs],
        "text_b": [pair.texts[1] for pair in pairs],
        "label": [float(pair.label) for pair in pairs],
    }
    pairs_df: pd.DataFrame = pd.DataFrame(data)
    pairs_df.to_csv(filename, index=False)
    print(f"Pairs saved to {filename}")


def get_genres_and_themes(data_type: DataType) -> Tuple[Set[str], Set[str]]:
    """
    Get predefined sets of genres and themes for anime or manga data.

    This function returns two sets containing valid genres and themes based on the
    specified data type. The sets are curated specifically for each content type
    to ensure accurate semantic similarity calculations.

    Args:
        data_type (DataType): Type of data to get genres/themes for ("anime" or "manga")

    Returns:
        Tuple[Set[str], Set[str]]: A tuple containing:
            - First element: Set of all valid genres for the data type
            - Second element: Set of all valid themes for the data type

    Raises:
        ValueError: If data_type is not "anime" or "manga"
    """
    all_genres: Set[str]
    all_themes: Set[str]
    if data_type == "anime":
        all_genres = {
            "Slice of Life",
            "Boys Love",
            "Drama",
            "Suspense",
            "Gourmet",
            "Erotica",
            "Romance",
            "Comedy",
            "Hentai",
            "Sports",
            "Supernatural",
            "Fantasy",
            "Girls Love",
            "Mystery",
            "Adventure",
            "Horror",
            "Award Winning",
            "Action",
            "Avant Garde",
            "Ecchi",
            "Sci-Fi",
        }

        all_themes = {
            "Military",
            "Survival",
            "Idols (Female)",
            "High Stakes Game",
            "Crossdressing",
            "Delinquents",
            "Vampire",
            "Video Game",
            "Action",
            "Adventure",
            "Comedy",
            "Drama",
            "Fantasy",
            "Horror",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Slice of Life",
            "Supernatural",
            "Thriller",
            "Sports",
            "Magical Realism",
            "Mecha",
            "Psychological",
            "Parody",
        }
    elif data_type == "manga":
        all_genres = {
            "Comedy",
            "Romance",
            "Gourmet",
            "Action",
            "Avant Garde",
            "Fantasy",
            "Sports",
            "Sci-Fi",
            "Suspense",
            "Erotica",
            "Adventure",
            "Slice of Life",
            "Ecchi",
            "Supernatural",
            "Horror",
            "Girls Love",
            "Mystery",
            "Award Winning",
            "Drama",
        }

        all_themes = {
            "Martial Arts",
            "Romantic Subtext",
            "Music",
            "Crossdressing",
            "Workplace",
            "Pets",
            "Medical",
            "Adult Cast",
            "Combat Sports",
            "Gag Humor",
            "Reincarnation",
            "Visual Arts",
            "Showbiz",
            "Racing",
            "Iyashikei",
            "Time Travel",
            "CGDCT",
            "Strategy Game",
            "Villainess",
            "Idols (Female)",
            "Gore",
            "Team Sports",
            "Video Game",
            "Super Power",
            "Samurai",
            "Organized Crime",
            "Parody",
            "Childcare",
            "Magical Sex Shift",
            "Love Polygon",
            "Performing Arts",
            "Anthropomorphic",
            "Historical",
            "Vampire",
            "Reverse Harem",
            "Isekai",
            "Mecha",
            "Delinquents",
            "Detective",
            "Idols (Male)",
            "Otaku Culture",
            "Mythology",
            "Military",
            "Mahou Shoujo",
            "High Stakes Game",
            "School",
            "Space",
            "Educational",
            "Psychological",
            "Harem",
            "Memoir",
            "Survival",
        }
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

    return all_genres, all_themes
