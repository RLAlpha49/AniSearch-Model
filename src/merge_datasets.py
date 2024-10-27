"""
Merges multiple anime or manga datasets into a single dataset.

This script provides functionality to merge and preprocess multiple anime or manga datasets
from various sources. It handles data cleaning, deduplication, and consolidation of
information across datasets.

Key features:
    - Loads datasets from CSV files and Hugging Face datasets library
    - Cleans and preprocesses text fields like titles and synopses
    - Merges datasets based on common identifiers while handling duplicates
    - Consolidates information from multiple sources while preserving data quality
    - Removes inappropriate content based on genres/demographics
    - Saves the final merged dataset with progress tracking

The script can be run from the command line with a required --type argument
specifying either 'anime' or 'manga'.

Example:
```
python merge_datasets.py --type anime
```

The merged dataset will be saved to model/merged_[type]_dataset.csv
"""

import ast
import os
import argparse
import logging
from logging.handlers import RotatingFileHandler
import sys
import re
from typing import Any, Optional
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import common  # pylint: disable=wrong-import-position

FILE_LOGGING_LEVEL = logging.DEBUG
CONSOLE_LOGGING_LEVEL = logging.INFO

# Create logs directory if not available and configure RotatingFileHandler with UTF-8 encoding
if not os.path.exists("./logs"):
    os.makedirs("./logs")

file_handler = RotatingFileHandler(
    "./logs/merge_datasets.log",
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=10,
    encoding="utf-8",
)
file_handler.setLevel(FILE_LOGGING_LEVEL)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

# Configure StreamHandler with UTF-8 encoding
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(CONSOLE_LOGGING_LEVEL)
stream_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(stream_formatter)

# Initialize logging with both handlers
logging.basicConfig(
    level=FILE_LOGGING_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[file_handler, stream_handler],
)


# Parse command-line arguments
def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for dataset type selection.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            type (str): Either 'anime' or 'manga' to specify dataset type to merge
    """
    parser = argparse.ArgumentParser(
        description="Merge anime or manga datasets into a single dataset."
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["anime", "manga"],
        required=True,
        help="Type of dataset to generate: 'anime' or 'manga'.",
    )
    return parser.parse_args()


# Function to clean synopses
def clean_synopsis(df: pd.DataFrame, synopsis_col: str, unwanted_phrases: list) -> None:
    """
    Clean synopsis text by removing entries containing unwanted phrases.

    Args:
        df: DataFrame containing the synopsis column
        synopsis_col: Name of the column containing synopsis text
        unwanted_phrases: List of phrases that indicate invalid synopsis content

    The function modifies the DataFrame in-place, setting matching synopses to empty strings.
    """
    logging.info("Cleaning synopses in column: %s", synopsis_col)
    for index, row in df.iterrows():
        if pd.notna(row[synopsis_col]):
            for phrase in unwanted_phrases:
                if phrase in row[synopsis_col]:
                    df.at[index, synopsis_col] = ""


def remove_numbered_list_synopsis(df: pd.DataFrame, synopsis_cols: list[str]) -> None:
    """
    Remove synopsis entries that are formatted as numbered lists.

    Args:
        df: DataFrame containing the synopsis columns
        synopsis_cols: List of column names containing synopsis text

    The function modifies the DataFrame in-place, setting numbered list synopses to empty strings.
    Uses regex pattern matching to identify numbered list formats.
    """
    logging.info("Removing numbered list synopses in columns: %s", synopsis_cols)
    numbered_list_pattern = re.compile(
        r"(?s)^.*?(\d+[-\d]*[.)]\s+.+?)(?:\n|$)", re.MULTILINE
    )

    for col in synopsis_cols:
        logging.info("Removing numbered list synopses in column: %s", col)
        df[col] = df[col].apply(
            lambda x: "" if pd.notna(x) and numbered_list_pattern.match(x) else x
        )


# Function to consolidate titles into a single 'title' column
def consolidate_titles(df: pd.DataFrame, title_columns: list) -> pd.Series:
    """
    Consolidate multiple title columns into a single title column.

    Args:
        df: DataFrame containing multiple title columns
        title_columns: List of column names containing titles to consolidate

    Returns:
        pd.Series: Consolidated titles, using first non-null value found across columns

    The function prioritizes existing 'title' column if present, then fills missing values
    from other title columns in order. Empty strings and 'unknown title' are treated as null.
    """
    logging.info("Consolidating titles into a single 'title' column.")
    if "title" in df.columns:
        consolidated_title = df["title"]
        logging.info("Found existing 'title' column.")
    else:
        consolidated_title = pd.Series([""] * len(df), index=df.index)
        logging.info("Initialized 'title' column as empty.")

    for col in title_columns:
        if col in df.columns:
            logging.info("Consolidating title from column: %s", col)
            consolidated_title = consolidated_title.where(
                consolidated_title.notna(), df[col]
            )
        else:
            logging.warning("Title column '%s' not found in DataFrame.", col)

    consolidated_title.replace(["", "unknown title"], pd.NA, inplace=True)
    missing_titles = consolidated_title.isna().sum()
    if missing_titles > 0:
        logging.warning(
            "Found %d entries with missing titles after consolidation.", missing_titles
        )
    else:
        logging.info("All titles consolidated successfully.")
    return consolidated_title


# Preprocess names by converting to lowercase and stripping whitespace
def preprocess_name(name: Any) -> str:
    """
    Preprocess a name string for consistent matching.

    Args:
        name: Input name value of any type

    Returns:
        str: Preprocessed name in lowercase with whitespace stripped
            Returns empty string if input is null/NaN

    Used to standardize names across datasets before matching/merging.
    """
    if pd.isna(name):
        return ""
    return str(name).strip().lower()


def preprocess_synopsis_columns(df: pd.DataFrame, synopsis_cols: list[str]) -> None:
    """
    Preprocess text in synopsis columns for consistency.

    Args:
        df: DataFrame containing synopsis columns
        synopsis_cols: List of column names containing synopsis text

    Applies common text preprocessing to each synopsis column in-place.
    Uses common.preprocess_text() for standardization.
    Logs warning if specified column not found.
    """
    logging.info("Preprocessing synopsis columns: %s", synopsis_cols)
    for col in synopsis_cols:
        if col in df.columns:
            logging.info("Preprocessing column: %s", col)
            df[col] = df[col].apply(common.preprocess_text)
        else:
            logging.warning("Synopsis column '%s' not found in DataFrame.", col)


def find_additional_info(
    row: pd.Series,
    additional_df: pd.DataFrame,
    description_col: str,
    name_columns: list,
) -> Optional[str]:
    """
    Find matching description information from additional dataset.

    Args:
        row: Series containing title information to match
        additional_df: DataFrame containing additional descriptions
        description_col: Name of column containing descriptions
        name_columns: List of column names to use for matching titles

    Returns:
        str | None: Matching description if found, None otherwise

    Attempts to match titles across multiple name columns and returns first matching description.
    """
    for merged_name_col in ["title", "title_english", "title_japanese"]:
        if pd.isna(row[merged_name_col]) or row[merged_name_col] == "":
            continue
        for additional_name_col in name_columns:
            if row[merged_name_col] in additional_df[additional_name_col].values:
                info = additional_df.loc[
                    additional_df[additional_name_col] == row[merged_name_col],
                    description_col,
                ]
                if isinstance(info, pd.Series):
                    info = info.dropna().iloc[0] if not info.dropna().empty else None
                    if info:
                        logging.debug(
                            "Found additional info for '%s' from column '%s'.",
                            row[merged_name_col],
                            description_col,
                        )
                        return info
    logging.debug(
        "No additional info found for row with titles: %s, %s, %s.",
        row.get("title", ""),
        row.get("title_english", ""),
        row.get("title_japanese", ""),
    )
    return None


# Function to add additional synopses or descriptions
def add_additional_info(
    merged: pd.DataFrame,
    additional_df: pd.DataFrame,
    description_col: str,
    name_columns: list[str],
    new_synopsis_col: str,
) -> pd.DataFrame:
    """
    Add additional synopsis information from supplementary dataset.

    Args:
        merged: Main DataFrame to update with additional info
        additional_df: DataFrame containing additional descriptions
        description_col: Name of column containing descriptions
        name_columns: List of columns to use for matching titles
        new_synopsis_col: Name for new column to store additional synopses

    Returns:
        pd.DataFrame: Updated DataFrame with additional synopsis information

    Matches entries between datasets and adds non-duplicate synopsis information.
    Uses tqdm for progress tracking during updates.
    """
    logging.info("Adding additional info to column: %s", new_synopsis_col)
    if new_synopsis_col not in merged.columns:
        merged[new_synopsis_col] = pd.NA
        logging.info("Initialized new synopsis column: %s", new_synopsis_col)

    for idx, row in tqdm(
        merged.iterrows(),
        total=merged.shape[0],
        desc=f"Adding additional info from '{new_synopsis_col}'",
    ):
        if pd.isna(row[new_synopsis_col]):
            info = find_additional_info(
                row, additional_df, description_col, name_columns
            )
            if info:
                merged.at[idx, new_synopsis_col] = info
                logging.debug(
                    "Added info to row %d in column '%s'.", idx, new_synopsis_col
                )

    added_count = merged[new_synopsis_col].notna().sum()
    logging.info(
        "Added %d entries to column '%s'.",
        added_count,
        new_synopsis_col,
    )
    return merged


# Function to remove duplicate information from specified columns
def remove_duplicate_infos(df: pd.DataFrame, info_cols: list[str]) -> pd.DataFrame:
    """
    Remove duplicate synopsis/description entries across columns.

    Args:
        df: DataFrame containing synopsis columns
        info_cols: List of column names containing synopsis information

    Returns:
        pd.DataFrame: DataFrame with duplicate synopses removed

    Keeps first occurrence of each unique synopsis and sets duplicates to NA.
    Processes row-by-row to maintain data integrity.
    """
    for index, row in df.iterrows():
        unique_infos = set()
        for col in info_cols:
            if pd.notna(row[col]) and row[col] not in unique_infos:
                unique_infos.add(row[col])
            else:
                df.at[index, col] = pd.NA
                logging.debug(
                    "Removed duplicate info for row %d in column '%s'.", index, col
                )
    logging.info("Duplicate removal completed.")
    return df


# Function to merge anime datasets
def merge_anime_datasets() -> pd.DataFrame:
    """
    Merge multiple anime datasets into a single comprehensive dataset.

    Returns:
        pd.DataFrame: Merged and cleaned anime dataset

    Performs the following operations:
        - Loads multiple anime datasets from files and Hugging Face
        - Cleans and standardizes text fields
        - Removes adult content and kids' content
        - Merges datasets based on IDs and titles
        - Consolidates synopsis information
        - Removes duplicates
        - Saves final dataset to CSV with progress tracking

    Raises:
        Exception: If any error occurs during merging process
    """
    logging.info("Starting to merge anime datasets.")
    try:
        # Load datasets
        logging.info("Loading anime datasets from CSV files.")
        myanimelist_dataset: pd.DataFrame = pd.read_csv("data/anime/Anime.csv")
        anime_dataset_2023: pd.DataFrame = pd.read_csv(
            "data/anime/anime-dataset-2023.csv"
        )
        animes: pd.DataFrame = pd.read_csv("data/anime/animes.csv")
        anime_4500: pd.DataFrame = pd.read_csv("data/anime/anime4500.csv")
        anime_2022: pd.DataFrame = pd.read_csv("data/anime/Anime-2022.csv")
        anime_data: pd.DataFrame = pd.read_csv("data/anime/Anime_data.csv")
        anime2: pd.DataFrame = pd.read_csv("data/anime/anime2.csv")
        mal_anime: pd.DataFrame = pd.read_csv("data/anime/mal_anime.csv")

        # Load using the datasets library
        logging.info("Loading 'anime_270' dataset from Hugging Face datasets.")
        anime_270 = load_dataset("johnidouglas/anime_270", split="train")
        anime_270_df: pd.DataFrame = anime_270.to_pandas()  # type: ignore

        logging.info("Loading 'wykonos/anime' dataset from Hugging Face datasets.")
        wykonos_dataset = load_dataset("wykonos/anime", split="train")
        wykonos_dataset_df: pd.DataFrame = wykonos_dataset.to_pandas()  # type: ignore

        # Drop specified columns from myanimelist_dataset
        columns_to_drop: list[str] = [
            "scored_by",
            "source",
            "members",
            "favorites",
            "start_date",
            "end_date",
            "episode_duration",
            "total_duration",
            "rating",
            "sfw",
            "approved",
            "created_at",
            "updated_at",
            "real_start_date",
            "real_end_date",
            "broadcast_day",
            "broadcast_time",
            "studios",
            "producers",
            "licensors",
        ]
        logging.info("Dropping unnecessary columns from 'myanimelist_dataset'.")
        myanimelist_dataset.drop(columns=columns_to_drop, inplace=True, errors="ignore")

        # Remove row if 'type' is 'Music'
        myanimelist_dataset = myanimelist_dataset[
            myanimelist_dataset["type"] != "music"
        ]

        # Remove row if 'demographics' contains 'Kids'
        myanimelist_dataset = myanimelist_dataset[
            ~myanimelist_dataset["demographics"].apply(
                lambda x: any(genre in ["Kids"] for genre in ast.literal_eval(x))
            )
        ]

        # Check for duplicates in the keys and remove them
        duplicate_checks: list[tuple[str, pd.DataFrame, str]] = [
            ("anime_id", myanimelist_dataset, "myanimelist_dataset"),
            ("anime_id", anime_dataset_2023, "anime_dataset_2023"),
            ("uid", animes, "animes"),
            ("ID", anime_2022, "anime_2022"),
        ]

        for key, df, name in duplicate_checks:
            if df[key].duplicated().any():
                logging.warning(
                    "Duplicate '%s' found in %s. Removing duplicates.", key, name
                )
                df.drop_duplicates(subset=key, inplace=True)
                df.to_csv(f"data/anime/{name}.csv", index=False)
                logging.info("Duplicates removed and updated '%s.csv'.", name)

        # Preprocess names for matching
        logging.info("Preprocessing names for matching.")
        preprocess_columns: dict[str, list[str]] = {
            "myanimelist_dataset": ["title", "title_english", "title_japanese"],
            "anime_dataset_2023": ["Name", "English name", "Other name"],
            "anime_4500": ["Title"],
            "wykonos_dataset_df": ["Name", "Japanese_name"],
            "anime_data": ["Name"],
            "anime2": ["Name"],
            "mal_anime": ["title"],
        }

        for df_name, cols in preprocess_columns.items():
            df = locals()[df_name]
            for col in cols:
                if col in df.columns:
                    logging.info("Preprocessing column '%s' in '%s'.", col, df_name)
                    df[col] = df[col].apply(preprocess_name)

        # Clean synopses in specific datasets
        logging.info("Cleaning synopses in specific datasets.")
        unwanted_phrases = sorted(
            [
                "A song",
                "A music video",
                "A new music video",
                "A series animated music video",
                "A short animation",
                "A short film",
                "A special music video",
                "An animated music",
                "An animated music video",
                "An animation",
                "An educational film",
                "An independent music",
                "An original song",
                "Animated music video",
                "Minna uta",
                "Minna Uta",
                "Music clip",
                "Music video",
                "No description available for this anime.",
                "No synopsis has been added for this series yet.",
                "No synopsis information has been added to this title.",
                "No synopsis yet",
                "Official music video",
                "Short film",
                "The animated film",
                "The animated music video",
                "The music video",
                "The official music",
                "This music video",
                "Unknown",
            ]
        )

        clean_synopsis(anime_dataset_2023, "Synopsis", unwanted_phrases)
        clean_synopsis(anime_2022, "Synopsis", unwanted_phrases)
        clean_synopsis(wykonos_dataset_df, "Description", unwanted_phrases)
        clean_synopsis(anime_data, "Description", unwanted_phrases)
        clean_synopsis(anime2, "Description", unwanted_phrases)
        clean_synopsis(mal_anime, "synopsis", unwanted_phrases)
        clean_synopsis(animes, "synopsis", unwanted_phrases)
        clean_synopsis(myanimelist_dataset, "synopsis", unwanted_phrases)

        # Merge datasets on 'anime_id'
        logging.info("Merging 'myanimelist_dataset' with 'anime_dataset_2023'.")
        final_merged_df: pd.DataFrame = pd.merge(
            myanimelist_dataset,
            anime_dataset_2023[["anime_id", "Synopsis", "Name"]].rename(
                columns={"Name": "title_anime_dataset_2023"}
            ),
            on="anime_id",
            how="outer",
        )
        final_merged_df.rename(
            columns={"Synopsis": "Synopsis anime_dataset_2023"}, inplace=True
        )
        logging.info("Dropped 'ID' and other unnecessary columns after first merge.")
        final_merged_df.drop(columns=["ID"], inplace=True, errors="ignore")

        logging.info("Merging with 'animes' dataset on 'uid'.")
        final_merged_df = pd.merge(
            final_merged_df,
            animes[["uid", "synopsis", "title"]].rename(
                columns={"title": "title_animes"}
            ),
            left_on="anime_id",
            right_on="uid",
            how="outer",
            suffixes=("", "_animes"),
        )
        final_merged_df.drop(columns=["uid"], inplace=True, errors="ignore")
        final_merged_df.rename(
            columns={"synopsis_animes": "Synopsis animes dataset"}, inplace=True
        )

        logging.info("Merging with 'anime_270_df' dataset on 'MAL_ID'.")
        final_merged_df = pd.merge(
            final_merged_df,
            anime_270_df[["MAL_ID", "sypnopsis", "Name"]].rename(
                columns={"Name": "title_anime_270"}
            ),
            left_on="anime_id",
            right_on="MAL_ID",
            how="outer",
        )
        final_merged_df.rename(
            columns={"sypnopsis": "Synopsis anime_270 Dataset"}, inplace=True
        )
        final_merged_df.drop(columns=["MAL_ID"], inplace=True, errors="ignore")

        logging.info("Merging with 'anime_2022' dataset on 'ID'.")
        final_merged_df = pd.merge(
            final_merged_df,
            anime_2022[["ID", "Synopsis", "Title"]].rename(
                columns={"Title": "title_anime_2022"}
            ),
            left_on="anime_id",
            right_on="ID",
            how="outer",
        )
        final_merged_df.rename(
            columns={"Synopsis": "Synopsis Anime-2022 Dataset"}, inplace=True
        )
        final_merged_df.drop(columns=["ID"], inplace=True, errors="ignore")

        # Consolidate all title columns into a single 'title' column
        logging.info("Consolidating all title columns into a single 'title' column.")
        title_columns: list[str] = [
            "title_anime_dataset_2023",
            "title_animes",
            "title_anime_270",
            "title_anime_2022",
        ]
        final_merged_df["title"] = consolidate_titles(final_merged_df, title_columns)

        # Drop redundant title columns
        logging.info("Dropping redundant title columns: %s", title_columns)
        final_merged_df.drop(columns=title_columns, inplace=True, errors="ignore")

        # Update the merged dataset with additional synopses from various sources
        logging.info("Adding additional synopses from various sources.")
        final_merged_df = add_additional_info(
            final_merged_df,
            anime_4500,
            "Description",
            ["Title"],
            "Synopsis anime4500 Dataset",
        )
        final_merged_df = add_additional_info(
            final_merged_df,
            wykonos_dataset_df,
            "Description",
            ["Name", "Japanese_name"],
            "Synopsis wykonos Dataset",
        )
        final_merged_df = add_additional_info(
            final_merged_df,
            anime_data,
            "Description",
            ["Name"],
            "Synopsis Anime_data Dataset",
        )
        final_merged_df = add_additional_info(
            final_merged_df,
            anime2,
            "Description",
            ["Name", "Japanese_name"],
            "Synopsis anime2 Dataset",
        )
        final_merged_df = add_additional_info(
            final_merged_df,
            mal_anime,
            "synopsis",
            ["title"],
            "Synopsis mal_anime Dataset",
        )

        synopsis_cols: list[str] = [
            "synopsis",
            "Synopsis anime_dataset_2023",
            "Synopsis animes dataset",
            "Synopsis anime_270 Dataset",
            "Synopsis Anime-2022 Dataset",
            "Synopsis anime4500 Dataset",
            "Synopsis wykonos Dataset",
            "Synopsis Anime_data Dataset",
            "Synopsis anime2 Dataset",
            "Synopsis mal_anime Dataset",
        ]
        preprocess_synopsis_columns(final_merged_df, synopsis_cols)

        logging.info("Removing duplicate synopses across columns: %s", synopsis_cols)
        final_merged_df = remove_duplicate_infos(final_merged_df, synopsis_cols)

        # Remove duplicates based on 'anime_id'
        logging.info("Removing duplicates based on 'anime_id'.")
        final_merged_df.drop_duplicates(subset=["anime_id"], inplace=True)

        # Remove rows with all empty or NaN synopsis columns
        logging.info("Removing rows with all empty or NaN synopsis columns.")
        initial_row_count = len(final_merged_df)
        final_merged_df = final_merged_df[
            final_merged_df[synopsis_cols].apply(
                lambda x: x.str.strip().replace("", pd.NA).notna().any(), axis=1
            )
        ]
        removed_rows = initial_row_count - len(final_merged_df)
        logging.info(
            "Removed %d rows with all empty or NaN synopsis columns.", removed_rows
        )

        # Save the updated merged dataset with a progress bar
        logging.info(
            "Saving the merged anime dataset to 'model/merged_anime_dataset.csv'."
        )
        chunk_size: int = 1000
        total_chunks: int = (len(final_merged_df) // chunk_size) + 1

        with open(
            "model/merged_anime_dataset.csv", "w", newline="", encoding="utf-8"
        ) as f:
            # Write the header
            final_merged_df.iloc[:0].to_csv(f, index=False)
            for i in tqdm(range(total_chunks), desc="Saving to CSV"):
                start: int = i * chunk_size
                end: int = start + chunk_size
                final_merged_df.iloc[start:end].to_csv(f, header=False, index=False)

        logging.info(
            "Anime datasets merged and saved to 'model/merged_anime_dataset.csv'."
        )
        return final_merged_df
    except Exception as e:
        logging.error(
            "An error occurred while merging anime datasets: %s", e, exc_info=True
        )
        raise


# Function to merge manga datasets
def merge_manga_datasets() -> pd.DataFrame:
    """
    Merge multiple manga datasets into a single comprehensive dataset.

    Returns:
        pd.DataFrame: Merged and cleaned manga dataset

    Performs the following operations:
        - Loads manga datasets from multiple CSV files
        - Removes adult content
        - Cleans and standardizes text fields
        - Merges datasets based on IDs and titles
        - Consolidates synopsis information
        - Removes duplicates
        - Saves final dataset to CSV with progress tracking

    Raises:
        Exception: If any error occurs during merging process
    """
    logging.info("Starting to merge manga datasets.")
    try:
        # Load datasets
        logging.info("Loading manga datasets from CSV files.")
        manga_main: pd.DataFrame = pd.read_csv("data/manga/manga.csv")  # Base dataset
        jikan: pd.DataFrame = pd.read_csv("data/manga/jikan.csv")
        data: pd.DataFrame = pd.read_csv("data/manga/data.csv")

        # Drop specified columns from manga_main if necessary
        columns_to_drop: list[str] = [
            "scored_by",
            "members",
            "favorites",
            "end_date",
            "sfw",
            "approved",
            "created_at",
            "updated_at",
            "real_start_date",
            "real_end_date",
            "authors",
            "serializations",
        ]
        logging.info("Dropping unnecessary columns from 'manga_main' dataset.")
        manga_main.drop(columns=columns_to_drop, inplace=True, errors="ignore")

        # Identify and remove rows with 'genres' containing 'Hentai' or 'Boys Love'
        logging.info("Identifying IDs with 'Hentai' or 'Boys Love' genres.")
        removed_ids = set(
            manga_main[
                manga_main["genres"].apply(
                    lambda x: any(
                        genre in ["Hentai", "Boys Love"]
                        for genre in ast.literal_eval(x)
                    )
                )
            ]["manga_id"]
        )
        logging.info("Removing rows with 'Hentai' or 'Boys Love' genres.")
        manga_main = manga_main[~manga_main["manga_id"].isin(removed_ids)]

        # Check for duplicates in the keys and remove them
        duplicate_checks: list[tuple[str, pd.DataFrame, str]] = [
            ("manga_id", manga_main, "manga_main"),
            ("mal_id", jikan, "jikan"),
            ("title", data, "data"),
        ]

        for key, df, name in duplicate_checks:
            if df[key].duplicated().any():
                logging.warning(
                    "Duplicate '%s' found in %s. Removing duplicates.", key, name
                )
                df.drop_duplicates(subset=key, inplace=True)
                df.to_csv(f"data/manga/{name}.csv", index=False)
                logging.info("Duplicates removed and updated '%s.csv'.", name)

        # Preprocess names for matching
        logging.info("Preprocessing names for matching.")
        preprocess_columns: dict[str, list[str]] = {
            "manga_main": ["title", "title_english", "title_japanese"],
            "jikan": ["title"],
            "data": ["title"],
        }

        for df_name, cols in preprocess_columns.items():
            df = locals()[df_name]
            for col in cols:
                if col in df.columns:
                    logging.info("Preprocessing column '%s' in '%s'.", col, df_name)
                    df[col] = df[col].apply(preprocess_name)

        # Clean synopses in specific datasets
        logging.info("Cleaning synopses in specific datasets.")
        clean_synopsis(manga_main, "synopsis", ["No synopsis"])
        clean_synopsis(
            data, "description", ["This entry currently doesn't have a synopsis."]
        )
        clean_synopsis(jikan, "synopsis", ["Looking for information on the"])
        clean_synopsis(jikan, "synopsis", ["No synopsis"])

        # Merge main dataset with jikan on 'manga_id' and 'mal_id'
        logging.info(
            "Merging 'manga_main' with 'jikan' dataset on 'manga_id' and 'mal_id'."
        )
        merged_df: pd.DataFrame = pd.merge(
            manga_main,
            jikan[~jikan["mal_id"].isin(removed_ids)][
                ["mal_id", "synopsis", "title"]
            ].rename(columns={"title": "title_jikan"}),
            left_on="manga_id",
            right_on="mal_id",
            how="outer",
            suffixes=("", "_jikan"),
        )
        merged_df.rename(
            columns={"synopsis_jikan": "Synopsis jikan Dataset"}, inplace=True
        )
        merged_df.drop(columns=["mal_id", "title_jikan"], inplace=True, errors="ignore")
        logging.info("Dropped 'mal_id' and 'title_jikan' after first merge.")

        # Merge with data on title
        logging.info("Merging with 'data' dataset on 'title'.")
        merged_df = add_additional_info(
            merged_df,
            data[~data["title"].isin(removed_ids)],
            "description",
            ["title"],
            "Synopsis data Dataset",
        )

        info_cols: list[str] = [
            "synopsis",
            "Synopsis jikan Dataset",
            "Synopsis data Dataset",
        ]
        preprocess_synopsis_columns(merged_df, info_cols)

        remove_numbered_list_synopsis(merged_df, info_cols)

        logging.info("Removing duplicate synopses and descriptions.")
        merged_df = remove_duplicate_infos(merged_df, info_cols)

        # Remove duplicates based on 'manga_id'
        logging.info("Removing duplicates based on 'manga_id'.")
        merged_df.drop_duplicates(subset=["manga_id"], inplace=True)

        # Remove rows with all empty or NaN synopsis columns
        logging.info("Removing rows with all empty or NaN synopsis columns.")
        initial_row_count = len(merged_df)
        merged_df = merged_df[
            merged_df[info_cols].apply(
                lambda x: x.str.strip().replace("", pd.NA).notna().any(), axis=1
            )
        ]
        removed_rows = initial_row_count - len(merged_df)
        logging.info(
            "Removed %d rows with all empty or NaN synopsis columns.", removed_rows
        )

        # Save the updated merged dataset with a progress bar
        logging.info(
            "Saving the merged manga dataset to 'model/merged_manga_dataset.csv'."
        )
        chunk_size: int = 1000
        total_chunks: int = (len(merged_df) // chunk_size) + 1

        with open(
            "model/merged_manga_dataset.csv", "w", newline="", encoding="utf-8"
        ) as f:
            # Write the header
            merged_df.iloc[:0].to_csv(f, index=False)
            logging.info("Writing data in chunks of %d.", chunk_size)
            for i in tqdm(range(total_chunks), desc="Saving to CSV"):
                start: int = i * chunk_size
                end: int = start + chunk_size
                merged_df.iloc[start:end].to_csv(f, header=False, index=False)

        logging.info(
            "Manga datasets merged and saved to 'model/merged_manga_dataset.csv'."
        )
        return merged_df
    except Exception as e:
        logging.error(
            "An error occurred while merging manga datasets: %s", e, exc_info=True
        )
        raise


def main() -> None:
    """
    Main function to parse command-line arguments and merge datasets.

    This function serves as the entry point for the dataset merging process. It parses
    command-line arguments to determine whether to merge anime or manga datasets, then
    calls the appropriate merging function.

    The function expects a single command-line argument --type which must be either
    'anime' or 'manga'. It will:
    1. Parse the command-line arguments
    2. Log the specified dataset type
    3. Call merge_anime_datasets() or merge_manga_datasets() based on the type
    4. Log an error if an invalid type is specified

    Returns:
        None

    Raises:
        No exceptions are raised directly, but underlying merge functions may raise exceptions
    """
    args = parse_args()
    dataset_type: str = args.type
    logging.info("Dataset type specified: '%s'.", dataset_type)

    if dataset_type == "anime":
        merge_anime_datasets()
    elif dataset_type == "manga":
        merge_manga_datasets()
    else:
        logging.error("Invalid type specified. Use 'anime' or 'manga'.")


if __name__ == "__main__":
    main()
