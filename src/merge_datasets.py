"""
This script merges multiple anime or manga datasets into a single dataset.

It performs the following operations based on the specified type (anime or manga):
- Loads various datasets from CSV files and the Hugging Face datasets library.
- Preprocesses names for matching by converting them to lowercase and stripping whitespace.
- Merges datasets based on common identifiers.
- Adds additional synopsis or description information from various sources.
- Removes duplicates and saves the merged dataset to a CSV file.
"""

import os
import argparse
import logging
from logging.handlers import RotatingFileHandler
import sys
from typing import Any, Optional
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

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
    Parses command-line arguments for the dataset merging script.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
        Specifically, it includes the 'type' argument which indicates whether to
        generate an 'anime' or 'manga' dataset.
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
def clean_synopsis(df: pd.DataFrame, synopsis_col: str, unwanted_phrase: str) -> None:
    """
    Sets the synopsis to empty string if it contains an unwanted phrase.

    Args:
        df (pd.DataFrame): The DataFrame to clean.
        synopsis_col (str): The name of the synopsis column.
        unwanted_phrase (str): The phrase indicating an invalid synopsis.
    """
    logging.info("Cleaning synopses in column: %s", synopsis_col)
    for index, row in df.iterrows():
        if pd.notna(row[synopsis_col]) and unwanted_phrase in row[synopsis_col]:
            df.at[index, synopsis_col] = ""


# Function to consolidate titles into a single 'title' column
def consolidate_titles(df: pd.DataFrame, title_columns: list) -> pd.Series:
    """
    Consolidates multiple title columns into a single 'title' column.

    Args:
        df (pd.DataFrame): The merged DataFrame.
        title_columns (list): List of title column names to consolidate.

    Returns:
        pd.Series: A consolidated 'title' series.
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
    Preprocesses a given name by converting it to a lowercase string and removing
    leading/trailing whitespace.

    Args:
        name (Any): The input name to be preprocessed. Can be of any type that can
        be converted to a string.

    Returns:
        str: The preprocessed name as a lowercase string with leading and trailing
        whitespace removed.
    """
    if pd.isna(name):
        return ""
    return str(name).strip().lower()


def find_additional_info(
    row: pd.Series,
    additional_df: pd.DataFrame,
    description_col: str,
    name_columns: list,
) -> Optional[str]:
    """
    Finds additional information for a given row based on matching names.

    Args:
        row (pd.Series): A row from the merged DataFrame.
        additional_df (pd.DataFrame): The additional DataFrame to search.
        description_col (str): The description column in the additional DataFrame.
        name_columns (list): The name columns to match against.

    Returns:
        str or None: The found description or None if not found.
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
    Adds additional synopsis information to a merged DataFrame from an additional DataFrame.

    Args:
        merged (pd.DataFrame): The merged DataFrame to update.
        additional_df (pd.DataFrame): The additional DataFrame containing new information.
        description_col (str): The description column in the additional DataFrame.
        name_columns (list): The name columns to match against in the additional DataFrame.
        new_synopsis_col (str): The name of the new synopsis column to add.

    Returns:
        pd.DataFrame: The updated merged DataFrame with additional information.
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
    Removes duplicate information across specified synopsis columns,
    keeping the first non-null value.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        info_cols (list): List of synopsis columns to check for duplicates.

    Returns:
        pd.DataFrame: The DataFrame with duplicates removed.
    """
    logging.info("Removing duplicate synopses across columns: %s", info_cols)
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
    Merges multiple anime datasets into a single DataFrame.

    Returns:
        pd.DataFrame: The merged anime DataFrame.
    """
    logging.info("Starting to merge anime datasets.")
    try:
        # Load datasets
        logging.info("Loading anime datasets from CSV files.")
        myanimelist_dataset: pd.DataFrame = pd.read_csv(
            "data/anime/Anime.csv"
        )  # Base dataset
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
        anime_270 = load_dataset("johnidouglas/anime_270", split="train")  # 269 Rows
        anime_270_df: pd.DataFrame = anime_270.to_pandas()  # type: ignore

        logging.info("Loading 'wykonos/anime' dataset from Hugging Face datasets.")
        wykonos_dataset = load_dataset("wykonos/anime", split="train")  # 18495 Rows
        wykonos_dataset_df: pd.DataFrame = wykonos_dataset.to_pandas()  # type: ignore

        # Drop specified columns from myanimelist_dataset
        columns_to_drop: list[str] = [
            "scored_by",
            "status",
            "source",
            "members",
            "favorites",
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
            df: pd.DataFrame = locals()[df_name]
            for col in cols:
                if col in df.columns:
                    logging.info("Preprocessing column '%s' in '%s'.", col, df_name)
                    df[col] = df[col].apply(preprocess_name)

        # Clean synopses in specific datasets
        logging.info("Cleaning synopses in specific datasets.")
        clean_synopsis(
            anime_dataset_2023, "Synopsis", "No description available for this anime."
        )
        clean_synopsis(anime_2022, "Synopsis", "Unknown")
        clean_synopsis(wykonos_dataset_df, "Description", "No synopsis yet")
        clean_synopsis(anime_data, "Description", "No synopsis yet")
        clean_synopsis(anime2, "Description", "No synopsis yet")
        clean_synopsis(
            mal_anime,
            "synopsis",
            "No synopsis information has been added to this title.",
        )

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
            anime_270_df[["MAL_ID", "sypnopsis", "Name"]].rename(  # type: ignore
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

        # Remove duplicate synopses
        synopsis_cols: list[str] = [
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
        logging.info("Removing duplicate synopses across columns: %s", synopsis_cols)
        final_merged_df = remove_duplicate_infos(final_merged_df, synopsis_cols)

        # Remove duplicates based on 'anime_id'
        logging.info("Removing duplicates based on 'anime_id'.")
        final_merged_df.drop_duplicates(subset=["anime_id"], inplace=True)

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
            logging.info("Writing data in chunks of %d.", chunk_size)
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
    Merges multiple manga datasets into a single DataFrame.

    Returns:
        pd.DataFrame: The merged manga DataFrame.
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
            df: pd.DataFrame = locals()[df_name]
            for col in cols:
                if col in df.columns:
                    logging.info("Preprocessing column '%s' in '%s'.", col, df_name)
                    df[col] = df[col].apply(preprocess_name)

        # Clean synopses in specific datasets
        logging.info("Cleaning synopses in specific datasets.")
        clean_synopsis(
            data, "description", "This entry currently doesn't have a synopsis."
        )
        clean_synopsis(jikan, "synopsis", "Looking for information on the")

        # Merge main dataset with jikan on 'manga_id' and 'mal_id'
        logging.info(
            "Merging 'manga_main' with 'jikan' dataset on 'manga_id' and 'mal_id'."
        )
        merged_df: pd.DataFrame = pd.merge(
            manga_main,
            jikan[["mal_id", "synopsis", "title"]].rename(
                columns={"title": "title_jikan"}
            ),
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
            data,
            "description",
            ["title"],
            "Synopsis data Dataset",
        )

        # Remove duplicate synopses and descriptions
        info_cols: list[str] = [
            "Synopsis jikan Dataset",
            "Synopsis data Dataset",
        ]
        logging.info("Removing duplicate synopses and descriptions.")
        merged_df = remove_duplicate_infos(merged_df, info_cols)

        # Remove duplicates based on 'manga_id'
        logging.info("Removing duplicates based on 'manga_id'.")
        merged_df.drop_duplicates(subset=["manga_id"], inplace=True)

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

    This function determines the type of dataset to merge based on the
    command-line argument 'type'. It supports merging 'anime' or 'manga'
    datasets. If an invalid type is specified, it logs an error message.
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
