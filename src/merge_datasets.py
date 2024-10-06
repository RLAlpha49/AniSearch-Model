"""
This script merges multiple anime or manga datasets into a single dataset.

It performs the following operations based on the specified type (anime or manga):
- Loads various datasets from CSV files and the Hugging Face datasets library.
- Preprocesses names for matching by converting them to lowercase and stripping whitespace.
- Merges datasets based on common identifiers.
- Adds additional synopsis or description information from various sources.
- Removes duplicates and saves the merged dataset to a CSV file.
"""

import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset


# Parse command-line arguments
def parse_args():
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
def clean_synopsis(df, synopsis_col, unwanted_phrase):
    """
    Sets the synopsis to empty string if it contains an unwanted phrase.

    Args:
        df (pd.DataFrame): The DataFrame to clean.
        synopsis_col (str): The name of the synopsis column.
        unwanted_phrase (str): The phrase indicating an invalid synopsis.
    """
    for index, row in df.iterrows():
        if pd.notna(row[synopsis_col]) and unwanted_phrase in row[synopsis_col]:
            df.at[index, synopsis_col] = ""


# Function to consolidate titles into a single 'title' column
def consolidate_titles(df, title_columns):
    """
    Consolidates multiple title columns into a single 'title' column.

    Args:
        df (pd.DataFrame): The merged DataFrame.
        title_columns (list): List of title column names to consolidate.

    Returns:
        pd.Series: A consolidated 'title' series.
    """
    # Initialize 'title' with the main title column if it exists
    if "title" in df.columns:
        consolidated_title = df["title"]
    else:
        consolidated_title = pd.Series([""] * len(df), index=df.index)

    # Iterate over other title columns to fill missing titles only if 'title' is not already set
    for col in title_columns:
        if col in df.columns:
            consolidated_title = consolidated_title.where(
                consolidated_title.notna(), df[col]
            )

    # Replace empty strings and placeholders with NaN
    consolidated_title.replace(["", "unknown title"], pd.NA, inplace=True)
    return consolidated_title


# Preprocess names by converting to lowercase and stripping whitespace
def preprocess_name(name):
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
    return str(name).strip().lower()


def find_additional_info(row, additional_df, description_col, name_columns):
    """Finds additional info for a given row."""
    for merged_name_col in ["title", "title_english", "title_japanese"]:
        for additional_name_col in name_columns:
            if row[merged_name_col] in additional_df[additional_name_col].values:
                info = additional_df.loc[
                    additional_df[additional_name_col] == row[merged_name_col],
                    description_col,
                ]
                if isinstance(info, pd.Series):
                    return info.dropna().iloc[0] if not info.dropna().empty else None
    return None


# Function to add additional synopses or descriptions
def add_additional_info(
    merged, additional_df, description_col, name_columns, new_synopsis_col
):
    """Adds additional synopsis information to a merged DataFrame from an additional DataFrame.

    Args:
        merged (pd.DataFrame): The main DataFrame to update with additional info.
        additional_df (pd.DataFrame): The DataFrame containing additional info.
        description_col (str): The name of the column in additional_df containing the synopsis.
        name_columns (list): List of column names in additional_df to use for matching.
        new_synopsis_col (str): The name of the new column to add to merged for additional info.

    Returns:
        pd.DataFrame: The updated merged DataFrame with the new synopsis or description column.
    """
    if new_synopsis_col not in merged.columns:
        merged[new_synopsis_col] = None

    def update_row_with_info(idx, row):
        """Updates a row with additional info if not a duplicate."""
        info = find_additional_info(row, additional_df, description_col, name_columns)
        if pd.notna(info):
            existing_infos = [
                merged.at[idx, col]
                for col in merged.columns
                if new_synopsis_col in col and pd.notna(merged.at[idx, col])
            ]
            if info not in existing_infos:
                merged.at[idx, new_synopsis_col] = info

    # Iterate over each row in the merged DataFrame with a progress bar
    for idx, row in tqdm(
        merged.iterrows(),
        total=merged.shape[0],
        desc=(
            f"Adding additional info from "
            f"{new_synopsis_col.replace('Synopsis ', '').replace('Description ', '')}"
        ),
    ):
        update_row_with_info(idx, row)

    return merged


# Function to remove duplicate synopses or descriptions
def remove_duplicate_infos(merged_dataframe, info_columns):
    """
    Removes duplicate synopses or descriptions from the merged DataFrame.

    Args:
        merged_dataframe (pd.DataFrame): The DataFrame containing merged data.
        info_columns (list): List of column names containing synopses or descriptions.

    Returns:
        pd.DataFrame: The DataFrame with duplicate synopses or descriptions removed.
    """
    for index, row in merged_dataframe.iterrows():
        unique_infos = set()
        for col in info_columns:
            if pd.notna(row[col]) and row[col] not in unique_infos:
                unique_infos.add(row[col])
            else:
                merged_dataframe.at[index, col] = None  # Remove duplicate info
    return merged_dataframe


# Function to merge anime datasets
def merge_anime_datasets():
    """
    Merges multiple anime datasets into a single DataFrame.

    Returns:
        pd.DataFrame: The merged anime DataFrame.
    """
    # Load datasets
    myanimelist_dataset = pd.read_csv("data/anime/Anime.csv")  # Base dataset
    anime_dataset_2023 = pd.read_csv("data/anime/anime-dataset-2023.csv")
    animes = pd.read_csv("data/anime/animes.csv")
    anime_4500 = pd.read_csv("data/anime/anime4500.csv")
    anime_2022 = pd.read_csv("data/anime/Anime-2022.csv")
    anime_data = pd.read_csv("data/anime/Anime_data.csv")
    anime2 = pd.read_csv("data/anime/anime2.csv")
    mal_anime = pd.read_csv("data/anime/mal_anime.csv")

    # Load using the datasets library
    anime_270 = load_dataset("johnidouglas/anime_270", split="train")  # 269 Rows
    anime_270_df = anime_270.to_pandas()

    wykonos_dataset = load_dataset("wykonos/anime", split="train")  # 18495 Rows
    wykonos_dataset_df = wykonos_dataset.to_pandas()

    # Drop specified columns from myanimelist_dataset
    columns_to_drop = [
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
    myanimelist_dataset.drop(columns=columns_to_drop, inplace=True, errors="ignore")

    # Check for duplicates in the keys and remove them, then save the cleaned data back to the files
    if myanimelist_dataset["anime_id"].duplicated().any():
        print(
            "Warning: Duplicate anime_id found in myanimelist_dataset. Removing duplicates."
        )
        myanimelist_dataset.drop_duplicates(subset="anime_id", inplace=True)
        myanimelist_dataset.to_csv("data/anime/Anime.csv", index=False)

    if anime_dataset_2023["anime_id"].duplicated().any():
        print(
            "Warning: Duplicate anime_id found in anime_dataset_2023. Removing duplicates."
        )
        anime_dataset_2023.drop_duplicates(subset="anime_id", inplace=True)
        anime_dataset_2023.to_csv("data/anime/anime-dataset-2023.csv", index=False)

    if animes["uid"].duplicated().any():
        print("Warning: Duplicate uid found in animes. Removing duplicates.")
        animes.drop_duplicates(subset="uid", inplace=True)
        animes.to_csv("data/anime/animes.csv", index=False)

    if anime_2022["ID"].duplicated().any():
        print("Warning: Duplicate ID found in anime_2022. Removing duplicates.")
        anime_2022.drop_duplicates(subset="ID", inplace=True)
        anime_2022.to_csv("data/anime/Anime-2022.csv", index=False)

    # Preprocess names for matching
    myanimelist_dataset["title"] = myanimelist_dataset["title"].apply(preprocess_name)
    myanimelist_dataset["title_english"] = myanimelist_dataset["title_english"].apply(
        preprocess_name
    )
    myanimelist_dataset["title_japanese"] = myanimelist_dataset["title_japanese"].apply(
        preprocess_name
    )
    anime_dataset_2023["Name"] = anime_dataset_2023["Name"].apply(preprocess_name)
    anime_dataset_2023["English name"] = anime_dataset_2023["English name"].apply(
        preprocess_name
    )
    anime_dataset_2023["Other name"] = anime_dataset_2023["Other name"].apply(
        preprocess_name
    )
    anime_4500["Title"] = anime_4500["Title"].apply(preprocess_name)
    wykonos_dataset_df["Name"] = wykonos_dataset_df["Name"].apply(preprocess_name)
    wykonos_dataset_df["Japanese_name"] = wykonos_dataset_df["Japanese_name"].apply(
        preprocess_name
    )
    anime_data["Name"] = anime_data["Name"].apply(preprocess_name)
    anime2["Name"] = anime2["Name"].apply(preprocess_name)
    mal_anime["title"] = mal_anime["title"].apply(preprocess_name)

    # Clean synopses in specific datasets
    clean_synopsis(
        anime_dataset_2023, "Synopsis", "No description available for this anime."
    )
    clean_synopsis(anime_2022, "Synopsis", "Unknown")
    clean_synopsis(wykonos_dataset_df, "Description", "No synopsis yet")
    clean_synopsis(anime_data, "Description", "No synopsis yet")
    clean_synopsis(anime2, "Description", "No synopsis yet")
    clean_synopsis(
        mal_anime, "synopsis", "No synopsis information has been added to this title."
    )

    # Merge datasets on anime_id
    final_merged_df = pd.merge(
        myanimelist_dataset,
        anime_dataset_2023[["anime_id", "Synopsis", "Name"]].rename(
            columns={"Name": "title_anime_dataset_2023"}
        ),
        on="anime_id",
        how="outer",
    )

    # Ensure the column is renamed correctly
    final_merged_df.rename(
        columns={"Synopsis": "Synopsis anime_dataset_2023"}, inplace=True
    )

    # Merge datasets on anime_id and uid
    final_merged_df = pd.merge(
        final_merged_df,
        animes[["uid", "synopsis", "title"]].rename(columns={"title": "title_animes"}),
        left_on="anime_id",
        right_on="uid",
        how="outer",
        suffixes=("", "_animes"),
    )

    final_merged_df.drop(columns=["uid"], inplace=True)
    final_merged_df.rename(
        columns={"synopsis_animes": "Synopsis animes dataset"}, inplace=True
    )

    # Merge the new dataset using MAL_ID
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
    final_merged_df.drop(columns=["MAL_ID"], inplace=True)

    # Merge the Anime-2022 dataset using ID
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
    final_merged_df.drop(columns=["ID"], inplace=True)

    # Consolidate all title columns into a single 'title' column
    title_columns = [
        "title_anime_dataset_2023",
        "title_animes",
        "title_anime_270",
        "title_anime_2022",
    ]
    final_merged_df["title"] = consolidate_titles(final_merged_df, title_columns)

    # Drop redundant title columns
    final_merged_df.drop(columns=title_columns, inplace=True, errors="ignore")

    # Remove duplicate synopses after ID-based merges
    synopsis_cols = [
        "Synopsis anime_dataset_2023",
        "Synopsis animes dataset",
        "Synopsis anime_270 Dataset",
        "Synopsis Anime-2022 Dataset",
    ]
    final_merged_df = remove_duplicate_infos(final_merged_df, synopsis_cols)

    # Update the merged dataset with additional synopses from anime4500.csv
    final_merged_df = add_additional_info(
        final_merged_df,
        anime_4500,
        "Description",
        ["Title"],
        "Synopsis anime4500 Dataset",
    )

    # Update the merged dataset with additional synopses from wykonos/anime
    final_merged_df = add_additional_info(
        final_merged_df,
        wykonos_dataset_df,
        "Description",
        ["Name", "Japanese_name"],
        "Synopsis wykonos Dataset",
    )

    # Update the merged dataset with additional synopses from Anime_data.csv
    final_merged_df = add_additional_info(
        final_merged_df,
        anime_data,
        "Description",
        ["Name"],
        "Synopsis Anime_data Dataset",
    )

    # Update the merged dataset with additional synopses from anime2.csv
    final_merged_df = add_additional_info(
        final_merged_df,
        anime2,
        "Description",
        ["Name", "Japanese_name"],
        "Synopsis anime2 Dataset",
    )

    # Update the merged dataset with additional synopses from mal_anime.csv
    final_merged_df = add_additional_info(
        final_merged_df,
        mal_anime,
        "synopsis",
        ["title"],
        "Synopsis mal_anime Dataset",
    )

    # Remove duplicates based on a subset of columns that should be unique
    final_merged_df.drop_duplicates(subset=["anime_id"], inplace=True)

    # Save the updated merged dataset with a progress bar
    chunk_size = 1000
    total_chunks = (len(final_merged_df) // chunk_size) + 1

    with open("model/merged_anime_dataset.csv", "w", newline="", encoding="utf-8") as f:
        # Write the header
        final_merged_df.iloc[:0].to_csv(f, index=False)

        # Write the data in chunks
        for i in tqdm(range(total_chunks), desc="Saving to CSV"):
            start = i * chunk_size
            end = start + chunk_size
            final_merged_df.iloc[start:end].to_csv(f, header=False, index=False)

    print("Anime datasets merged and saved to model/merged_anime_dataset.csv")


# Function to merge manga datasets
def merge_manga_datasets():
    """
    Merges multiple manga datasets into a single DataFrame.

    Returns:
        pd.DataFrame: The merged manga DataFrame.
    """
    # Load datasets
    manga_main = pd.read_csv("data/manga/manga.csv")  # Main dataset
    jikan = pd.read_csv("data/manga/jikan.csv")  # To be merged via mal_id and manga_id
    data = pd.read_csv("data/manga/data.csv")  # To be merged via title

    columns_to_drop = [
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
    manga_main.drop(columns=columns_to_drop, inplace=True, errors="ignore")

    # Check for duplicates in the keys and remove them, then save the cleaned data back to the files
    if manga_main["manga_id"].duplicated().any():
        print("Warning: Duplicate manga_id found in manga_main. Removing duplicates.")
        manga_main.drop_duplicates(subset="manga_id", inplace=True)
        manga_main.to_csv("data/manga/manga.csv", index=False)

    if jikan["mal_id"].duplicated().any():
        print("Warning: Duplicate mal_id found in jikan. Removing duplicates.")
        jikan.drop_duplicates(subset="mal_id", inplace=True)
        jikan.to_csv("data/manga/jikan.csv", index=False)

    if data["title"].duplicated().any():
        print("Warning: Duplicate title found in data. Removing duplicates.")
        data.drop_duplicates(subset="title", inplace=True)
        data.to_csv("data/manga/data.csv", index=False)

    # Preprocess names for matching
    manga_main["title"] = manga_main["title"].apply(preprocess_name)
    manga_main["title_english"] = manga_main["title_english"].apply(preprocess_name)
    manga_main["title_japanese"] = manga_main["title_japanese"].apply(preprocess_name)
    jikan["title"] = jikan["title"].apply(preprocess_name)
    data["title"] = data["title"].apply(preprocess_name)

    clean_synopsis(data, "description", "This entry currently doesn't have a synopsis.")

    # Merge main dataset with jikan on manga_id and mal_id
    merged_df = pd.merge(
        manga_main,
        jikan[["mal_id", "synopsis", "title"]].rename(columns={"title": "title_jikan"}),
        left_on="manga_id",
        right_on="mal_id",
        how="outer",
        suffixes=("", "_jikan"),
    )
    merged_df.rename(columns={"synopsis_jikan": "Synopsis jikan Dataset"}, inplace=True)
    merged_df.drop(columns=["mal_id", "title_jikan"], inplace=True)

    # Merge with data on title
    merged_df = add_additional_info(
        merged_df,
        data,
        "description",
        ["title"],
        "Synopsis data Dataset",
    )

    # Remove duplicate synopses and descriptions
    info_cols = ["Synopsis jikan Dataset", "Synopsis data Dataset"]
    merged_df = remove_duplicate_infos(merged_df, info_cols)

    # Remove duplicates based on manga_id
    merged_df.drop_duplicates(subset=["manga_id"], inplace=True)

    # Save the updated merged dataset with a progress bar
    chunk_size = 1000
    total_chunks = (len(merged_df) // chunk_size) + 1

    with open("model/merged_manga_dataset.csv", "w", newline="", encoding="utf-8") as f:
        # Write the header
        merged_df.iloc[:0].to_csv(f, index=False)

        # Write the data in chunks
        for i in tqdm(range(total_chunks), desc="Saving to CSV"):
            start = i * chunk_size
            end = start + chunk_size
            merged_df.iloc[start:end].to_csv(f, header=False, index=False)

    print("Manga datasets merged and saved to model/merged_manga_dataset.csv")


def main():
    """
    Main function to parse command-line arguments and merge datasets.

    This function determines the type of dataset to merge based on the
    command-line argument 'type'. It supports merging 'anime' or 'manga'
    datasets. If an invalid type is specified, it prints an error message.
    """
    args = parse_args()
    dataset_type = args.type

    if dataset_type == "anime":
        merge_anime_datasets()
    elif dataset_type == "manga":
        merge_manga_datasets()
    else:
        print("Invalid type specified. Use 'anime' or 'manga'.")


if __name__ == "__main__":
    main()
