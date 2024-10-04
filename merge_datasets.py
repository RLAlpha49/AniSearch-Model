"""
This script merges multiple anime datasets into a single dataset.

It performs the following operations:
- Loads various anime datasets from CSV files and the Hugging Face datasets library.
- Preprocesses names for matching by converting them to lowercase and stripping whitespace.
- Merges datasets based on common identifiers such as anime_id and uid.
- Adds additional synopsis information from various sources.
- Removes duplicates and saves the merged dataset to a CSV file.
"""

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

# Load datasets
anime_dataset_2023 = pd.read_csv(
    "data/anime-dataset-2023.csv"
)  # Base dataset, 24906 Rows
animes = pd.read_csv("data/animes.csv")  # 16217 Rows
anime_additional = pd.read_csv("data/Anime.csv")  # 18495 Rows
anime_4500 = pd.read_csv("data/anime4500.csv")  # 4500 Rows
anime_20220927_raw = pd.read_csv("data/anime-20220927-raw.csv")  # 16586 Rows
anime_2022 = pd.read_csv("data/Anime-2022.csv")  # 21461 Rows
anime_data = pd.read_csv("data/Anime_data.csv")  # 2453 Rows
anime2 = pd.read_csv("data/anime2.csv")  # 18495 Rows
mal_anime = pd.read_csv("data/mal_anime.csv")  # 24262 Rows

# Load the new datasets using the datasets library
new_dataset = load_dataset("johnidouglas/anime_270", split="train")  # 269 Rows
new_dataset_df = new_dataset.to_pandas()

wykonos_dataset = load_dataset("wykonos/anime", split="train")  # 18495
wykonos_dataset_df = wykonos_dataset.to_pandas()

# Drop specified columns from anime_dataset_2023
columns_to_drop = [
    "Status",
    "Source",
    "Duration",
    "Rating",
    "Rank",
    "Popularity",
    "Favorites",
    "Scored By",
    "Members",
    "Premiered",
    "Producers",
    "Licensors",
    "Studios",
]
anime_dataset_2023.drop(columns=columns_to_drop, inplace=True, errors="ignore")

# Check for duplicates in the keys and remove them, then save the cleaned data back to the files
if anime_dataset_2023["anime_id"].duplicated().any():
    print(
        "Warning: Duplicate anime_id found in anime_dataset_2023. Removing duplicates."
    )
    anime_dataset_2023.drop_duplicates(subset="anime_id", inplace=True)
    anime_dataset_2023.to_csv("data/anime-dataset-2023.csv", index=False)

if animes["uid"].duplicated().any():
    print("Warning: Duplicate uid found in animes. Removing duplicates.")
    animes.drop_duplicates(subset="uid", inplace=True)
    animes.to_csv("data/animes.csv", index=False)

if anime_additional["Rank"].duplicated().any():
    print("Warning: Duplicate Rank found in anime_additional. Removing duplicates.")
    anime_additional.drop_duplicates(subset="Rank", inplace=True)
    anime_additional.to_csv("data/Anime.csv", index=False)

if anime_20220927_raw["id"].duplicated().any():
    print("Warning: Duplicate id found in anime_20220927_raw. Removing duplicates.")
    anime_20220927_raw.drop_duplicates(subset="id", inplace=True)
    anime_20220927_raw.to_csv("data/anime-20220927-raw.csv", index=False)

if anime_2022["ID"].duplicated().any():
    print("Warning: Duplicate ID found in anime_2022. Removing duplicates.")
    anime_2022.drop_duplicates(subset="ID", inplace=True)
    anime_2022.to_csv("data/Anime-2022.csv", index=False)


# Preprocess names for matching
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


anime_dataset_2023["Name"] = anime_dataset_2023["Name"].apply(preprocess_name)
anime_dataset_2023["English name"] = anime_dataset_2023["English name"].apply(
    preprocess_name
)
anime_dataset_2023["Other name"] = anime_dataset_2023["Other name"].apply(
    preprocess_name
)
anime_additional["Name"] = anime_additional["Name"].apply(preprocess_name)
anime_additional["Japanese_name"] = anime_additional["Japanese_name"].apply(
    preprocess_name
)
anime_4500["Title"] = anime_4500["Title"].apply(preprocess_name)
anime_20220927_raw["title_english"] = anime_20220927_raw["title_english"].apply(
    preprocess_name
)
anime_20220927_raw["title_romaji"] = anime_20220927_raw["title_romaji"].apply(
    preprocess_name
)
anime_20220927_raw["title_native"] = anime_20220927_raw["title_native"].apply(
    preprocess_name
)
wykonos_dataset_df["Name"] = wykonos_dataset_df["Name"].apply(preprocess_name)
wykonos_dataset_df["Japanese_name"] = wykonos_dataset_df["Japanese_name"].apply(
    preprocess_name
)
anime_data["Name"] = anime_data["Name"].apply(preprocess_name)
anime2["Name"] = anime2["Name"].apply(preprocess_name)
anime2["Japanese_name"] = anime2["Japanese_name"].apply(preprocess_name)
mal_anime["title"] = mal_anime["title"].apply(preprocess_name)


# Function to add additional synopses
def add_additional_synopsis(
    merged, additional_df, description_col, name_columns, new_synopsis_col
):
    """Adds additional synopsis information to a merged DataFrame from an additional DataFrame.

    Args:
        merged (pd.DataFrame): The main DataFrame to update with additional synopsis.
        additional_df (pd.DataFrame): The DataFrame containing additional synopsis information.
        description_col (str): The name of the column in additional_df containing the synopsis.
        name_columns (list): List of column names in additional_df to use for matching.
        new_synopsis_col (str): The name of the new column to add to merged for additional synopsis.

    Returns:
        pd.DataFrame: The updated merged DataFrame with the new synopsis column.
    """
    # Ensure the new synopsis column exists
    if new_synopsis_col not in merged.columns:
        merged[new_synopsis_col] = None

    # Iterate over each row in the additional dataset with a progress bar
    for _, row in tqdm(
        additional_df.iterrows(),
        total=additional_df.shape[0],
        desc=f"Merging {new_synopsis_col}",
    ):
        # Find matches using vectorized operations
        matches = merged[
            (merged["Name"].isin([row[col] for col in name_columns]))
            | (merged["English name"].isin([row[col] for col in name_columns]))
            | (merged["Other name"].isin([row[col] for col in name_columns]))
        ]

        # If matches are found, update the synopsis
        if not matches.empty:
            for match_idx in matches.index:
                # Check if the synopsis is already present in any of the existing synopsis columns
                existing_synopses = [
                    merged.at[match_idx, col]
                    for col in merged.columns
                    if "Synopsis" in col and pd.notna(merged.at[match_idx, col])
                ]
                if row[description_col] not in existing_synopses:
                    # Add the new synopsis to the new column if it's not a duplicate
                    merged.at[match_idx, new_synopsis_col] = row[description_col]

    return merged


# Function to remove duplicate synopses after merging based on IDs
def remove_duplicate_synopses(merged_dataframe, synopsis_columns):
    """
    Removes duplicate synopses from the merged DataFrame.

    Args:
        merged_dataframe (pd.DataFrame): The DataFrame containing merged data.
        synopsis_columns (list): List of column names containing synopses.

    Returns:
        pd.DataFrame: The DataFrame with duplicate synopses removed.
    """
    for index, row in merged_dataframe.iterrows():
        unique_synopses = set()
        for col in synopsis_columns:
            if pd.notna(row[col]) and row[col] not in unique_synopses:
                unique_synopses.add(row[col])
            else:
                merged_dataframe.at[index, col] = None  # Remove duplicate synopsis
    return merged_dataframe


# Merge datasets on anime_id and uid
final_merged_df = pd.merge(
    anime_dataset_2023,
    animes[["uid", "synopsis"]],
    left_on="anime_id",
    right_on="uid",
    how="outer",
)

final_merged_df.drop(columns=["uid"], inplace=True)
final_merged_df.rename(columns={"synopsis": "Synopsis animes dataset"}, inplace=True)

# Merge the new dataset using MAL_ID
final_merged_df = pd.merge(
    final_merged_df,
    new_dataset_df[["MAL_ID", "sypnopsis"]],
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
    anime_2022[["ID", "Synopsis"]],
    left_on="anime_id",
    right_on="ID",
    how="outer",
    suffixes=("", "_Anime2022"),
)

final_merged_df.rename(
    columns={"Synopsis_Anime2022": "Synopsis Anime-2022 Dataset"}, inplace=True
)
final_merged_df.drop(columns=["ID"], inplace=True)

# Remove duplicate synopses after ID-based merges
synopsis_cols = [
    "Synopsis animes dataset",
    "Synopsis anime_270 Dataset",
    "Synopsis Anime-2022 Dataset",
]
final_merged_df = remove_duplicate_synopses(final_merged_df, synopsis_cols)

# Update the merged dataset with additional synopses from Anime.csv
final_merged_df = add_additional_synopsis(
    final_merged_df,
    anime_additional,
    "Description",
    ["Name", "Japanese_name"],
    "Synopsis Anime Dataset",
)

# Update the merged dataset with additional synopses from anime4500.csv
final_merged_df = add_additional_synopsis(
    final_merged_df, anime_4500, "Description", ["Title"], "Synopsis anime4500 Dataset"
)

# Update the merged dataset with additional synopses from anime-20220927-raw.csv
final_merged_df = add_additional_synopsis(
    final_merged_df,
    anime_20220927_raw,
    "description",
    ["title_english", "title_romaji", "title_native"],
    "Synopsis anime-20220927-raw Dataset",
)

# Update the merged dataset with additional synopses from wykonos/anime
final_merged_df = add_additional_synopsis(
    final_merged_df,
    wykonos_dataset_df,
    "Description",
    ["Name", "Japanese_name"],
    "Synopsis wykonos Dataset",
)

# Remove entries with "No synopsis yet - check back soon!" from the Description column in anime_data
anime_data = anime_data[
    anime_data["Description"] != "No synopsis yet - check back soon!"
]

# Update the merged dataset with additional synopses from Anime_data.csv
final_merged_df = add_additional_synopsis(
    final_merged_df,
    anime_data,
    "Description",
    ["Name"],
    "Synopsis Anime_data Dataset",
)

# Update the merged dataset with additional synopses from anime2.csv
final_merged_df = add_additional_synopsis(
    final_merged_df,
    anime2,
    "Description",
    ["Name", "Japanese_name"],
    "Synopsis anime2 Dataset",
)

# Update the merged dataset with additional synopses from mal_anime.csv
final_merged_df = add_additional_synopsis(
    final_merged_df,
    mal_anime,
    "synopsis",
    ["title"],
    "Synopsis mal_anime Dataset",
)


# Remove duplicates based on a subset of columns that should be unique
final_merged_df.drop_duplicates(subset=["anime_id"], inplace=True)

# Save the updated merged dataset with a progress bar
CHUNK_SIZE = 1000
total_chunks = (len(final_merged_df) // CHUNK_SIZE) + 1

with open("model/merged_anime_dataset.csv", "w", newline="", encoding="utf-8") as f:
    # Write the header
    final_merged_df.iloc[:0].to_csv(f, index=False)

    # Write the data in chunks
    for i in tqdm(range(total_chunks), desc="Saving to CSV"):
        start = i * CHUNK_SIZE
        end = start + CHUNK_SIZE
        final_merged_df.iloc[start:end].to_csv(f, header=False, index=False)

print("Datasets merged and saved to model/merged_anime_dataset.csv")
