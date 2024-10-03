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

# Load the new datasets using the datasets library
new_dataset = load_dataset("johnidouglas/anime_270", split="train")  # 269 Rows
new_dataset_df = new_dataset.to_pandas()

wykonos_dataset = load_dataset("wykonos/anime", split="train")
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


# Function to add additional synopses
def add_additional_synopsis(
    merged_df, additional_df, description_col, name_columns, new_synopsis_col
):
    # Ensure the new synopsis column exists
    if new_synopsis_col not in merged_df.columns:
        merged_df[new_synopsis_col] = None

    # Iterate over each row in the additional dataset with a progress bar
    for idx, row in tqdm(
        additional_df.iterrows(),
        total=additional_df.shape[0],
        desc=f"Merging {new_synopsis_col}",
    ):
        # Find matches using vectorized operations
        matches = merged_df[
            (merged_df["Name"].isin([row[col] for col in name_columns]))
            | (merged_df["English name"].isin([row[col] for col in name_columns]))
            | (merged_df["Other name"].isin([row[col] for col in name_columns]))
        ]

        # If matches are found, update the synopsis
        if not matches.empty:
            for match_idx in matches.index:
                # Add the new synopsis to the new column
                merged_df.at[match_idx, new_synopsis_col] = row[description_col]

    return merged_df


# Merge datasets on anime_id and uid
merged_df = pd.merge(
    anime_dataset_2023,
    animes[["uid", "synopsis"]],
    left_on="anime_id",
    right_on="uid",
    how="left",
)

merged_df.drop(columns=["uid"], inplace=True)
merged_df.rename(columns={"synopsis": "Synopsis animes dataset"}, inplace=True)

# Merge the new dataset using MAL_ID
merged_df = pd.merge(
    merged_df,
    new_dataset_df[["MAL_ID", "sypnopsis"]],
    left_on="anime_id",
    right_on="MAL_ID",
    how="left",
)
merged_df.rename(columns={"sypnopsis": "Synopsis anime_270 Dataset"}, inplace=True)
merged_df.drop(columns=["MAL_ID"], inplace=True)

# Merge the Anime-2022 dataset using ID
merged_df = pd.merge(
    merged_df,
    anime_2022[["ID", "Synopsis"]],
    left_on="anime_id",
    right_on="ID",
    how="left",
    suffixes=("", "_Anime2022"),
)

merged_df.rename(
    columns={"Synopsis_Anime2022": "Synopsis Anime-2022 Dataset"}, inplace=True
)
merged_df.drop(columns=["ID"], inplace=True)

# Update the merged dataset with additional synopses from Anime.csv
merged_df = add_additional_synopsis(
    merged_df,
    anime_additional,
    "Description",
    ["Name", "Japanese_name"],
    "Synopsis Anime Dataset",
)

# Update the merged dataset with additional synopses from anime4500.csv
merged_df = add_additional_synopsis(
    merged_df, anime_4500, "Description", ["Title"], "Synopsis anime4500 Dataset"
)

# Update the merged dataset with additional synopses from anime-20220927-raw.csv
merged_df = add_additional_synopsis(
    merged_df,
    anime_20220927_raw,
    "description",
    ["title_english", "title_romaji", "title_native"],
    "Synopsis anime-20220927-raw Dataset",
)

# Update the merged dataset with additional synopses from wykonos/anime
merged_df = add_additional_synopsis(
    merged_df,
    wykonos_dataset_df,
    "Description",
    ["Name", "Japanese_name"],
    "Synopsis wykonos Dataset",
)

# Remove duplicates based on a subset of columns that should be unique
merged_df.drop_duplicates(subset=["anime_id"], inplace=True)

# Save the updated merged dataset with a progress bar
CHUNK_SIZE = 1000
total_chunks = (len(merged_df) // CHUNK_SIZE) + 1

with open("model/merged_anime_dataset.csv", "w", newline="", encoding="utf-8") as f:
    # Write the header
    merged_df.iloc[:0].to_csv(f, index=False)

    # Write the data in chunks
    for i in tqdm(range(total_chunks), desc="Saving to CSV"):
        start = i * CHUNK_SIZE
        end = start + CHUNK_SIZE
        merged_df.iloc[start:end].to_csv(f, header=False, index=False)

print("Datasets merged and saved to model/merged_anime_dataset.csv")
