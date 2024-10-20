import ast
import os
from functools import partial
from multiprocessing import Pool, cpu_count
import pandas as pd
import random
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# List of genres and themes to help build negative pairs
all_genres = {
    "Action",
    "Adventure",
    "Ecchi",
    "Girls Love",
    "Mystery",
    "Hentai",
    "Drama",
    "Romance",
    "Horror",
    "Gourmet",
    "Award Winning",
    "Erotica",
    "Sci-Fi",
    "Fantasy",
    "Sports",
    "Supernatural",
    "Avant Garde",
    "Boys Love",
    "Suspense",
    "Slice of Life",
    "Comedy",
}

all_themes = {
    "Harem",
    "Educational",
    "High Stakes Game",
    "Adult Cast",
    "Anthropomorphic",
    "Iyashikei",
    "Samurai",
    "Pets",
    "Mythology",
    "Idols (Male)",
    "Gore",
    "Visual Arts",
    "Magical Sex Shift",
    "Romantic Subtext",
    "Time Travel",
    "Racing",
    "CGDCT",
    "Detective",
    "Mecha",
    "Psychological",
    "Mahou Shoujo",
    "Childcare",
    "Performing Arts",
    "Combat Sports",
    "Medical",
    "Space",
    "Otaku Culture",
    "Survival",
    "Idols (Female)",
    "Super Power",
    "Reverse Harem",
    "Parody",
    "Love Polygon",
    "School",
    "Strategy Game",
    "Military",
    "Video Game",
    "Historical",
    "Reincarnation",
    "Team Sports",
    "Martial Arts",
    "Crossdressing",
    "Isekai",
    "Workplace",
    "Vampire",
    "Delinquents",
    "Organized Crime",
    "Showbiz",
    "Gag Humor",
    "Music",
}


# Function to calculate partial similarity based on genres and themes
def calculate_similarity(
    genres_a, genres_b, themes_a, themes_b, genre_weight=0.35, theme_weight=0.65
):
    genre_similarity = len(genres_a.intersection(genres_b)) / max(
        len(genres_a.union(genres_b)), 1
    )
    theme_similarity = len(themes_a.intersection(themes_b)) / max(
        len(themes_a.union(themes_b)), 1
    )

    # Weighted average of genre and theme similarities
    similarity = (genre_weight * genre_similarity) + (theme_weight * theme_similarity)
    return similarity


# Function to process a single row for partial positive pairs
def generate_partial_positive_pairs(
    i, df, synopses_columns, partial_threshold, max_partial_per_row
):
    row_a = df.iloc[i]
    row_a_partial_count = 0
    pairs = []
    partial_count = 0
    for j in range(len(df)):
        if i != j:
            row_b = df.iloc[j]
            try:
                genres_a = (
                    set(ast.literal_eval(row_a["genres"]))
                    if pd.notnull(row_a["genres"])
                    else set()
                )
                genres_b = (
                    set(ast.literal_eval(row_b["genres"]))
                    if pd.notnull(row_b["genres"])
                    else set()
                )

                themes_a = (
                    set(ast.literal_eval(row_a["themes"]))
                    if pd.notnull(row_a["themes"])
                    else set()
                )
                themes_b = (
                    set(ast.literal_eval(row_b["themes"]))
                    if pd.notnull(row_b["themes"])
                    else set()
                )

                # Calculate partial similarity based on genres and themes
                similarity = calculate_similarity(
                    genres_a, genres_b, themes_a, themes_b
                )

                if similarity >= partial_threshold + 0.01:
                    # If similarity is above a certain threshold, treat as a partial positive pair
                    valid_synopses_a = [
                        col for col in synopses_columns if pd.notnull(row_a[col])
                    ]
                    valid_synopses_b = [
                        col for col in synopses_columns if pd.notnull(row_b[col])
                    ]

                    # Only create a pair if both entries have at least one valid synopsis
                    if valid_synopses_a and valid_synopses_b:
                        col_a = random.choice(valid_synopses_a)
                        col_b = random.choice(valid_synopses_b)

                        pairs.append(
                            InputExample(
                                texts=[row_a[col_a], row_b[col_b]], label=similarity
                            )
                        )  # Partial positive pair
                        partial_count += 1
                        row_a_partial_count += 1

                        if partial_count >= max_partial_per_row:
                            break
                        if row_a_partial_count >= max_partial_per_row:
                            break
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(e)
                continue

    return pairs


# Function to process a single row for negative pairs
def generate_negative_pairs(
    i, df, synopses_columns, partial_threshold, max_negative_per_row
):
    row_a = df.iloc[i]
    row_a_negative_count = 0
    pairs = []
    negative_count = 0
    for j in range(len(df)):
        if i != j:
            row_b = df.iloc[j]
            try:
                genres_a = (
                    set(ast.literal_eval(row_a["genres"]))
                    if pd.notnull(row_a["genres"])
                    else set()
                )
                genres_b = (
                    set(ast.literal_eval(row_b["genres"]))
                    if pd.notnull(row_b["genres"])
                    else set()
                )

                themes_a = (
                    set(ast.literal_eval(row_a["themes"]))
                    if pd.notnull(row_a["themes"])
                    else set()
                )
                themes_b = (
                    set(ast.literal_eval(row_b["themes"]))
                    if pd.notnull(row_b["themes"])
                    else set()
                )

                # Calculate partial similarity based on genres and themes
                similarity = calculate_similarity(
                    genres_a, genres_b, themes_a, themes_b
                )

                if similarity <= partial_threshold - 0.01:
                    # If similarity is below a certain threshold, treat as a negative pair
                    valid_synopses_a = [
                        col for col in synopses_columns if pd.notnull(row_a[col])
                    ]
                    valid_synopses_b = [
                        col for col in synopses_columns if pd.notnull(row_b[col])
                    ]

                    # Only create a pair if both entries have at least one valid synopsis
                    if valid_synopses_a and valid_synopses_b:
                        col_a = random.choice(valid_synopses_a)
                        col_b = random.choice(valid_synopses_b)

                        pairs.append(
                            InputExample(
                                texts=[row_a[col_a], row_b[col_b]], label=similarity
                            )
                        )  # Partial or negative pair
                        negative_count += 1
                        row_a_negative_count += 1

                        if negative_count >= max_negative_per_row:
                            break
                        if row_a_negative_count >= max_negative_per_row:
                            break
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(e)
                continue

    return pairs


# Function to create positive and negative pairs
def create_pairs(
    df,
    max_negative_pairs=25000,
    max_partial_positive_pairs=25000,
    partial_threshold=0.5,
):
    pairs = []
    synopses_columns = [col for col in df.columns if "synopsis" in col.lower()]

    # Positive pairs (different synopses for the same entry, score = 1.0)
    for _, row in tqdm(df.iterrows(), desc="Creating positive pairs", total=len(df)):
        valid_synopses = [row[col] for col in synopses_columns if pd.notnull(row[col])]
        unique_synopses = list(set(valid_synopses))  # Remove duplicates
        if len(unique_synopses) > 1:
            for i, synopsis_i in enumerate(unique_synopses):
                for _, synopsis_j in enumerate(unique_synopses[i + 1 :], start=i + 1):
                    pairs.append(
                        InputExample(texts=[synopsis_i, synopsis_j], label=1.0)
                    )  # Positive pair

    max_partial_per_row = int(max_partial_positive_pairs / len(df))
    max_negative_per_row = int(max_negative_pairs / len(df))
    print(f"Max partial per row: {max_partial_per_row}")
    print(f"Max negative per row: {max_negative_per_row}")

    # Create a pool of workers
    num_workers = cpu_count() - 2
    with Pool(processes=num_workers) as pool:
        # Generate partial positive pairs in parallel
        partial_func = partial(
            generate_partial_positive_pairs,
            df=df,
            synopses_columns=synopses_columns,
            partial_threshold=partial_threshold,
            max_partial_per_row=max_partial_per_row,
        )
        partial_results = list(
            tqdm(
                pool.imap_unordered(partial_func, range(len(df))),
                total=len(df),
                desc="Creating partial positive pairs",
            )
        )

    # Flatten and add to pairs
    for sublist in partial_results:
        pairs.extend(sublist)

    print(f"Total partial positive pairs: {len(pairs)}")

    # Reset the pool for negative pairs
    with Pool(processes=num_workers) as pool:
        # Generate negative pairs in parallel
        negative_func = partial(
            generate_negative_pairs,
            df=df,
            synopses_columns=synopses_columns,
            partial_threshold=partial_threshold,
            max_negative_per_row=max_negative_per_row,
        )
        negative_results = list(
            tqdm(
                pool.imap_unordered(negative_func, range(len(df))),
                total=len(df),
                desc="Creating negative pairs",
            )
        )

    # Flatten and add to pairs
    for sublist in negative_results:
        pairs.extend(sublist)

    print(f"Total negative pairs: {len(pairs) - max_partial_positive_pairs}")

    return pairs


# Generate the training pairs
def get_pairs(df, use_saved_pairs=True, saved_pairs_file="model/anime_pairs.csv"):
    if use_saved_pairs and os.path.exists(saved_pairs_file):
        print(f"Loading pairs from {saved_pairs_file}")
        pairs_df = pd.read_csv(saved_pairs_file)
        pairs = [
            InputExample(texts=[row["text_a"], row["text_b"]], label=row["label"])
            for _, row in pairs_df.iterrows()
        ]
    else:
        print("Creating new pairs")
        pairs = create_pairs(
            df,
            max_negative_pairs=25000,
            max_partial_positive_pairs=25000,
            partial_threshold=0.5,
        )
        save_pairs_to_csv(pairs, saved_pairs_file)
    return pairs


# Save pairs to a CSV file
def save_pairs_to_csv(pairs, filename):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    data = {
        "text_a": [pair.texts[0] for pair in pairs],
        "text_b": [pair.texts[1] for pair in pairs],
        "label": [pair.label for pair in pairs],
    }
    pairs_df = pd.DataFrame(data)
    pairs_df.to_csv(filename, index=False)
    print(f"Pairs saved to {filename}")


def main():
    # Load your dataset
    df = pd.read_csv("model/merged_anime_dataset.csv")

    # Get the pairs
    pairs = get_pairs(
        df, use_saved_pairs=False, saved_pairs_file="model/anime_pairs.csv"
    )

    # Split the pairs into training and validation sets
    train_pairs, val_pairs = train_test_split(pairs, test_size=0.1)

    # Load the SBERT model
    model = SentenceTransformer("sentence-t5-base")
    model.max_seq_length = 1128
    print(model)

    # Create a DataLoader
    print("Creating DataLoader")
    train_dataloader = DataLoader(train_pairs, shuffle=True, batch_size=3)

    # Prepare validation data for the evaluator
    val_sentences_1 = [pair.texts[0] for pair in val_pairs]
    val_sentences_2 = [pair.texts[1] for pair in val_pairs]
    val_labels = [pair.label for pair in val_pairs]

    # Create the evaluator
    evaluator = EmbeddingSimilarityEvaluator(
        val_sentences_1, val_sentences_2, val_labels
    )

    # Define the loss function (MultipleNegativesRankingLoss, CosineSimilarityLoss, or TripletLoss)
    train_loss = losses.CosineSimilarityLoss(model=model)

    # Fine-tuning the model
    print("Fine-tuning the model")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=1,
        evaluation_steps=1000,
        output_path="model/fine_tuned_sbert_anime_model",
        warmup_steps=500,
        optimizer_params={"lr": 2e-2},
    )

    # Save the model
    model.save("model/fine_tuned_sbert_anime_model")


if __name__ == "__main__":
    main()
