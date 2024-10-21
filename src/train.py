"""
This module contains functions and classes for training a SentenceTransformer model
using positive, partial positive, and negative pairs of text data. It includes functions
for generating these pairs, saving them to CSV files, and loading them for training.
The main function parses command-line arguments and fine-tunes a pre-trained SentenceTransformer
model on the provided dataset.
"""

import argparse
import ast
import os
import logging
from functools import partial
from multiprocessing import Pool, cpu_count
import random
import gc
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import torch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf  # pylint: disable=wrong-import-position

# Set TensorFlow's logging level to ERROR
tf.get_logger().setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer, InputExample, losses, util  # pylint: disable=wrong-import-position # noqa: E402
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator  # pylint: disable=wrong-import-position # noqa: E402


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

# Combine genres and themes
all_categories = list(all_genres) + list(all_themes)

# Load a pre-trained Sentence Transformer model for encoding
encoder_model = SentenceTransformer("sentence-t5-base")

# Generate embeddings for all categories
category_embeddings = encoder_model.encode(all_categories, convert_to_tensor=False)

# Create a mapping from category to its embedding
category_to_embedding = {
    category: embedding
    for category, embedding in zip(all_categories, category_embeddings)
}


# Function to calculate semantic similarity between genres/themes
def calculate_semantic_similarity(
    genres_a, genres_b, themes_a, themes_b, genre_weight=0.35, theme_weight=0.65
):
    """
    Calculate the semantic similarity between two sets of genres and themes.

    Args:
        genres_a (set): Set of genres for the first item.
        genres_b (set): Set of genres for the second item.
        themes_a (set): Set of themes for the first item.
        themes_b (set): Set of themes for the second item.
        genre_weight (float): Weight for the genre similarity. Default is 0.35.
        theme_weight (float): Weight for the theme similarity. Default is 0.65.

    Returns:
        float: The weighted semantic similarity score.
    """
    # Calculate cosine similarity for genres
    try:
        if len(genres_a) > 0 and len(genres_b) > 0:
            genre_sim_values = [
                cosine_similarity(
                    [category_to_embedding[g1]],  # type: ignore
                    [category_to_embedding[g2]],  # type: ignore
                )[0][0]
                for g1 in genres_a
                for g2 in genres_b
                if g1 in category_to_embedding and g2 in category_to_embedding
            ]
            genre_sim = np.mean(genre_sim_values)
        else:
            genre_sim = 0.0
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(e)
        genre_sim = 0.0

    # Calculate cosine similarity for themes
    try:
        if len(themes_a) > 0 and len(themes_b) > 0:
            theme_sim_values = [
                cosine_similarity(
                    [category_to_embedding[t1]],  # type: ignore
                    [category_to_embedding[t2]],  # type: ignore
                )[0][0]
                for t1 in themes_a
                for t2 in themes_b
                if t1 in category_to_embedding and t2 in category_to_embedding
            ]
            theme_sim = np.mean(theme_sim_values)
        else:
            theme_sim = 0.0
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(e)
        theme_sim = 0.0

    # Weighted similarity
    similarity = (genre_weight * genre_sim) + (theme_weight * theme_sim)
    return similarity


# Save pairs to a CSV file
def save_pairs_to_csv(pairs, filename):
    """
    Save pairs of texts and their labels to a CSV file.

    Args:
        pairs (list): List of InputExample pairs.
        filename (str): Path to the CSV file where pairs will be saved.
    """
    # Ensure the directory exists
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    data = {
        "text_a": [pair.texts[0] for pair in pairs],
        "text_b": [pair.texts[1] for pair in pairs],
        "label": [pair.label for pair in pairs],
    }
    pairs_df = pd.DataFrame(data)
    pairs_df.to_csv(filename, index=False)
    print(f"Pairs saved to {filename}")


# Function to create positive pairs
def create_positive_pairs(df, synopses_columns, encoder_model, positive_pairs_file):  # pylint: disable=redefined-outer-name
    """
    Create positive pairs of synopses from the dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        synopses_columns (list): List of columns containing synopses.
        encoder_model (SentenceTransformer): Pre-trained Sentence Transformer model.
        positive_pairs_file (str): Path to the file where positive pairs will be saved.

    Returns:
        list: List of positive InputExample pairs.
    """
    positive_pairs = []
    for _, row in tqdm(df.iterrows(), desc="Creating positive pairs", total=len(df)):
        valid_synopses = [row[col] for col in synopses_columns if pd.notnull(row[col])]
        unique_synopses = list(set(valid_synopses))  # Remove duplicates
        if len(unique_synopses) > 1:
            # Encode all synopses
            embeddings = encoder_model.encode(unique_synopses, convert_to_tensor=False)
            for i, embedding_i in enumerate(embeddings):
                for j, embedding_j in enumerate(embeddings[i + 1 :], start=i + 1):
                    # Check if the length condition is met
                    longer_length = max(
                        len(unique_synopses[i]), len(unique_synopses[j])
                    )
                    shorter_length = min(
                        len(unique_synopses[i]), len(unique_synopses[j])
                    )
                    if shorter_length >= 0.5 * longer_length:
                        # Calculate cosine similarity
                        similarity = util.pytorch_cos_sim(
                            torch.tensor(embedding_i), torch.tensor(embedding_j)
                        ).item()
                        if similarity >= 0.8:
                            positive_pairs.append(
                                InputExample(
                                    texts=[unique_synopses[i], unique_synopses[j]],
                                    label=similarity,
                                )
                            )  # Positive pair with semantic similarity score

    # Save positive pairs
    save_pairs_to_csv(positive_pairs, positive_pairs_file)
    return positive_pairs


# Function to process a single row for partial positive pairs
def generate_partial_positive_pairs(
    i, df, synopses_columns, partial_threshold, max_partial_per_row, max_attempts=25
):
    """
    Generate partial positive pairs for a single row in the dataframe.

    Args:
        i (int): Index of the row to process.
        df (pd.DataFrame): DataFrame containing the data.
        synopses_columns (list): List of columns containing synopses.
        partial_threshold (float): Threshold for partial similarity.
        max_partial_per_row (int): Maximum number of partial positive pairs per row.
        max_attempts (int): Maximum number of attempts to find pairs. Default is 25.

    Returns:
        list: List of partial positive InputExample pairs.
    """
    row_a = df.iloc[i]
    pairs = []
    partial_count = 0
    row_a_partial_count = 0
    attempts = 0
    row_indices = list(range(len(df)))
    row_indices.remove(i)
    used_indices = set()

    while attempts < max_attempts and partial_count < max_partial_per_row:
        available_indices = list(set(row_indices) - used_indices)
        if not available_indices:
            break
        j = random.choice(available_indices)
        used_indices.add(j)
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
            similarity = calculate_semantic_similarity(
                genres_a, genres_b, themes_a, themes_b
            )

            if similarity >= partial_threshold + 0.01 and similarity <= 0.8:
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
                            texts=[row_a[col_a], row_b[col_b]],
                            label=similarity,  # type: ignore
                        )
                    )  # Partial positive pair
                    partial_count += 1
                    row_a_partial_count += 1

                    if row_a_partial_count >= max_partial_per_row:
                        break
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(e)
            continue
        attempts += 1

    return pairs


# Function to process a single row for negative pairs
def generate_negative_pairs(
    i, df, synopses_columns, partial_threshold, max_negative_per_row, max_attempts=25
):
    """
    Generate negative pairs for a single row in the dataframe.

    Args:
        i (int): Index of the row to process.
        df (pd.DataFrame): DataFrame containing the data.
        synopses_columns (list): List of columns containing synopses.
        partial_threshold (float): Threshold for partial similarity.
        max_negative_per_row (int): Maximum number of negative pairs per row.
        max_attempts (int): Maximum number of attempts to find pairs. Default is 25.

    Returns:
        list: List of negative InputExample pairs.
    """
    row_a = df.iloc[i]
    pairs = []
    negative_count = 0
    row_a_negative_count = 0
    attempts = 0
    row_indices = list(range(len(df)))
    row_indices.remove(i)
    used_indices = set()

    while attempts < max_attempts and negative_count < max_negative_per_row:
        available_indices = list(set(row_indices) - used_indices)
        if not available_indices:
            break
        j = random.choice(available_indices)
        used_indices.add(j)
        row_b = df.iloc[j]
        try:
            # Check for NaN values before parsing
            genres_a = row_a["genres"]
            genres_b = row_b["genres"]
            themes_a = row_a["themes"]
            themes_b = row_b["themes"]

            if (
                pd.isna(genres_a)
                or pd.isna(genres_b)
                or pd.isna(themes_a)
                or pd.isna(themes_b)
            ):
                continue  # Skip rows with NaN values

            # Compute similarity
            similarity = calculate_semantic_similarity(
                set(ast.literal_eval(genres_a)),
                set(ast.literal_eval(genres_b)),
                set(ast.literal_eval(themes_a)),
                set(ast.literal_eval(themes_b)),
            )

            if similarity <= partial_threshold - 0.01 and similarity >= 0.15:
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
                            texts=[row_a[col_a], row_b[col_b]],
                            label=similarity,  # type: ignore
                        )
                    )  # Partial or negative pair
                    negative_count += 1
                    row_a_negative_count += 1

                    if row_a_negative_count >= max_negative_per_row:
                        break
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(e)
            continue
        attempts += 1

    return pairs


# Function to create partial positive pairs
def create_partial_positive_pairs(
    df,
    synopses_columns,
    partial_threshold,
    max_partial_per_row,
    partial_positive_pairs_file,
):
    """
    Create partial positive pairs from the dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        synopses_columns (list): List of columns containing synopses.
        partial_threshold (float): Threshold for partial similarity.
        max_partial_per_row (int): Maximum number of partial positive pairs per row.
        partial_positive_pairs_file (str): Path to where partial positive pairs will be saved.

    Returns:
        list: List of partial positive InputExample pairs.
    """
    num_workers = max(cpu_count() - 2 - 4, 1)
    with Pool(processes=num_workers) as pool:
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

    partial_positive_pairs = [pair for sublist in partial_results for pair in sublist]
    save_pairs_to_csv(partial_positive_pairs, partial_positive_pairs_file)
    return partial_positive_pairs


# Function to create negative pairs
def create_negative_pairs(
    df,
    synopses_columns,
    partial_threshold,
    max_negative_per_row,
    negative_pairs_file,
):
    """
    Create negative pairs from the dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        synopses_columns (list): List of columns containing synopses.
        partial_threshold (float): Threshold for partial similarity.
        max_negative_per_row (int): Maximum number of negative pairs per row.
        negative_pairs_file (str): Path to the file where negative pairs will be saved.

    Returns:
        list: List of negative InputExample pairs.
    """
    num_workers = max(cpu_count() - 2 - 4, 1)
    with Pool(processes=num_workers) as pool:
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

    negative_pairs = [pair for sublist in negative_results for pair in sublist]
    save_pairs_to_csv(negative_pairs, negative_pairs_file)
    return negative_pairs


# Function to create positive and negative pairs
def create_pairs(
    df,
    max_negative_pairs,
    max_partial_positive_pairs,
    partial_threshold=0.5,
    positive_pairs_file=None,
    partial_positive_pairs_file=None,
    negative_pairs_file=None,
):
    """
    Create positive, partial positive, and negative pairs from the dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        max_negative_pairs (int): Maximum number of negative pairs.
        max_partial_positive_pairs (int): Maximum number of partial positive pairs.
        partial_threshold (float): Threshold for partial similarity. Default is 0.5.
        positive_pairs_file (str): Path to the file where positive pairs will be saved.
        partial_positive_pairs_file (str): Path to where partial positive pairs will be saved.
        negative_pairs_file (str): Path to the file where negative pairs will be saved.

    Returns:
        tuple: Lists of positive, partial positive, and negative InputExample pairs.
    """
    synopses_columns = [col for col in df.columns if "synopsis" in col.lower()]

    # Load a pre-trained Sentence Transformer model for encoding
    encoder_model = SentenceTransformer("sentence-t5-xl")  # pylint: disable=redefined-outer-name

    # Generate positive pairs if not already saved
    positive_pairs = []
    if positive_pairs_file is None or not os.path.exists(positive_pairs_file):
        positive_pairs = create_positive_pairs(
            df, synopses_columns, encoder_model, positive_pairs_file
        )
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()

    # Generate partial positive pairs if not already saved
    partial_positive_pairs = []
    if partial_positive_pairs_file is None or not os.path.exists(
        partial_positive_pairs_file
    ):
        max_partial_per_row = (
            int(max_partial_positive_pairs / len(df)) if len(df) > 0 else 0
        )
        partial_positive_pairs = create_partial_positive_pairs(
            df,
            synopses_columns,
            partial_threshold,
            max_partial_per_row,
            partial_positive_pairs_file,
        )
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()

    # Generate negative pairs if not already saved
    negative_pairs = []
    if negative_pairs_file is None or not os.path.exists(negative_pairs_file):
        max_negative_per_row = int(max_negative_pairs / len(df)) if len(df) > 0 else 0
        negative_pairs = create_negative_pairs(
            df,
            synopses_columns,
            partial_threshold,
            max_negative_per_row,
            negative_pairs_file,
        )
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()

    return positive_pairs, partial_positive_pairs, negative_pairs


# Function to get the pairs
def get_pairs(
    df,
    use_saved_pairs,
    saved_pairs_directory,
    max_negative_pairs,
    max_partial_positive_pairs,
):
    """
    Get positive, partial positive, and negative pairs from the dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        use_saved_pairs (bool): Whether to use saved pairs.
        saved_pairs_directory (str): Directory to save/load pairs.
        max_negative_pairs (int): Maximum number of negative pairs.
        max_partial_positive_pairs (int): Maximum number of partial positive pairs.

    Returns:
        list: Combined list of positive, partial positive, and negative InputExample pairs.
    """
    positive_pairs_file = os.path.join(saved_pairs_directory, "positive_pairs.csv")
    partial_positive_pairs_file = os.path.join(
        saved_pairs_directory, "partial_positive_pairs.csv"
    )
    negative_pairs_file = os.path.join(saved_pairs_directory, "negative_pairs.csv")

    # Initialize lists to store pairs
    positive_pairs = []
    partial_positive_pairs = []
    negative_pairs = []

    # Load existing pairs if available
    if use_saved_pairs:
        if os.path.exists(positive_pairs_file):
            print(f"Loading positive pairs from {positive_pairs_file}")
            positive_pairs_df = pd.read_csv(positive_pairs_file)
            positive_pairs = [
                InputExample(texts=[row["text_a"], row["text_b"]], label=row["label"])
                for _, row in positive_pairs_df.iterrows()
            ]

        if os.path.exists(partial_positive_pairs_file):
            print(f"Loading partial positive pairs from {partial_positive_pairs_file}")
            partial_positive_pairs_df = pd.read_csv(partial_positive_pairs_file)
            partial_positive_pairs = [
                InputExample(texts=[row["text_a"], row["text_b"]], label=row["label"])
                for _, row in partial_positive_pairs_df.iterrows()
            ]

        if os.path.exists(negative_pairs_file):
            print(f"Loading negative pairs from {negative_pairs_file}")
            negative_pairs_df = pd.read_csv(negative_pairs_file)
            negative_pairs = [
                InputExample(texts=[row["text_a"], row["text_b"]], label=row["label"])
                for _, row in negative_pairs_df.iterrows()
            ]

    # Generate missing pairs
    if not positive_pairs or not partial_positive_pairs or not negative_pairs:
        print("Generating missing pairs")
        (
            generated_positive_pairs,
            generated_partial_positive_pairs,
            generated_negative_pairs,
        ) = create_pairs(
            df,
            max_negative_pairs=max_negative_pairs,
            max_partial_positive_pairs=max_partial_positive_pairs,
            partial_threshold=0.5,
            positive_pairs_file=positive_pairs_file,
            partial_positive_pairs_file=partial_positive_pairs_file,
            negative_pairs_file=negative_pairs_file,
        )

        # Only update the lists with newly generated pairs if they were missing
        if not positive_pairs:
            positive_pairs = generated_positive_pairs
        if not partial_positive_pairs:
            partial_positive_pairs = generated_partial_positive_pairs
        if not negative_pairs:
            negative_pairs = generated_negative_pairs

    # Combine all pairs
    pairs = positive_pairs + partial_positive_pairs + negative_pairs
    return pairs


def main():
    """
    Main function to train a SentenceTransformer model.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a SentenceTransformer model.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-t5-base",
        help="Name of the model to train. Default is 'sentence-t5-base'.",
    )
    parser.add_argument(
        "--use_saved_pairs",
        type=bool,
        default=False,
        help="Whether to use saved pairs. Default is False.",
    )
    parser.add_argument(
        "--saved_pairs_directory",
        type=str,
        default="model",
        help="Directory to save/load pairs. Default is 'model'.",
    )
    parser.add_argument(
        "--max_negative_pairs",
        type=int,
        default=50000,
        help="Maximum number of negative pairs. Default is 50000.",
    )
    parser.add_argument(
        "--max_partial_positive_pairs",
        type=int,
        default=50000,
        help="Maximum number of partial positive pairs. Default is 50000.",
    )
    parser.add_argument(
        "--output_model_path",
        type=str,
        default="model/fine_tuned_sbert_anime_model",
        help="Path to save the fine-tuned model. Default is 'model/fine_tuned_sbert_anime_model'.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for the optimizer. Default is 2e-5 (0.00002).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=3,
        help="Batch size for training. Default is 3.",
    )
    parser.add_argument(
        "--evaluations_per_epoch",
        type=int,
        default=20,
        help="Number of evaluations per epoch. Default is 20.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of epochs for training. Default is 2.",
    )
    parser.add_argument(
        "--loss_function",
        type=str,
        choices=["cosine", "cosent", "angle"],
        default="cosine",
        help="Loss function to use: 'cosine', 'cosent', or 'angle'. Default is 'cosine'.",
    )
    args = parser.parse_args()

    # Load your dataset
    df = pd.read_csv("model/merged_anime_dataset.csv")

    # Get the pairs
    pairs = get_pairs(
        df,
        use_saved_pairs=args.use_saved_pairs,
        saved_pairs_directory=args.saved_pairs_directory,
        max_negative_pairs=args.max_negative_pairs,
        max_partial_positive_pairs=args.max_partial_positive_pairs,
    )

    # Split the pairs into training and validation sets
    train_pairs, val_pairs = train_test_split(pairs, test_size=0.1)

    # Load the SBERT model
    model = SentenceTransformer(args.model_name)
    model.max_seq_length = 1128
    print(model)

    # Create a DataLoader
    print("Creating DataLoader")
    train_dataloader = DataLoader(train_pairs, shuffle=True, batch_size=args.batch_size)

    # Calculate the number of batches per epoch
    num_batches_per_epoch = len(train_dataloader)

    # Calculate evaluation steps
    evaluation_steps = num_batches_per_epoch // args.evaluations_per_epoch

    # Prepare validation data for the evaluator
    val_sentences_1 = [pair.texts[0] for pair in val_pairs]
    val_sentences_2 = [pair.texts[1] for pair in val_pairs]
    val_labels = [pair.label for pair in val_pairs]

    # Create the evaluator
    evaluator = EmbeddingSimilarityEvaluator(
        val_sentences_1,
        val_sentences_2,
        val_labels,
        main_similarity="cosine",
        write_csv=True,
        precision="float32",
    )

    # Define the loss function based on the argument
    if args.loss_function == "cosine":
        train_loss = losses.CosineSimilarityLoss(model=model)
    elif args.loss_function == "cosent":
        train_loss = losses.CoSENTLoss(model=model)
    elif args.loss_function == "angle":
        train_loss = losses.AnglELoss(model=model)
    else:
        raise ValueError(f"Unsupported loss function: {args.loss_function}")

    # Fine-tuning the model
    print("Fine-tuning the model")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        evaluation_steps=evaluation_steps,
        output_path=args.output_model_path,
        warmup_steps=evaluation_steps // 2,
        optimizer_params={"lr": args.learning_rate},
    )

    # Save the model
    model.save(args.output_model_path)


if __name__ == "__main__":
    main()
