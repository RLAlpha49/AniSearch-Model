"""
This module handles the generation of training pairs for a sentence transformer model.

It provides functionality to create three types of pairs:

1. Positive pairs: Pairs of synopses from same entry with high similarity (>=0.8)

2. Partial positive pairs: Pairs from different entries with moderate similarity
   (>=0.5 and <0.8)

3. Negative pairs: Pairs from different entries with low similarity (<0.5)

The similarity between entries is calculated based on their genres and themes using
semantic embeddings. The module uses multiprocessing for efficient pair generation
and includes functions for both single-row processing and batch processing.

Functions:
    calculate_semantic_similarity: Calculate weighted similarity between genres/themes
    create_positive_pairs: Generate pairs from same-entry synopses with high sim
    generate_partial_positive_pairs: Generate pairs from different entries with
        moderate similarity
    create_partial_positive_pairs: Orchestrate partial positive pair generation
    generate_negative_pairs: Generate pairs from different entries with low sim
    create_negative_pairs: Orchestrate negative pair generation with multiprocessing

The module supports saving generated pairs to CSV files and includes proper error
handling and logging throughout the pair generation process. For all pair types,
the shorter synopsis must be at least 50% the length of the longer synopsis.
"""

import ast
import logging
import random
from functools import partial
from multiprocessing import Pool
from typing import List, Optional, Set, Dict
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import InputExample, SentenceTransformer, util
from tqdm import tqdm
from training.common.data_utils import save_pairs_to_csv  # pylint: disable=import-error no-name-in-module


# Function to calculate semantic similarity between genres/themes
def calculate_semantic_similarity(
    category_to_embedding: Dict[str, NDArray[np.float64]],
    genres_a: Set[str],
    genres_b: Set[str],
    themes_a: Set[str],
    themes_b: Set[str],
    genre_weight: float = 0.35,
    theme_weight: float = 0.65,
) -> float:
    """
    Calculate the semantic similarity between two sets of genres and themes.

    Args:
        category_to_embedding (Dict[str, NDArray[np.float64]]): Dictionary mapping
            categories to embeddings
        genres_a (Set[str]): Set of genres for the first item
        genres_b (Set[str]): Set of genres for the second item
        themes_a (Set[str]): Set of themes for the first item
        themes_b (Set[str]): Set of themes for the second item
        genre_weight (float, optional): Weight for genre similarity.
            Defaults to 0.35
        theme_weight (float, optional): Weight for theme similarity.
            Defaults to 0.65

    Returns:
        float: Weighted semantic similarity score between 0 and 1
    """
    # Calculate cosine similarity for genres
    try:
        if len(genres_a) > 0 and len(genres_b) > 0:
            genre_sim_values = [
                cosine_similarity(
                    [category_to_embedding[g1].astype(np.float64)],  # type: ignore
                    [category_to_embedding[g2].astype(np.float64)],  # type: ignore
                )[0][0]
                for g1 in genres_a
                for g2 in genres_b
                if g1 in category_to_embedding and g2 in category_to_embedding
            ]
            genre_sim = float(np.mean(genre_sim_values)) if genre_sim_values else 0.0
        else:
            genre_sim = 0.0
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Error calculating genre similarity: %s", e)
        genre_sim = 0.0

    # Calculate cosine similarity for themes
    try:
        if len(themes_a) > 0 and len(themes_b) > 0:
            theme_sim_values = [
                cosine_similarity(
                    [category_to_embedding[t1].astype(np.float64)],  # type: ignore
                    [category_to_embedding[t2].astype(np.float64)],  # type: ignore
                )[0][0]
                for t1 in themes_a
                for t2 in themes_b
                if t1 in category_to_embedding and t2 in category_to_embedding
            ]
            theme_sim = float(np.mean(theme_sim_values)) if theme_sim_values else 0.0
        else:
            theme_sim = 0.0
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Error calculating theme similarity: %s", e)
        theme_sim = 0.0

    # Weighted similarity
    similarity = (genre_weight * genre_sim) + (theme_weight * theme_sim)
    return float(similarity)


# Function to create positive pairs
def create_positive_pairs(
    df: pd.DataFrame,
    synopses_columns: List[str],
    encoder_model: SentenceTransformer,
    positive_pairs_file: Optional[str],
) -> List[InputExample]:
    """
    Create positive pairs of synopses from the same entry with high similarity.

    Args:
        df (pd.DataFrame): DataFrame containing the data
        synopses_columns (List[str]): List of column names containing synopses
        encoder_model (SentenceTransformer): Pre-trained sentence transformer model
        positive_pairs_file (Optional[str]): Path to save positive pairs CSV, if provided

    Returns:
        List[InputExample]: List of positive pairs with similarity scores >= 0.8.
        Each pair consists of synopses from the same entry where the shorter
        synopsis is at least 50% the length of the longer one.
    """
    positive_pairs: List[InputExample] = []
    for _, row in tqdm(df.iterrows(), desc="Creating positive pairs", total=len(df)):
        valid_synopses: List[str] = [
            str(row[col]) for col in synopses_columns if pd.notnull(row[col])
        ]
        unique_synopses: List[str] = list(set(valid_synopses))  # Remove duplicates
        if len(unique_synopses) > 1:
            # Encode all synopses
            embeddings: NDArray[np.float64] = encoder_model.encode(
                unique_synopses, convert_to_tensor=False
            )
            for i, embedding_i in enumerate(embeddings):
                for j, embedding_j in enumerate(embeddings[i + 1 :], start=i + 1):
                    # Check if the length condition is met
                    longer_length: int = max(
                        len(unique_synopses[i]), len(unique_synopses[j])
                    )
                    shorter_length: int = min(
                        len(unique_synopses[i]), len(unique_synopses[j])
                    )
                    if shorter_length >= 0.5 * longer_length:
                        # Calculate cosine similarity
                        similarity: float = util.pytorch_cos_sim(
                            torch.tensor(embedding_i), torch.tensor(embedding_j)
                        ).item()
                        if similarity >= 0.8:
                            positive_pairs.append(
                                InputExample(
                                    texts=[unique_synopses[i], unique_synopses[j]],
                                    label=float(similarity),
                                )
                            )  # Positive pair with semantic similarity score

    # Save positive pairs
    save_pairs_to_csv(positive_pairs, positive_pairs_file)
    return positive_pairs


# Function to process a single row for partial positive pairs
def generate_partial_positive_pairs(
    i: int,
    df: pd.DataFrame,
    synopses_columns: List[str],
    partial_threshold: float,
    max_partial_per_row: int,
    category_to_embedding: Dict[str, NDArray[np.float64]],
    valid_indices: List[int],
    max_attempts: int = 200,
) -> List[InputExample]:
    """
    Generate partial positive pairs for a single row in the dataframe.

    Args:
        i (int): Index of the row to process
        df (pd.DataFrame): DataFrame containing the data
        synopses_columns (List[str]): List of column names containing synopses
        partial_threshold (float): Minimum similarity threshold for partial positives
        max_partial_per_row (int): Maximum number of partial positive pairs per row
        category_to_embedding (Dict[str, NDArray[np.float64]]): Category embedding dict
        valid_indices (List[int]): List of valid row indices to sample from
        max_attempts (int, optional): Max attempts to find pairs. Defaults to 200

    Returns:
        List[InputExample]: List of partial positive pairs with similarity between
        partial_threshold+0.01 and 0.8. Each pair consists of synopses from different
        entries where the shorter synopsis is at least 50% the length of the longer one.
    """
    row_a: pd.Series = df.iloc[i]
    pairs: List[InputExample] = []
    partial_count: int = 0
    row_a_partial_count: int = 0
    attempts: int = 0
    used_indices: Set[int] = set()

    while attempts < max_attempts and partial_count < max_partial_per_row:
        available_indices: List[int] = list(set(valid_indices) - used_indices)
        if not available_indices:
            break
        j: int = random.choice(available_indices)
        used_indices.add(j)
        row_b: pd.Series = df.iloc[j]
        try:
            genres_a: Set[str] = (
                set(ast.literal_eval(row_a["genres"]))
                if pd.notnull(row_a["genres"])
                else set()
            )
            genres_b: Set[str] = (
                set(ast.literal_eval(row_b["genres"]))
                if pd.notnull(row_b["genres"])
                else set()
            )

            themes_a: Set[str] = (
                set(ast.literal_eval(row_a["themes"]))
                if pd.notnull(row_a["themes"])
                else set()
            )
            themes_b: Set[str] = (
                set(ast.literal_eval(row_b["themes"]))
                if pd.notnull(row_b["themes"])
                else set()
            )

            # Calculate partial similarity based on genres and themes
            similarity: float = calculate_semantic_similarity(
                category_to_embedding, genres_a, genres_b, themes_a, themes_b
            )

            if similarity >= partial_threshold + 0.01 and similarity <= 0.8:
                # If similarity is above a certain threshold, treat as a partial positive pair
                valid_synopses_a: List[str] = [
                    col for col in synopses_columns if pd.notnull(row_a[col])
                ]
                valid_synopses_b: List[str] = [
                    col for col in synopses_columns if pd.notnull(row_b[col])
                ]

                # Only create a pair if both entries have at least one valid synopsis
                if valid_synopses_a and valid_synopses_b:
                    col_a: str = random.choice(valid_synopses_a)
                    col_b: str = random.choice(valid_synopses_b)

                    # Check if the length condition is met
                    longer_length: int = max(len(row_a[col_a]), len(row_b[col_b]))
                    shorter_length: int = min(len(row_a[col_a]), len(row_b[col_b]))
                    if shorter_length >= 0.5 * longer_length:
                        pairs.append(
                            InputExample(
                                texts=[str(row_a[col_a]), str(row_b[col_b])],
                                label=float(similarity),
                            )
                        )  # Partial positive pair
                        partial_count += 1
                        row_a_partial_count += 1

                        if row_a_partial_count >= max_partial_per_row:
                            break
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("Error processing rows (%d, %d): %s", i, j, e)
            continue
        attempts += 1

    return pairs


# Function to process a single row for negative pairs
def generate_negative_pairs(
    i,
    df,
    synopses_columns,
    partial_threshold,
    max_negative_per_row,
    category_to_embedding,
    valid_indices,
    max_attempts=50,
):
    """
    Generate negative pairs for a single row in the dataframe.

    Args:
        i (int): Index of the row to process
        df (pd.DataFrame): DataFrame containing the data
        synopses_columns (List[str]): List of column names containing synopses
        partial_threshold (float): Maximum similarity threshold for negatives
        max_negative_per_row (int): Maximum number of negative pairs per row
        category_to_embedding (Dict[str, NDArray[np.float64]]): Category embedding dict
        valid_indices (List[int]): List of valid row indices to sample from
        max_attempts (int, optional): Max attempts to find pairs. Defaults to 50

    Returns:
        List[InputExample]: List of negative pairs with similarity between 0.15 and
        partial_threshold-0.01. Each pair consists of synopses from different entries
        where the shorter synopsis is at least 50% the length of the longer one.
    """
    row_a = df.iloc[i]
    pairs = []
    negative_count = 0
    row_a_negative_count = 0
    attempts = 0
    used_indices = set()

    while attempts < max_attempts and negative_count < max_negative_per_row:
        available_indices = list(set(valid_indices) - used_indices)
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
                category_to_embedding,
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

                    # Check if the length condition is met
                    longer_length = max(len(row_a[col_a]), len(row_b[col_b]))
                    shorter_length = min(len(row_a[col_a]), len(row_b[col_b]))
                    if shorter_length >= 0.5 * longer_length:
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
    df: pd.DataFrame,
    synopses_columns: List[str],
    partial_threshold: float,
    max_partial_per_row: int,
    partial_positive_pairs_file: Optional[str],
    num_workers: int,
    category_to_embedding: Dict[str, NDArray[np.float64]],
) -> List[InputExample]:
    """
    Create partial positive pairs from the dataframe using multiprocessing.

    Args:
        df (pd.DataFrame): DataFrame containing the data
        synopses_columns (List[str]): List of column names containing synopses
        partial_threshold (float): Minimum similarity threshold for partial positives
        max_partial_per_row (int): Maximum number of partial positive pairs per row
        partial_positive_pairs_file (Optional[str]): Path to save pairs CSV, if provided
        num_workers (int): Number of worker processes for multiprocessing
        category_to_embedding (Dict[str, NDArray[np.float64]]): Category embedding dict

    Returns:
        List[InputExample]: List of partial positive pairs with similarity between
        partial_threshold+0.01 and 0.8. Each pair consists of synopses from different
        entries where the shorter synopsis is at least 50% the length of the longer one.
    """
    row_indices: List[int] = list(range(len(df)))
    valid_indices: List[int] = []

    for j in tqdm(row_indices, desc="Processing rows for partial positive pairs"):
        themes = df.iloc[j]["themes"]
        if isinstance(themes, str) and len(themes) > 0:
            themes_j = (
                set(ast.literal_eval(themes)) if isinstance(themes, str) else set()
            )
            if themes_j:
                valid_indices.append(j)

    num_workers = max(num_workers, 1)
    with Pool(processes=num_workers) as pool:
        partial_func = partial(
            generate_partial_positive_pairs,
            df=df,
            synopses_columns=synopses_columns,
            partial_threshold=partial_threshold,
            max_partial_per_row=max_partial_per_row,
            category_to_embedding=category_to_embedding,
            valid_indices=valid_indices,
        )
        partial_results: List[List[InputExample]] = list(
            tqdm(
                pool.imap_unordered(partial_func, range(len(df))),
                total=len(df),
                desc="Creating partial positive pairs",
            )
        )

    partial_positive_pairs: List[InputExample] = [
        pair for sublist in partial_results for pair in sublist
    ]
    save_pairs_to_csv(partial_positive_pairs, partial_positive_pairs_file)
    return partial_positive_pairs


# Function to create negative pairs
def create_negative_pairs(
    df: pd.DataFrame,
    synopses_columns: List[str],
    partial_threshold: float,
    max_negative_per_row: int,
    negative_pairs_file: Optional[str],
    num_workers: int,
    category_to_embedding: Dict[str, NDArray[np.float64]],
):
    """
    Create negative pairs from the dataframe using multiprocessing.

    Args:
        df (pd.DataFrame): DataFrame containing the data
        synopses_columns (List[str]): List of column names containing synopses
        partial_threshold (float): Maximum similarity threshold for negatives
        max_negative_per_row (int): Maximum number of negative pairs per row
        negative_pairs_file (Optional[str]): Path to save pairs CSV, if provided
        num_workers (int): Number of worker processes for multiprocessing
        category_to_embedding (Dict[str, NDArray[np.float64]]): Category embedding dict

    Returns:
        List[InputExample]: List of negative pairs with similarity between 0.15 and
        partial_threshold-0.01. Each pair consists of synopses from different entries
        where the shorter synopsis is at least 50% the length of the longer one.
    """
    row_indices = list(range(len(df)))
    valid_indices = []

    for j in tqdm(row_indices, desc="Processing rows for negative pairs"):
        genres = df.iloc[j]["genres"]
        themes = df.iloc[j]["themes"]
        if (isinstance(genres, str) and len(genres) > 0) or (
            isinstance(themes, str) and len(themes) > 0
        ):
            genres_j = (
                set(ast.literal_eval(genres)) if isinstance(genres, str) else set()
            )
            themes_j = (
                set(ast.literal_eval(themes)) if isinstance(themes, str) else set()
            )
            if genres_j and themes_j:
                valid_indices.append(j)

    num_workers = max(num_workers - 2, 1)
    with Pool(processes=num_workers) as pool:
        negative_func = partial(
            generate_negative_pairs,
            df=df,
            synopses_columns=synopses_columns,
            partial_threshold=partial_threshold,
            max_negative_per_row=max_negative_per_row,
            category_to_embedding=category_to_embedding,
            valid_indices=valid_indices,
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
