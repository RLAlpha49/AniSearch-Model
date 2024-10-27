"""
Provides functionality for semantic similarity search in anime and manga datasets.

This module handles loading pre-trained models and embeddings, calculating semantic
similarities between descriptions, and saving evaluation results. It supports both
anime and manga datasets and uses sentence transformers for embedding generation.

Key Features:
    - Model and embedding loading with automatic device selection
    - Batched similarity calculation using cosine similarity
    - Deduplication of results based on titles
    - Comprehensive evaluation result logging
    - Support for multiple synopsis/description columns

The module is designed to work with pre-computed embeddings stored in numpy arrays
and uses efficient tensor operations for similarity calculations.

Functions:
    load_model_and_embeddings: Loads model, dataset and embeddings for similarity search
    calculate_similarities: Computes semantic similarities between descriptions
    save_evaluation_results: Logs evaluation results with timestamps and metadata
"""

import os
import warnings
import json
from datetime import datetime
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from src import common

# Disable oneDNN for TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Suppress specific warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"`clean_up_tokenization_spaces` was not set. It will be set to `True` by default.",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"1Torch was not compiled with flash attention.",
)
warnings.filterwarnings(
    "ignore",
    message=r"The name tf.losses.sparse_softmax_cross_entropy is deprecated.",
)


def load_model_and_embeddings(
    model_name: str, dataset_type: str
) -> Tuple[SentenceTransformer, pd.DataFrame, List[str], str]:
    """
    Load the model, dataset and pre-computed embeddings for similarity search.

    Handles loading of the appropriate sentence transformer model, dataset and
    pre-computed embeddings based on the specified dataset type. Supports both
    anime and manga datasets with their respective synopsis columns.

    Args:
        model_name (str): Name of the sentence transformer model to load.
            Will prepend 'sentence-transformers/' if not already present.
        dataset_type (str): Type of dataset to load ('anime' or 'manga').
            Determines which dataset and embeddings to load.

    Returns:
        tuple:
            - SentenceTransformer: Loaded model instance
            - pd.DataFrame: Dataset containing titles and synopses
            - List[str]: Names of synopsis columns in the dataset
            - str: Directory path containing pre-computed embeddings

    Raises:
        ValueError: If dataset_type is not 'anime' or 'manga'
    """
    if not model_name.startswith("sentence-transformers/"):
        model_name = f"sentence-transformers/{model_name}"

    if dataset_type == "anime":
        dataset_path = "model/merged_anime_dataset.csv"
        synopsis_columns = [
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
        embeddings_save_dir = f"model/anime/{model_name.split('/')[-1]}"
    elif dataset_type == "manga":
        dataset_path = "model/merged_manga_dataset.csv"
        synopsis_columns = [
            "synopsis",
            "Synopsis jikan Dataset",
            "Synopsis data Dataset",
        ]
        embeddings_save_dir = f"model/manga/{model_name.split('/')[-1]}"
    else:
        raise ValueError("Invalid dataset type specified. Use 'anime' or 'manga'.")

    df = common.load_dataset(dataset_path)
    model = SentenceTransformer(model_name, device="cpu")
    return model, df, synopsis_columns, embeddings_save_dir


def calculate_similarities(
    model: SentenceTransformer,
    df: pd.DataFrame,
    synopsis_columns: List[str],
    embeddings_save_dir: str,
    new_description: str,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Find semantically similar titles by comparing embeddings.

    Calculates cosine similarities between a new description's embedding and
    pre-computed embeddings from the dataset. Returns the top-N most similar
    titles, removing duplicates across different synopsis columns.

    Args:
        model (SentenceTransformer): Model to encode the new description
        df (pd.DataFrame): Dataset containing titles and synopses
        synopsis_columns (List[str]): Columns containing synopsis text
        embeddings_save_dir (str): Directory containing pre-computed embeddings
        new_description (str): Description to find similar titles for
        top_n (int, optional): Number of similar titles to return. Defaults to 10.

    Returns:
        List[Dict[str, Any]]: Top similar titles, each containing:
            - rank: Position in results (1-based)
            - title: Title of the anime/manga
            - synopsis: Plot description/synopsis
            - similarity: Cosine similarity score
            - source_column: Column the synopsis came from

    Raises:
        ValueError: If no valid embeddings are found in embeddings_save_dir
    """
    processed_description = common.preprocess_text(new_description)
    new_pooled_embedding = model.encode(
        [processed_description], convert_to_tensor=True, device="cpu"
    )

    cosine_similarities_dict = {}
    for col in synopsis_columns:
        embeddings_file = os.path.join(
            embeddings_save_dir, f"embeddings_{col.replace(' ', '_')}.npy"
        )
        if not os.path.exists(embeddings_file):
            print(f"Embeddings file not found for column '{col}': {embeddings_file}")
            continue

        existing_embeddings = np.load(embeddings_file)
        existing_embeddings_tensor = torch.tensor(existing_embeddings).to("cpu")

        with torch.no_grad():
            cosine_similarities = (
                util.pytorch_cos_sim(new_pooled_embedding, existing_embeddings_tensor)
                .squeeze(0)
                .cpu()
                .numpy()
            )

        cosine_similarities_dict[col] = cosine_similarities

    if not cosine_similarities_dict:
        raise ValueError(
            "No valid embeddings were loaded. Please check your embeddings directory and files."
        )

    all_top_indices = []
    for col, cosine_similarities in cosine_similarities_dict.items():
        top_indices_unsorted = np.argsort(cosine_similarities)[-top_n:]
        top_indices = top_indices_unsorted[
            np.argsort(cosine_similarities[top_indices_unsorted])[::-1]
        ]
        all_top_indices.extend([(idx, col) for idx in top_indices])

    all_top_indices.sort(
        key=lambda x: cosine_similarities_dict[x[1]][x[0]], reverse=True
    )

    seen_names = set()
    top_results: List[Dict[str, Any]] = []
    for idx, col in all_top_indices:
        if len(top_results) >= top_n:
            break
        name = df.iloc[idx]["title"]
        if name in seen_names:
            continue
        synopsis = df.iloc[idx][col]
        similarity = cosine_similarities_dict[col][idx]
        top_results.append(
            {
                "rank": len(top_results) + 1,
                "title": name,
                "synopsis": synopsis,
                "similarity": float(similarity),
                "source_column": col,
            }
        )
        seen_names.add(name)

    return top_results


def save_evaluation_results(
    evaluation_file: str,
    model_name: str,
    dataset_type: str,
    new_description: str,
    top_results: List[Dict[str, Any]],
) -> str:
    """
    Save similarity search results with metadata for evaluation.

    Appends the search results and metadata to a JSON file for later analysis.
    Creates a new file if it doesn't exist. Each entry includes a timestamp,
    model information, dataset type, query description, and similarity results.

    Args:
        evaluation_file (str): Path to save/append results
        model_name (str): Name of model used for embeddings
        dataset_type (str): Type of dataset searched ('anime' or 'manga')
        new_description (str): Query description used for search
        top_results (List[Dict[str, Any]]): Similarity search results

    Returns:
        str: Path to the evaluation file

    The saved JSON structure includes:
        - timestamp: When the search was performed
        - model_name: Model used for embeddings
        - dataset_type: Type of dataset searched
        - new_description: Query description
        - top_similarities: List of similar titles and their scores
    """
    if os.path.exists(evaluation_file):
        with open(evaluation_file, "r", encoding="utf-8") as f:
            try:
                evaluation_data = json.load(f)
            except json.JSONDecodeError:
                evaluation_data = []
    else:
        evaluation_data = []

    test_result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "dataset_type": dataset_type,
        "new_description": new_description,
        "top_similarities": top_results,
    }

    evaluation_data.append(test_result)

    with open(evaluation_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_data, f, indent=4)

    return evaluation_file
