"""
This module provides functions for loading models and embeddings, calculating cosine
similarities, and saving evaluation results for anime and manga datasets.

Functions:
    load_model_and_embeddings(model_name, dataset_type)
    calculate_similarities(
        model, df, synopsis_columns, embeddings_save_dir,
        new_description, top_n=10
    )
    save_evaluation_results(
        evaluation_file, model_name, dataset_type,
        new_description, top_results
    )
"""

import os
import warnings
import json
from datetime import datetime
import numpy as np
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


def load_model_and_embeddings(model_name, dataset_type):
    """
    Load the model and embeddings for a given dataset type.

    Args:
        model_name (str): The name of the model to be loaded.
        dataset_type (str): The type of dataset ('anime' or 'manga').

    Returns:
        tuple: A tuple containing the loaded model, dataframe, list of synopsis
        columns, and embeddings save directory.

    Raises:
        ValueError: If an invalid dataset type is specified.
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
    model, df, synopsis_columns, embeddings_save_dir, new_description, top_n=10
):
    """
    Calculate the cosine similarities between a new description and existing embeddings.

    Args:
        model (SentenceTransformer): The model used to encode the new description.
        df (DataFrame): The dataframe containing the dataset.
        synopsis_columns (list): List of columns containing the synopses.
        embeddings_save_dir (str): Directory where the embeddings are saved.
        new_description (str): The new description to compare against existing embeddings.
        top_n (int, optional): The number of top results to return. Defaults to 10.

    Returns:
        list: A list of dictionaries containing the top similarity results.
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
    top_results = []
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
    evaluation_file, model_name, dataset_type, new_description, top_results
):
    """
    Save the evaluation results to a JSON file.

    Args:
        evaluation_file (str): Path to the evaluation results file.
        model_name (str): Name of the model used for evaluation.
        dataset_type (str): Type of the dataset (e.g., anime, manga).
        new_description (str): The new description used for similarity comparison.
        top_results (list): List of top similarity results.

    Returns:
        str: Path to the saved evaluation results file.
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
