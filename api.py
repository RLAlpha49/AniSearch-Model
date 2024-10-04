"""
This module implements a Flask application that provides an API endpoint
for finding the most similar anime descriptions based on a given model
and description.

The application uses Sentence Transformers to encode descriptions and
calculate cosine similarities between them. It supports multiple synopsis
columns from a dataset and returns the top N most similar descriptions.
"""

import os
import warnings
import logging
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import torch

# Disable oneDNN for TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from sentence_transformers import (  # pylint: disable=wrong-import-position  # noqa: E402
    SentenceTransformer,
    util,
)

# Suppress the specific FutureWarning and DeprecationWarning from transformers
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"`clean_up_tokenization_spaces` was not set. It will be set to `True` by default.",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
)

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Load the merged dataset
df = pd.read_csv("model/merged_anime_dataset.csv")

# List of synopsis columns to consider
synopsis_columns = [
    "synopsis",
    "Synopsis anime_dataset_2023",
    "Synopsis animes dataset",
    "Synopsis anime_270 Dataset",
    "Synopsis Anime-2022 Dataset",
    "Synopsis Anime Dataset",
    "Synopsis anime4500 Dataset",
    "Synopsis anime-20220927-raw Dataset",
    "Synopsis wykonos Dataset",
    "Synopsis Anime_data Dataset",
    "Synopsis anime2 Dataset",
    "Synopsis mal_anime Dataset",
]


def load_embeddings(model_name, col):
    """
    Load embeddings for a given model and column.

    Args:
        model_name (str): The name of the model used to generate embeddings.
        col (str): The column name for which embeddings are to be loaded.

    Returns:
        np.ndarray: A numpy array containing the embeddings for the specified column.

    Raises:
        FileNotFoundError: If the embeddings file does not exist.
    """
    embeddings_file = f"model/{model_name}/embeddings_{col.replace(' ', '_')}.npy"
    return np.load(embeddings_file)


def calculate_cosine_similarities(model, model_name, new_embedding, col):
    """
    Calculate cosine similarities between new embedding and existing embeddings for a given column.

    Args:
        model (SentenceTransformer): The sentence transformer model used for encoding.
        new_embedding (np.ndarray): The embedding of the new description.
        col (str): The column name for which to calculate similarities.

    Returns:
        np.ndarray: A numpy array of cosine similarity scores.

    Raises:
        ValueError: If the dimensions of the existing embeddings do not match
                    the model's embedding dimension.
    """
    existing_embeddings = load_embeddings(model_name, col)
    if existing_embeddings.shape[1] != model.get_sentence_embedding_dimension():
        raise ValueError(f"Incompatible dimension for embeddings in {col}")
    return (
        util.pytorch_cos_sim(
            torch.tensor(new_embedding), torch.tensor(existing_embeddings)
        )
        .flatten()
        .cpu()
        .numpy()
    )


def find_top_similarities(cosine_similarities_dict, num_similarities=10):
    """
    Find the top N most similar descriptions across all columns based on cosine similarity scores.

    Args:
        cosine_similarities_dict (dict): A dictionary where keys are column names
            and values are arrays of cosine similarity scores.
        num_similarities (int, optional): The number of top similarities to find.
            Defaults to 10.

    Returns:
        list: A list of tuples, containing the index and column name of the similar descriptions.
    """
    all_top_indices = []
    for col, cosine_similarities in cosine_similarities_dict.items():
        top_indices_unsorted = np.argsort(cosine_similarities)[-num_similarities:]
        top_indices = top_indices_unsorted[
            np.argsort(cosine_similarities[top_indices_unsorted])[::-1]
        ]
        all_top_indices.extend([(idx, col) for idx in top_indices])
    all_top_indices.sort(
        key=lambda x: cosine_similarities_dict[x[1]][x[0]], reverse=True
    )
    return all_top_indices


@app.route("/anisearchmodel", methods=["POST"])
def get_similarities():
    """
    Handle POST requests to find and return the top N most similar anime descriptions.

    Expects a JSON payload with 'model' and 'description' fields.

    Returns:
        Response: A JSON response with the top similar anime descriptions and the similarity scores.

    Raises:
        400 Bad Request: If the 'model' or 'description' fields are missing from the request.
    """
    data = request.json
    model_name = data.get("model")
    description = data.get("description")

    # Log the incoming request
    logging.info(
        "Received request with model: %s and description: %s", model_name, description
    )

    if not model_name or not description:
        logging.error("Model name or description missing in the request.")
        return jsonify({"error": "Model name and description are required"}), 400

    # Load the SBERT model
    model = SentenceTransformer(model_name)

    # Encode the new description
    processed_description = description.lower().strip()
    new_pooled_embedding = model.encode([processed_description])

    # Calculate cosine similarities for each synopsis column
    cosine_similarities_dict = {
        col: calculate_cosine_similarities(model, model_name, new_pooled_embedding, col)
        for col in synopsis_columns
    }

    # Find and return the top N most similar descriptions
    all_top_indices = find_top_similarities(cosine_similarities_dict)

    seen_anime_names = set()
    results = []

    for idx, col in all_top_indices:
        name = df.iloc[idx]["title"]
        if name not in seen_anime_names:
            synopsis = df.iloc[idx][col]
            similarity = float(cosine_similarities_dict[col][idx])
            results.append(
                {
                    "rank": len(results) + 1,
                    "name": name,
                    "synopsis": synopsis,
                    "similarity": similarity,
                }
            )
            seen_anime_names.add(name)
            if len(results) >= 10:
                break

    logging.info("Returning %d results", len(results))
    return jsonify(results)


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() in ["true", "1"]
    app.run(debug=debug_mode, threaded=True)
