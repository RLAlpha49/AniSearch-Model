"""
This module implements a Flask application that provides API endpoints
for finding the most similar anime or manga descriptions based on a given model
and description.

The application uses Sentence Transformers to encode descriptions and
calculate cosine similarities between them. It supports multiple synopsis
columns from a dataset and returns the top N most similar descriptions.
"""

# pylint: disable=import-error, global-variable-not-assigned, global-statement

from logging.handlers import RotatingFileHandler
import os
import warnings
import logging
import gc
import threading
import time
import sys
from flask import Flask, request, jsonify, abort
import numpy as np
import pandas as pd
import torch
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from sentence_transformers import SentenceTransformer, util
from werkzeug.exceptions import HTTPException

# Disable oneDNN for TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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

FILE_LOGGING_LEVEL = logging.DEBUG
CONSOLE_LOGGING_LEVEL = logging.INFO

# Create logs directory if not available and configure RotatingFileHandler with UTF-8 encoding
if not os.path.exists("./logs"):
    os.makedirs("./logs")

file_handler = RotatingFileHandler(
    "./logs/api.log",
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

app = Flask(__name__)

# Variable to track the last request time
last_request_time = time.time()
last_request_time_lock = threading.Lock()


def update_last_request_time():
    """
    Updates the last request time to the current time.

    This function acquires a lock to ensure thread safety and updates the
    global variable `last_request_time` with the current time.
    """
    with last_request_time_lock:
        global last_request_time
        last_request_time = time.time()


def periodic_memory_clear():
    """
    Periodically clears memory if the application has been inactive for a specified duration.

    This function runs in a separate thread and checks the time since the last request.
    If the time exceeds 60 seconds, it clears the GPU cache and performs garbage collection
    to free up memory resources.

    The function logs the start of the thread and each memory clearing event.
    """
    logging.info("Starting the periodic memory clear thread.")
    while True:
        with last_request_time_lock:
            current_time = time.time()
            if current_time - last_request_time > 60:
                logging.info("Clearing memory due to inactivity.")
                torch.cuda.empty_cache()
                gc.collect()
        time.sleep(60)


threading.Thread(target=periodic_memory_clear, daemon=True).start()

# Initialize the limiter
limiter = Limiter(get_remote_address, app=app, default_limits=["1 per second"])

# Load the merged datasets
anime_df = pd.read_csv("model/merged_anime_dataset.csv")
manga_df = pd.read_csv("model/merged_manga_dataset.csv")

# List of synopsis columns to consider for anime and manga
anime_synopsis_columns = [
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

manga_synopsis_columns = [
    "synopsis",
    "Synopsis jikan Dataset",
    "Synopsis data Dataset",
]

allowed_models = [
    "sentence-transformers/all-distilroberta-v1",
    "sentence-transformers/all-MiniLM-L6-v1",
    "sentence-transformers/all-MiniLM-L12-v1",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/all-mpnet-base-v1",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-roberta-large-v1",
    "sentence-transformers/gtr-t5-base",
    "sentence-transformers/gtr-t5-large",
    "sentence-transformers/gtr-t5-xl",
    "sentence-transformers/multi-qa-distilbert-dot-v1",
    "sentence-transformers/multi-qa-mpnet-base-cos-v1",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "sentence-transformers/paraphrase-distilroberta-base-v2",
    "sentence-transformers/paraphrase-mpnet-base-v2",
    "sentence-transformers/sentence-t5-base",
    "sentence-transformers/sentence-t5-large",
    "sentence-transformers/sentence-t5-xl",
]


def validate_input(data):
    """
    Validate the input data for the request.

    Args:
        data (dict): The input data containing model name and description.

    Raises:
        werkzeug.exceptions.HTTPException: If the model name or description is missing,
                                           if the description is too long,
                                           or if the model name is invalid.
    """
    model_name = data.get("model")
    description = data.get("description")

    if not model_name or not description:
        logging.error("Model name or description missing in the request.")
        abort(400, description="Model name and description are required")

    if len(description) > 1000:
        logging.error("Description too long.")
        abort(400, description="Description is too long")

    if model_name not in allowed_models:
        logging.error("Invalid model name.")
        abort(400, description="Invalid model name")


def load_embeddings(model_name, col, dataset_type):
    """
    Load embeddings for a given model and column.

    Args:
        model_name (str): The name of the model used to generate embeddings.
        col (str): The column name for which embeddings are to be loaded.
        dataset_type (str): The type of dataset ('anime' or 'manga').

    Returns:
        np.ndarray: A numpy array containing the embeddings for the specified column.

    Raises:
        FileNotFoundError: If the embeddings file does not exist.
    """
    embeddings_file = (
        f"model/{dataset_type}/{model_name}/embeddings_{col.replace(' ', '_')}.npy"
    )
    return np.load(embeddings_file)


def calculate_cosine_similarities(model, model_name, new_embedding, col, dataset_type):
    """
    Calculate cosine similarities between new embedding and existing embeddings for a given column.

    Args:
        model (SentenceTransformer): The sentence transformer model used for encoding.
        new_embedding (np.ndarray): The embedding of the new description.
        col (str): The column name for which to calculate similarities.
        dataset_type (str): The type of dataset ('anime' or 'manga').

    Returns:
        np.ndarray: A numpy array of cosine similarity scores.

    Raises:
        ValueError: If the dimensions of the existing embeddings do not match
                    the model's embedding dimension.
    """
    existing_embeddings = load_embeddings(model_name, col, dataset_type)
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


def get_similarities(model_name, description, dataset_type):
    """
    Find and return the top N most similar descriptions for a given dataset type.

    Args:
        model_name (str): The name of the model to use.
        description (str): The description to compare against the dataset.
        dataset_type (str): The type of dataset ('anime' or 'manga').

    Returns:
        list: List of dictionaries containing top similar descriptions and their similarity scores.
    """
    update_last_request_time()

    # Validate model name
    if model_name not in allowed_models:
        raise ValueError("Invalid model name")

    # Select the appropriate dataset and synopsis columns
    if dataset_type == "anime":
        df = anime_df
        synopsis_columns = anime_synopsis_columns
    else:
        df = manga_df
        synopsis_columns = manga_synopsis_columns

    model = SentenceTransformer(model_name)
    processed_description = description.lower().strip()
    new_pooled_embedding = model.encode([processed_description])

    cosine_similarities_dict = {
        col: calculate_cosine_similarities(
            model, model_name, new_pooled_embedding, col, dataset_type
        )
        for col in synopsis_columns
    }

    all_top_indices = find_top_similarities(cosine_similarities_dict)

    seen_names = set()
    results = []

    for idx, col in all_top_indices:
        name = df.iloc[idx]["title"]
        if name not in seen_names:
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
            seen_names.add(name)
            if len(results) >= 10:
                break

    # Clear memory
    del model, new_pooled_embedding, cosine_similarities_dict
    torch.cuda.empty_cache()
    gc.collect()

    return results


@app.route("/anisearchmodel/anime", methods=["POST"])
@limiter.limit("1 per second")
def get_anime_similarities():
    """
    Handle POST requests to find and return the top N most similar anime descriptions.

    Expects a JSON payload with 'model' and 'description' fields.

    Returns:
        Response: A JSON response with the top similar anime descriptions and the similarity scores.

    Raises:
        400 Bad Request: If the 'model' or 'description' fields are missing from the request.
    """
    try:
        data = request.json
        if data is None:
            raise ValueError("Request payload is missing or not in JSON format")
        validate_input(data)
        model_name = data.get("model")
        description = data.get("description")

        logging.info(
            "Received anime request with model: %s and description: %s",
            model_name,
            description,
        )

        results = get_similarities(model_name, description, "anime")
        logging.info("Returning %d anime results", len(results))
        return jsonify(results)

    except ValueError as e:
        logging.error("Validation error: %s", e)
        return jsonify({"error": "Bad Request"}), 400
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Internal server error: %s", e)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/anisearchmodel/manga", methods=["POST"])  # type: ignore
@limiter.limit("1 per second")
def get_manga_similarities():
    """
    Handle POST requests to find and return the top N most similar manga descriptions.

    Expects a JSON payload with 'model' and 'description' fields.

    Returns:
        Response: A JSON response with the top similar manga descriptions and the similarity scores.

    Raises:
        400 Bad Request: If the 'model' or 'description' fields are missing from the request.
    """
    try:
        data = request.json
        if data is None:
            raise ValueError("Request payload is missing or not in JSON format")
        validate_input(data)
        model_name = data.get("model")
        description = data.get("description")

        logging.info(
            "Received manga request with model: %s and description: %s",
            model_name,
            description,
        )

        results = get_similarities(model_name, description, "manga")
        logging.info("Returning %d manga results", len(results))
        return jsonify(results)

    except HTTPException as e:
        logging.error("HTTP error: %s", e)
        return jsonify({"error": e.description}), e.code
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Internal server error: %s", e)
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() in ["true", "1"]
    app.run(debug=debug_mode, threaded=True)
