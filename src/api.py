"""
This module implements a Flask application that provides API endpoints
for finding the most similar anime or manga descriptions based on a given model
and description.

The application uses Sentence Transformers to encode descriptions and
calculate cosine similarities between them. It supports multiple synopsis
columns from a dataset and returns the top N most similar descriptions.
"""

# pylint: disable=import-error, global-variable-not-assigned, global-statement

import os
import warnings
import logging
import gc
import threading
import time
import sys
from typing import Any, List, Dict, Tuple
from concurrent_log_handler import ConcurrentRotatingFileHandler
from flask import Flask, request, jsonify, abort, Response, make_response
from flask_cors import CORS
import numpy as np
import pandas as pd
import torch
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from sentence_transformers import SentenceTransformer, util
from werkzeug.exceptions import HTTPException

# Determine the device to use based on the environment variable
device = (
    "cuda"
    if os.getenv("DEVICE", "cpu") == "cuda" and torch.cuda.is_available()
    else "cpu"
)

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

file_handler = ConcurrentRotatingFileHandler(
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
CORS(
    app,
    resources={
        r"/*": {"origins": ["https://anisearch.alpha49.com", "http://localhost:3000"]}
    },
)

# Variable to track the last request time
last_request_time = time.time()
last_request_time_lock = threading.Lock()


def update_last_request_time() -> None:
    """
    Updates the last request time to the current time.

    This function acquires a lock to ensure thread safety and updates the
    global variable `last_request_time` with the current time.
    """
    with last_request_time_lock:
        global last_request_time
        last_request_time = time.time()


def clear_memory() -> None:
    """
    Clears the GPU cache and performs garbage collection to free up memory resources.
    """
    torch.cuda.empty_cache()
    gc.collect()


def periodic_memory_clear() -> None:
    """
    Periodically clears memory if the application has been inactive for a specified duration.

    This function runs in a separate thread and checks the time since the last request.
    If the time exceeds 300 seconds, it clears the GPU cache and performs garbage collection
    to free up memory resources.

    The function logs the start of the thread and each memory clearing event.
    """
    logging.info("Starting the periodic memory clear thread.")
    while True:
        with last_request_time_lock:
            current_time = time.time()
            if current_time - last_request_time > 300:
                logging.debug("Clearing memory due to inactivity.")
                clear_memory()
        time.sleep(300)


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


def validate_input(data: Dict[str, Any]) -> None:
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


def load_embeddings(model_name: str, col: str, dataset_type: str) -> np.ndarray:
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


def calculate_cosine_similarities(
    model: SentenceTransformer,
    model_name: str,
    new_embedding: np.ndarray,
    col: str,
    dataset_type: str,
) -> np.ndarray:
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
    model_name = model_name.replace("sentence-transformers/", "")
    existing_embeddings = load_embeddings(model_name, col, dataset_type)
    if existing_embeddings.shape[1] != model.get_sentence_embedding_dimension():
        raise ValueError(f"Incompatible dimension for embeddings in {col}")
    new_embedding_tensor = torch.tensor(new_embedding).to(device)
    existing_embeddings_tensor = torch.tensor(existing_embeddings).to(device)
    return (
        util.pytorch_cos_sim(new_embedding_tensor, existing_embeddings_tensor)
        .flatten()
        .cpu()
        .numpy()
    )


def find_top_similarities(
    cosine_similarities_dict: Dict[str, np.ndarray], num_similarities: int = 10
) -> List[Tuple[int, str]]:
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
        key=lambda x: cosine_similarities_dict[x[1]][x[0]],  # type: ignore
        reverse=True,
    )  # type: ignore
    return all_top_indices


def get_similarities(
    model_name: str,
    description: str,
    dataset_type: str,
    page: int = 1,
    results_per_page: int = 10,
) -> List[Dict[str, Any]]:
    """
    Find and return the top N most similar descriptions for a given dataset type.

    Args:
        model_name (str): The name of the model to use.
        description (str): The description to compare against the dataset.
        dataset_type (str): The type of dataset ('anime' or 'manga').
        page (int, optional): The page number of results to return. Defaults to 1.
        results_per_page (int, optional): The number of results per page. Defaults to 10.

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

    model = SentenceTransformer(model_name, device=device)
    processed_description = description.lower().strip()
    new_pooled_embedding = model.encode([processed_description])

    cosine_similarities_dict = {
        col: calculate_cosine_similarities(
            model, model_name, new_pooled_embedding, col, dataset_type
        )
        for col in synopsis_columns
    }

    all_top_indices = find_top_similarities(
        cosine_similarities_dict, num_similarities=page * results_per_page
    )

    seen_names = set()
    results: List[Dict[str, Any]] = []

    for idx, col in all_top_indices:
        name = df.iloc[idx]["title"]
        if name not in seen_names:
            row_data = df.iloc[idx].to_dict()  # Convert the entire row to a dictionary
            # Keep only the relevant synopsis column
            relevant_synopsis = df.iloc[idx][col]
            row_data = {
                k: v
                for k, v in row_data.items()
                if k not in synopsis_columns or k == col
            }
            row_data.update(
                {
                    "rank": len(results) + 1,
                    "similarity": float(cosine_similarities_dict[col][idx]),
                    "synopsis": relevant_synopsis,  # Ensure the correct synopsis is included
                }
            )
            results.append(row_data)
            seen_names.add(name)
            if len(results) >= page * results_per_page:
                break

    # Clear memory
    del model, new_pooled_embedding, cosine_similarities_dict
    clear_memory()

    # Calculate start and end indices for pagination
    start_index = (page - 1) * results_per_page
    end_index = start_index + results_per_page

    return results[start_index:end_index]


@app.route("/anisearchmodel/anime", methods=["POST"])
@limiter.limit("1 per second")
def get_anime_similarities() -> Response:
    """
    Handle POST requests to find and return the top N most similar anime descriptions.

    Expects a JSON payload with 'model', 'description', 'page', and 'resultsPerPage' fields.

    Returns:
        Response: A JSON response with the top similar anime descriptions and the similarity scores.

    Raises:
        400 Bad Request: If the 'model' or 'description' fields are missing from the request.
    """
    try:
        clear_memory()
        data = request.json
        if data is None:
            raise ValueError("Request payload is missing or not in JSON format")
        validate_input(data)
        model_name = data.get("model")
        description = data.get("description")
        page = data.get("page", 1)
        results_per_page = data.get("resultsPerPage", 10)

        # Get the client's IP address
        client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
        logging.info(
            "Received anime request from IP: %s with model: %s, description: %s, page: %d, resultsPerPage: %d",
            client_ip,
            model_name,
            description,
            page,
            results_per_page,
        )

        results = get_similarities(
            model_name, description, "anime", page, results_per_page
        )
        logging.info("Returning %d anime results", len(results))
        clear_memory()
        return jsonify(results)

    except ValueError as e:
        logging.error("Validation error: %s", e)
        return make_response(jsonify({"error": "Bad Request"}), 400)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Internal server error: %s", e)
        return make_response(jsonify({"error": "Internal server error"}), 500)


@app.route("/anisearchmodel/manga", methods=["POST"])  # type: ignore
@limiter.limit("1 per second")
def get_manga_similarities() -> Response:
    """
    Handle POST requests to find and return the top N most similar manga descriptions.

    Expects a JSON payload with 'model', 'description', 'page', and 'resultsPerPage' fields.

    Returns:
        Response: A JSON response with the top similar manga descriptions and the similarity scores.

    Raises:
        400 Bad Request: If the 'model' or 'description' fields are missing from the request.
    """
    try:
        clear_memory()
        data = request.json
        if data is None:
            raise ValueError("Request payload is missing or not in JSON format")
        validate_input(data)
        model_name = data.get("model")
        description = data.get("description")
        page = data.get("page", 1)
        results_per_page = data.get("resultsPerPage", 10)

        # Get the client's IP address
        client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)

        logging.info(
            "Received manga request from IP: %s with model: %s, description: %s, page: %d, resultsPerPage: %d",
            client_ip,
            model_name,
            description,
            page,
            results_per_page,
        )

        results = get_similarities(
            model_name, description, "manga", page, results_per_page
        )
        logging.info("Returning %d manga results", len(results))
        clear_memory()
        return jsonify(results)

    except HTTPException as e:
        logging.error("HTTP error: %s", e)
        return make_response(jsonify({"error": e.description}), e.code)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Internal server error: %s", e)
        return make_response(jsonify({"error": "Internal server error"}), 500)


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() in ["true", "1"]
    app.run(debug=debug_mode, threaded=True, port=21493)
