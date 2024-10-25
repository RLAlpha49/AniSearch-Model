"""
This module implements a Flask application that provides API endpoints for finding
similar anime or manga descriptions.

The application uses Sentence Transformers and custom models to encode descriptions
and calculate cosine similarities. It supports multiple synopsis columns from
different datasets and returns paginated results of the most similar items.

Key Features:
- Supports multiple pre-trained and custom Sentence Transformer models
- Handles both anime and manga similarity searches
- Implements rate limiting and CORS
- Provides memory management for GPU resources
- Includes comprehensive logging
- Returns paginated results with similarity scores

The API endpoints are:
- POST /anisearchmodel/anime: Find similar anime based on description
- POST /anisearchmodel/manga: Find similar manga based on description
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
from src.custom_transformer import CustomT5EncoderModel

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
    Updates the last request time to the current time in a thread-safe manner.

    This function is used to track when the last API request was made, which helps
    with memory management and cleanup of unused resources.
    """
    with last_request_time_lock:
        global last_request_time
        last_request_time = time.time()


def clear_memory() -> None:
    """
    Frees up system memory and GPU cache.

    This function performs two cleanup operations:
    1. Empties the GPU cache if CUDA is being used
    2. Runs Python's garbage collector to free memory
    """
    torch.cuda.empty_cache()
    gc.collect()


def periodic_memory_clear() -> None:
    """
    Runs a background thread that periodically cleans up memory.

    The thread monitors the time since the last API request. If no requests have been
    made for over 300 seconds (5 minutes), it triggers memory cleanup to free resources.

    The function runs indefinitely until the application is shut down.
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
    "sentence-transformers/sentence-t5-xxl",
    "toobi/anime",
    "sentence-transformers/fine_tuned_sbert_anime_model",
    "fine_tuned_sbert_anime_model",
    "fine_tuned_sbert_model_anime",
]


def validate_input(data: Dict[str, Any]) -> None:
    """
    Validates the input data for API requests.

    This function checks that:
    1. Both model name and description are provided
    2. The description length is within acceptable limits
    3. The specified model is in the list of allowed models

    Args:
        data: Dictionary containing the request data with 'model' and 'description' keys

    Raises:
        HTTPException: If any validation check fails, with appropriate error message and status code
    """
    model_name = data.get("model")
    description = data.get("description")

    if not model_name or not description:
        logging.error("Model name or description missing in the request.")
        abort(400, description="Model name and description are required")

    if len(description) > 2000:
        logging.error("Description too long.")
        abort(400, description="Description is too long")

    if model_name not in allowed_models:
        logging.error("Invalid model name.")
        abort(400, description="Invalid model name")


def load_embeddings(model_name: str, col: str, dataset_type: str) -> np.ndarray:
    """
    Loads pre-computed embeddings for a specific model and dataset column.

    Args:
        model_name: Name of the model used to generate the embeddings
        col: Name of the synopsis column
        dataset_type: Type of dataset ('anime' or 'manga')

    Returns:
        NumPy array containing the pre-computed embeddings

    Raises:
        FileNotFoundError: If the embeddings file doesn't exist
    """
    embeddings_file = (
        f"model/{dataset_type}/{model_name}/embeddings_{col.replace(' ', '_')}.npy"
    )
    return np.load(embeddings_file)


def calculate_cosine_similarities(
    model: SentenceTransformer | CustomT5EncoderModel,
    model_name: str,
    new_embedding: np.ndarray,
    col: str,
    dataset_type: str,
) -> np.ndarray:
    """
    Calculates cosine similarities between a new embedding and existing embeddings.

    This function:
    1. Loads pre-computed embeddings for the specified column
    2. Verifies embedding dimensions match
    3. Computes cosine similarity scores using GPU if available

    Args:
        model: The transformer model used for encoding
        model_name: Name of the model
        new_embedding: Embedding vector of the input description
        col: Name of the synopsis column
        dataset_type: Type of dataset ('anime' or 'manga')

    Returns:
        Array of cosine similarity scores between the new embedding and all existing embeddings

    Raises:
        ValueError: If embedding dimensions don't match
    """
    model_name = model_name.replace("sentence-transformers/", "")
    model_name = model_name.replace("toobi/", "")
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
    Finds the top N most similar descriptions across all synopsis columns.

    This function:
    1. Processes similarity scores from all columns
    2. Sorts them in descending order
    3. Returns indices and column names for the top matches

    Args:
        cosine_similarities_dict: Dictionary mapping column names to arrays of similarity scores
        num_similarities: Number of top similarities to return (default: 10)

    Returns:
        List of tuples containing (index, column_name) for the top similar descriptions,
        sorted by similarity score in descending order
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
    Finds the most similar descriptions in the specified dataset.

    This function:
    1. Loads and validates the appropriate model
    2. Encodes the input description
    3. Calculates similarities with all stored descriptions
    4. Returns paginated results with metadata

    Args:
        model_name: Name of the model to use
        description: Input description to find similarities for
        dataset_type: Type of dataset ('anime' or 'manga')
        page: Page number for pagination (default: 1)
        results_per_page: Number of results per page (default: 10)

    Returns:
        List of dictionaries containing similar items with metadata and similarity scores

    Raises:
        ValueError: If model name is invalid or model loading fails
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

    if (
        model_name == "fine_tuned_sbert_anime_model"
        or model_name == "fine_tuned_sbert_model_anime"
    ):
        load_model_name = f"model/{model_name}"
    else:
        load_model_name = model_name

    # Load the complete SentenceTransformer model
    try:
        model = SentenceTransformer(load_model_name, device=device)
    except Exception as e:
        raise ValueError(f"Failed to load model '{load_model_name}': {e}") from e

    processed_description = description.strip()
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
        relevant_synopsis = df.iloc[idx][col]

        # Check if the relevant synopsis is valid
        if pd.isna(relevant_synopsis) or relevant_synopsis.strip() == "":
            continue

        if name not in seen_names:
            row_data = df.iloc[idx].to_dict()  # Convert the entire row to a dictionary
            # Keep only the relevant synopsis column
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
    API endpoint for finding similar anime based on a description.

    This endpoint:
    1. Validates the request payload
    2. Processes the description using the specified model
    3. Returns paginated results of similar anime

    Expected JSON payload:
        {
            "model": str,          # Name of the model to use
            "description": str,    # Input description to find similarities for
            "page": int,           # Optional: Page number (default: 1)
            "resultsPerPage": int  # Optional: Results per page (default: 10)
        }

    Returns:
        JSON response containing:
        - List of similar anime with metadata
        - Similarity scores
        - Pagination information

    Raises:
        400: If request validation fails
        500: If internal processing error occurs
    """
    try:
        clear_memory()
        data = request.json
        if data is None:
            raise ValueError("Request payload is missing or not in JSON format")
        validate_input(data)
        model_name = data.get("model")
        if model_name == "sentence-transformers/fine_tuned_sbert_anime_model":
            model_name = "fine_tuned_sbert_model_anime"
        description = data.get("description")
        page = data.get("page", 1)
        results_per_page = data.get("resultsPerPage", 10)

        # Get the client's IP address
        client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
        logging.info(
            "Received anime request from IP: %s with model: %s, "
            "description: %s, page: %d, resultsPerPage: %d",
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
    API endpoint for finding similar manga based on a description.

    This endpoint:
    1. Validates the request payload
    2. Processes the description using the specified model
    3. Returns paginated results of similar manga

    Expected JSON payload:
        {
            "model": str,          # Name of the model to use
            "description": str,    # Input description to find similarities for
            "page": int,           # Optional: Page number (default: 1)
            "resultsPerPage": int  # Optional: Results per page (default: 10)
        }

    Returns:
        JSON response containing:
        - List of similar manga with metadata
        - Similarity scores
        - Pagination information

    Raises:
        400: If request validation fails
        500: If internal processing error occurs
    """
    try:
        clear_memory()
        data = request.json
        if data is None:
            raise ValueError("Request payload is missing or not in JSON format")
        validate_input(data)
        model_name = data.get("model")
        if model_name == "sentence-transformers/fine_tuned_sbert_anime_model":
            model_name = "fine_tuned_sbert_model_anime"
        description = data.get("description")
        page = data.get("page", 1)
        results_per_page = data.get("resultsPerPage", 10)

        # Get the client's IP address
        client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)

        logging.info(
            "Manga request - IP: %s, model: %s, desc: %s, "
            "page: %d, results/page: %d",
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
