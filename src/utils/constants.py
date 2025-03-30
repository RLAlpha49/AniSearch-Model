"""
Constants used throughout the anime search model application.
"""

# Default paths
ANIME_DATASET_PATH = "model/merged_anime_dataset.csv"
MANGA_DATASET_PATH = "model/merged_manga_dataset.csv"
MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"  # A good default cross-encoder model
)
NUM_RESULTS = 5  # Default number of results to return
DEFAULT_BATCH_SIZE = 256  # Default batch size for processing

# Alternative cross-encoder models that can be used
ALTERNATIVE_MODELS = {
    # MS Marco models - Text Ranking
    "ms_marco_models": {
        "ms-marco-MiniLM-L2-v2": "cross-encoder/ms-marco-MiniLM-L2-v2",
        "ms-marco-MiniLM-L4-v2": "cross-encoder/ms-marco-MiniLM-L4-v2",
        "ms-marco-MiniLM-L6-v2": "cross-encoder/ms-marco-MiniLM-L6-v2",  # Default model
        "ms-marco-MiniLM-L12-v2": "cross-encoder/ms-marco-MiniLM-L12-v2",
        "ms-marco-TinyBERT-L2": "cross-encoder/ms-marco-TinyBERT-L2",
        "ms-marco-TinyBERT-L2-v2": "cross-encoder/ms-marco-TinyBERT-L2-v2",
        "ms-marco-TinyBERT-L4": "cross-encoder/ms-marco-TinyBERT-L4",
        "ms-marco-TinyBERT-L6": "cross-encoder/ms-marco-TinyBERT-L6",
        "ms-marco-electra-base": "cross-encoder/ms-marco-electra-base",
    }
}
