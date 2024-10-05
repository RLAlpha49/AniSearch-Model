"""
This script generates Sentence-BERT (SBERT) embeddings for either an anime or manga dataset.

It performs the following operations:
- Loads a pre-trained SBERT model specified by the user.
- Preprocesses text data from various synopsis or description columns in the dataset.
- Generates embeddings for each synopsis or description using batched processing.
- Saves the generated embeddings to disk under separate directories for anime and manga.
- Records and saves evaluation data including model and hardware information.
"""

# pylint: disable=E0401, E0611
import os
import time
import warnings
import argparse
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModel
import common

# Disable oneDNN for TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from sentence_transformers import (  # pylint: disable=wrong-import-position, wrong-import-order  # noqa: E402
    SentenceTransformer,
    models,
)

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


# Parse command-line arguments
def parse_args():
    """
    Parses command-line arguments for the SBERT embedding generation script.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
        Specifically, it includes:
        - 'model': The name of the SBERT model to use.
        - 'type': The type of dataset ('anime' or 'manga') for which to generate embeddings.
    """
    parser = argparse.ArgumentParser(
        description="Generate SBERT embeddings for anime or manga dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The model name to use (e.g., 'all-mpnet-base-v1').",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["anime", "manga"],
        required=True,
        help="Type of dataset to generate embeddings for: 'anime' or 'manga'.",
    )
    return parser.parse_args()


args = parse_args()

# Determine device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Parameters
MODEL_NAME = args.model
DATASET_TYPE = args.type

if not MODEL_NAME.startswith("sentence-transformers/"):
    MODEL_NAME = f"sentence-transformers/{MODEL_NAME}"

if DEVICE == "cuda":
    BATCH_SIZE = 512
    if MODEL_NAME in [
        "sentence-transformers/gtr-t5-xl",
        "sentence-transformers/sentence-t5-xl",
        "sentence-transformers/sentence-t5-xxl"
    ]:
        BATCH_SIZE = 8
        # Limited by GPU memory, must not go past Dedicated GPU memory (Will Freeze/Slow Down).
        # Change as needed.
        if MODEL_NAME == "sentence-transformers/sentence-t5-xxl":
            DEVICE = "cpu"
else:
    BATCH_SIZE = 128

# Load the merged dataset based on type
if DATASET_TYPE == "anime":
    DATASET_PATH = "model/merged_anime_dataset.csv"
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
    embeddings_save_dir = f"model/anime/{MODEL_NAME.split('/')[-1]}"
    EVALUATION_FILE = "model/evaluation_results_anime.json"
elif DATASET_TYPE == "manga":
    DATASET_PATH = "model/merged_manga_dataset.csv"
    synopsis_columns = [
        "synopsis",
        "Synopsis jikan Dataset",
        "Synopsis data Dataset",
    ]
    embeddings_save_dir = f"model/manga/{MODEL_NAME.split('/')[-1]}"
    EVALUATION_FILE = "model/evaluation_results_manga.json"
else:
    raise ValueError("Invalid dataset type specified. Use 'anime' or 'manga'.")

# Create directory for model-specific embeddings
os.makedirs(embeddings_save_dir, exist_ok=True)

# Load the merged dataset
df = common.load_dataset(DATASET_PATH)

# Preprocess each synopsis or description column
for col in synopsis_columns:
    df[f"Processed_{col}"] = df[col].fillna("").apply(common.preprocess_text)

# Load the underlying Hugging Face model to access config
hf_model = AutoModel.from_pretrained(MODEL_NAME)

# Initialize SBERT components
word_embedding_model = models.Transformer(MODEL_NAME)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=True,
)

# Load pre-trained SBERT model
model = SentenceTransformer(
    MODEL_NAME, device=DEVICE, modules=[word_embedding_model, pooling_model]
)


# Function to get SBERT embeddings
def get_sbert_embeddings(dataframe, sbert_model, batch_size, column_name):
    """
    Generate SBERT embeddings for a given DataFrame column using batched processing.

    Args:
        dataframe (pandas.DataFrame): The input DataFrame containing the text data.
        sbert_model (SentenceTransformer): The SBERT model to use for generating embeddings.
        batch_size (int): The number of texts to process in each batch.
        column_name (str): The name of the DataFrame column containing the text data.

    Returns:
        numpy.ndarray: A 2D array of embeddings, each row corresponds to a text in input DataFrame.
    """
    embeddings_list = []
    for i in tqdm(
        range(0, len(dataframe), batch_size),
        desc=f"Generating Embeddings for {column_name}",
    ):
        batch_texts = dataframe[column_name].iloc[i : i + batch_size].tolist()
        if batch_texts:
            if (
                MODEL_NAME == "sentence-transformers/sentence-t5-xxl"
                and DEVICE == "cuda"
            ):
                # Use mixed precision for this specific model
                with torch.no_grad():
                    with torch.amp.autocast("cuda"):
                        batch_embeddings = sbert_model.encode(
                            batch_texts, convert_to_numpy=True, show_progress_bar=False
                        )
            else:
                # Standard encoding for other models
                with torch.no_grad():
                    batch_embeddings = sbert_model.encode(
                        batch_texts, convert_to_numpy=True, show_progress_bar=False
                    )
            embeddings_list.append(batch_embeddings)
    torch.cuda.empty_cache()
    if embeddings_list:
        return np.vstack(embeddings_list)
    return np.array([])


# Measure the time taken to generate embeddings for each column
all_embeddings = {}
start_time = time.time()
for col in synopsis_columns:
    processed_col = f"Processed_{col}"
    embeddings = get_sbert_embeddings(df, model, BATCH_SIZE, processed_col)
    all_embeddings[col] = embeddings
end_time = time.time()
embedding_generation_time = end_time - start_time

# Save the embeddings for each column
for col, embeddings in all_embeddings.items():
    if embeddings.size > 0:
        save_path = os.path.join(
            embeddings_save_dir, f"embeddings_{col.replace(' ', '_')}.npy"
        )
        np.save(save_path, embeddings)
    else:
        print(f"No embeddings generated for column: {col}")

# Prepare evaluation data
additional_info = {
    "dataset_info": {
        "num_samples": len(df),
        "preprocessing": "text normalization",
        "source": [DATASET_PATH],
    },
    "model_info": {
        "num_layers": hf_model.config.num_hidden_layers,
        "hidden_size": hf_model.config.hidden_size,
        "max_seq_length": (
            word_embedding_model.max_seq_length
            if hasattr(word_embedding_model, "max_seq_length")
            else None
        ),
    },
    "timing": {"embedding_generation_time": embedding_generation_time},
    "type": DATASET_TYPE,
    "device": DEVICE,
}

# Calculate total number of embeddings
total_num_embeddings = sum(
    emb.shape[0] for emb in all_embeddings.values() if emb.size > 0
)

# Save evaluation data
common.save_evaluation_data(
    model_name=MODEL_NAME,
    batch_size=BATCH_SIZE,
    num_embeddings=total_num_embeddings,
    additional_info=additional_info,
)
