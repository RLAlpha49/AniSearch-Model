"""
This script generates Sentence-BERT (SBERT) embeddings for an anime dataset.

It performs the following operations:
- Loads a pre-trained SBERT model specified by the user.
- Preprocesses text data from various synopsis columns in the dataset.
- Generates embeddings for each synopsis using batched processing.
- Saves the generated embeddings to disk for later use.
- Records and saves evaluation data including model and hardware information.
"""

# pylint: disable=E0401, E0611
import os
import time
import platform
import warnings
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer, models
from tqdm import tqdm
import torch
from transformers import AutoModel
import common

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers.tokenization_utils_base"
)

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Generate SBERT embeddings for anime dataset."
)
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="The model name to use (e.g., 'all-mpnet-base-v1').",
)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Parameters
MODEL_NAME = args.model
if not MODEL_NAME.startswith("sentence-transformers/"):
    MODEL_NAME = f"sentence-transformers/{MODEL_NAME}"

if DEVICE == "cuda":
    BATCH_SIZE = 256
    if MODEL_NAME == "sentence-transformers/gtr-t5-xl":
        BATCH_SIZE = 12
        # Limited by GPU memory, must not go past Dedicated GPU memory (Will Freeze/Slow Down).
        # Change as needed.
else:
    BATCH_SIZE = 128

# Load the merged dataset
df = common.load_dataset("model/merged_anime_dataset.csv")

# List of synopsis columns to process individually
synopsis_columns = [
    "Synopsis",
    "Synopsis animes dataset",
    "Synopsis anime_270 Dataset",
    "Synopsis Anime-2022 Dataset",
    "Synopsis Anime Dataset",
    "Synopsis anime4500 Dataset",
    "Synopsis anime-20220927-raw Dataset",
    "Synopsis wykonos Dataset",
]

# Preprocess each synopsis column
for col in synopsis_columns:
    df[f"Processed_{col}"] = df[col].fillna("").apply(common.preprocess_text)

# Load the underlying Hugging Face model to access config
hf_model = AutoModel.from_pretrained(MODEL_NAME)

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


# Get SBERT embeddings for each synopsis column
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
        batch_embeddings = sbert_model.encode(batch_texts, convert_to_numpy=True)
        embeddings_list.append(batch_embeddings)
    return np.vstack(embeddings_list)


# Measure the time taken to generate embeddings for each column
all_embeddings = {}
start_time = time.time()
for col in synopsis_columns:
    processed_col = f"Processed_{col}"
    embeddings = get_sbert_embeddings(df, model, BATCH_SIZE, processed_col)
    all_embeddings[col] = embeddings
end_time = time.time()
embedding_generation_time = end_time - start_time

# Create directory for model-specific embeddings
model_dir = f"model/{MODEL_NAME.split('/')[-1]}"
os.makedirs(model_dir, exist_ok=True)

# Save the embeddings for each column
for col, embeddings in all_embeddings.items():
    np.save(f"{model_dir}/embeddings_{col.replace(' ', '_')}.npy", embeddings)

# Save evaluation data
additional_info = {
    "dataset_info": {
        "num_samples": len(df),
        "preprocessing": "text normalization",
        "source": ["model/merged_anime_dataset.csv"],
    },
    "model_info": {
        "num_layers": hf_model.config.num_hidden_layers,
        "hidden_size": hf_model.config.hidden_size,
        "max_seq_length": word_embedding_model.max_seq_length,
    },
    "hardware_info": {
        "device": DEVICE,
        "gpu_model": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "N/A",
        "gpu_memory": (
            torch.cuda.get_device_properties(0).total_memory
            if DEVICE == "cuda"
            else "N/A"
        ),
    },
    "timing": {"embedding_generation_time": embedding_generation_time},
    "environment": {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": SentenceTransformer._version,  # pylint: disable=W0212
    },
}

common.save_evaluation_data(
    model_name=MODEL_NAME,
    batch_size=BATCH_SIZE,
    num_embeddings=sum(len(emb) for emb in all_embeddings.values()),
    additional_info=additional_info,
)
