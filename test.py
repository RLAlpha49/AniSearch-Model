"""
This script tests a Sentence-BERT (SBERT) model by comparing a new description
against a merged anime or manga dataset to find the most similar synopses or descriptions.

It performs the following operations:
- Loads a pre-trained SBERT model specified by the user.
- Preprocesses a new description for comparison.
- Loads precomputed embeddings for various synopsis or description columns.
- Calculates cosine similarities between the new description and existing synopses/descriptions.
- Outputs the top N most similar synopses or descriptions.
"""

# pylint: disable=E0401, E0611, E1101
import os
import argparse
import warnings
from datetime import datetime
import json
import numpy as np
import torch
import common

# Disable oneDNN for TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from sentence_transformers import (  # pylint: disable=wrong-import-position, wrong-import-order  # noqa: E402
    SentenceTransformer,
    util,
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
    Parses command-line arguments for the SBERT model testing script.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
        Specifically, it includes:
        - 'model': The name of the SBERT model to use.
        - 'type': The type of dataset ('anime' or 'manga') to test against.
        - 'top_n': The number of top similar descriptions to retrieve (default is 10).
    """
    parser = argparse.ArgumentParser(
        description="Test SBERT model with a new description."
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
        help="Type of dataset to test against: 'anime' or 'manga'.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Number of top similar descriptions to retrieve.",
    )
    return parser.parse_args()


args = parse_args()

# New description to compare
NEW_DESCRIPTION = (
    "The main character is a 37 year old man who is stabbed and dies, "
    "but is reborn as a slime in a different world."
)

# Determine device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Parameters
MODEL_NAME = args.model
DATASET_TYPE = args.type
TOP_N = args.top_n

if not MODEL_NAME.startswith("sentence-transformers/"):
    MODEL_NAME = f"sentence-transformers/{MODEL_NAME}"

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

# Load the merged dataset
df = common.load_dataset(DATASET_PATH)

# Preprocess the new description
processed_description = common.preprocess_text(NEW_DESCRIPTION)

# Load the SBERT model
model = SentenceTransformer(MODEL_NAME, device=DEVICE)

# Define embedding dimension based on the model being tested
hf_model = SentenceTransformer(MODEL_NAME)
EMBEDDING_DIM = hf_model.get_sentence_embedding_dimension()

# Encode the new description
NEW_POOLED_EMBEDDING = model.encode(
    [processed_description], convert_to_tensor=True, device=DEVICE
)

# Initialize a dictionary to store cosine similarities for each synopsis column
cosine_similarities_dict = {}

# Calculate cosine similarity for each synopsis column
for col in synopsis_columns:
    # Load the embeddings for the current column
    embeddings_file = os.path.join(
        embeddings_save_dir, f"embeddings_{col.replace(' ', '_')}.npy"
    )
    if not os.path.exists(embeddings_file):
        print(f"Embeddings file not found for column '{col}': {embeddings_file}")
        continue

    existing_embeddings = np.load(embeddings_file)

    # Ensure the dimensions match
    if existing_embeddings.shape[1] != EMBEDDING_DIM:
        print(
            f"Skipping column '{col}' due to incompatible embedding dimensions: "
            f"expected {EMBEDDING_DIM}, got {existing_embeddings.shape[1]}"
        )
        continue

    # Convert existing embeddings to tensor
    existing_embeddings_tensor = torch.tensor(existing_embeddings).to(DEVICE)

    # Calculate cosine similarity
    with torch.no_grad():
        cosine_similarities = (
            util.pytorch_cos_sim(NEW_POOLED_EMBEDDING, existing_embeddings_tensor)
            .squeeze(0)
            .cpu()
            .numpy()
        )

    # Store the cosine similarities
    cosine_similarities_dict[col] = cosine_similarities

# Check if any similarities were calculated
if not cosine_similarities_dict:
    raise ValueError(
        "No valid embeddings were loaded. Please check your embeddings directory and files."
    )

# Find and print the top N most similar descriptions across all columns
all_top_indices = []
for col, cosine_similarities in cosine_similarities_dict.items():
    if len(cosine_similarities) < TOP_N:
        print(
            f"Warning: Column '{col}' has fewer entries "
            f"({len(cosine_similarities)}) than TOP_N ({TOP_N})."
        )
    # Find the indices of the top N most similar descriptions in descending order
    top_indices_unsorted = np.argsort(cosine_similarities)[-TOP_N:]
    # Sort the top indices based on similarity scores in descending order
    top_indices = top_indices_unsorted[
        np.argsort(cosine_similarities[top_indices_unsorted])[::-1]
    ]
    all_top_indices.extend([(idx, col) for idx in top_indices])

# Sort all top indices by similarity score
all_top_indices.sort(key=lambda x: cosine_similarities_dict[x[1]][x[0]], reverse=True)

# Track seen anime/manga names to avoid duplicates
seen_names = set()

# Print the top similar descriptions, avoiding duplicates
print(
    f"\nTop {TOP_N} similar {'synopses' if DATASET_TYPE == 'anime' else 'descriptions'}:"
)
RANK = 1
for idx, col in all_top_indices:
    name = df.iloc[idx]["title"]
    if name not in seen_names:
        synopsis = df.iloc[idx][col]
        similarity = cosine_similarities_dict[col][idx]
        print(f"\n{RANK}: {name} - {synopsis} (Similarity: {similarity:.4f})")
        seen_names.add(name)
        RANK += 1
        if RANK > TOP_N:
            break

# Prepare the top results
top_results = []
for idx, col in all_top_indices:
    if len(top_results) >= TOP_N:
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

# Load existing evaluation data
if os.path.exists(EVALUATION_FILE):
    with open(EVALUATION_FILE, "r", encoding="utf-8") as f:
        try:
            evaluation_data = json.load(f)
        except json.JSONDecodeError:
            evaluation_data = []
else:
    evaluation_data = []

# Append the new test result
test_result = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_name": MODEL_NAME,
    "dataset_type": DATASET_TYPE,
    "new_description": NEW_DESCRIPTION,
    "top_similarities": top_results,
}

evaluation_data.append(test_result)

# Save the updated evaluation data
with open(EVALUATION_FILE, "w", encoding="utf-8") as f:
    json.dump(evaluation_data, f, indent=4)

print(f"\nTest results saved to {EVALUATION_FILE}")
