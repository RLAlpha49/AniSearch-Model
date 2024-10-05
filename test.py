"""
This script tests a Sentence-BERT (SBERT) model by comparing a new description
against a merged anime dataset to find the most similar synopses.

It performs the following operations:
- Loads a pre-trained SBERT model specified by the user.
- Preprocesses a new description for comparison.
- Loads precomputed embeddings for various synopsis columns.
- Calculates cosine similarities between the new description and existing synopses.
- Outputs the top N most similar synopses.
"""

# pylint: disable=E0401, E0611, E1101
import os
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import common

# Disable oneDNN for TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from sentence_transformers import (  # pylint: disable=wrong-import-position, wrong-import-order  # noqa: E402
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

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Test SBERT model with a new description.")
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="The model name to use (e.g., 'all-mpnet-base-v1').",
)
args = parser.parse_args()

# New description
NEW_DESCRIPTION = (
    "The main character is a 37 year old man who is stabbed and dies, "
    "but is reborn as a slime in a different world."
)

# Preprocess the new description
processed_description = common.preprocess_text(NEW_DESCRIPTION)

# Load the SBERT model
model = SentenceTransformer(args.model)

# Define embedding dimension based on the model being tested
hf_model = SentenceTransformer(args.model)
EMBEDDING_DIM = hf_model.get_sentence_embedding_dimension()

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

# Encode the new description
NEW_POOLED_EMBEDDING = model.encode([processed_description])

# Initialize a dictionary to store cosine similarities for each synopsis column
cosine_similarities_dict = {}

# Calculate cosine similarity for each synopsis column
for col in synopsis_columns:
    # Load the embeddings for the current column
    embeddings_file = f"model/{args.model}/embeddings_{col.replace(' ', '_')}.npy"
    existing_embeddings = np.load(embeddings_file)

    # Ensure the dimensions match
    if existing_embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Incompatible dimension for embeddings in {col}: "
            f"expected {EMBEDDING_DIM}, got {existing_embeddings.shape[1]}"
        )

    # Calculate cosine similarity
    cosine_similarities = (
        util.pytorch_cos_sim(
            torch.tensor(NEW_POOLED_EMBEDDING), torch.tensor(existing_embeddings)
        )
        .flatten()
        .cpu()
        .numpy()
    )

    # Store the cosine similarities
    cosine_similarities_dict[col] = cosine_similarities

# Find and print the top N most similar descriptions across all columns
NUM_SIMILARITIES = 10
all_top_indices = []

for col, cosine_similarities in cosine_similarities_dict.items():
    # Find the indices of the top N most similar descriptions in descending order
    top_indices_unsorted = np.argsort(cosine_similarities)[-NUM_SIMILARITIES:]
    # Sort the top indices based on similarity scores in descending order
    top_indices = top_indices_unsorted[
        np.argsort(cosine_similarities[top_indices_unsorted])[::-1]
    ]
    all_top_indices.extend([(idx, col) for idx in top_indices])

# Sort all top indices by similarity score
all_top_indices.sort(key=lambda x: cosine_similarities_dict[x[1]][x[0]], reverse=True)

# Track seen anime names to avoid duplicates
seen_anime_names = set()

# Print the top similar descriptions, avoiding duplicates
print("Top similar descriptions:")
RANK = 1
for idx, col in all_top_indices:
    name = df.iloc[idx]["title"]
    if name not in seen_anime_names:
        synopsis = df.iloc[idx][col]
        similarity = cosine_similarities_dict[col][idx]
        print(f"\n{RANK}: {name} - {synopsis} (Similarity: {similarity})")
        seen_anime_names.add(name)
        RANK += 1
        if RANK > NUM_SIMILARITIES:
            break
