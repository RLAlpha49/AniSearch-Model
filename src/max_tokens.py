"""
This module calculates the maximum token count for different transformer models across
anime and manga datasets.

The module processes multiple synopsis columns from anime and manga datasets, tokenizing
the text using various transformer models to determine the maximum token length needed
for each model. This information is useful for setting appropriate maximum sequence
lengths when training or using these models.
"""

from typing import List, Dict
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


def calculate_max_tokens(
    dataset_path: str,
    synopsis_columns: List[str],
    model_names: List[str],
    batch_size: int = 64,
) -> Dict[str, int]:
    """
    Calculate the maximum token count for each model across specified synopsis columns in a dataset.

    Args:
        dataset_path (str): Path to the CSV dataset file.
        synopsis_columns (list): List of column names containing synopsis text to analyze.
        model_names (list): List of model names/paths to test for tokenization.
        batch_size (int, optional): Batch size for processing. Defaults to 64.

    Returns:
        dict: Dictionary mapping model names to their maximum token counts.
            Example: {'model-name': max_token_count}
    """
    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Dictionary to store the highest token count for each model
    model_max_token_counts = {}

    for model_name in model_names:
        # Initialize the tokenizer for the current model
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, model_max_length=100000, clean_up_tokenization_spaces=True
        )

        # Variable to store the maximum token count for the current model
        current_max_tokens = 0

        for column in synopsis_columns:
            if column not in df.columns:
                print(f"Column '{column}' not found in dataset. Skipping...")
                continue

            # Drop NaN values and convert to list
            synopses = df[column].dropna().tolist()

            # Process in batches
            for i in tqdm(
                range(0, len(synopses), batch_size),
                desc=f"Processing {model_name} - {column}",
            ):
                batch = synopses[i : i + batch_size]
                # Tokenize each text individually to avoid iteration issues
                for text in batch:
                    tokens = tokenizer(
                        text, add_special_tokens=True, max_length=100000
                    )["input_ids"]
                    tokens_count = len(tokens)  # type: ignore
                    # Update current_max_tokens if the current tokens_count is higher
                    if tokens_count > current_max_tokens:
                        current_max_tokens = tokens_count

        # Store the maximum token count for the current model
        model_max_token_counts[model_name] = current_max_tokens

    return model_max_token_counts


# List of models to test
model_list = [
    "toobi/anime",
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
]

# Columns to process for anime dataset
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

# Columns to process for manga dataset
manga_synopsis_columns = [
    "synopsis",
    "Synopsis jikan Dataset",
    "Synopsis data Dataset",
]

# Calculate max tokens for anime dataset
anime_max_tokens = calculate_max_tokens(
    "model/merged_anime_dataset.csv", anime_synopsis_columns, model_list
)
print("Anime Dataset:")
for current_model, token_count in anime_max_tokens.items():
    print(f"Highest token count for model '{current_model}': {token_count}")

# Calculate max tokens for manga dataset
manga_max_tokens = calculate_max_tokens(
    "model/merged_manga_dataset.csv", manga_synopsis_columns, model_list
)
print("\nManga Dataset:")
for current_model, token_count in manga_max_tokens.items():
    print(f"Highest token count for model '{current_model}': {token_count}")

# Find and print the overall maximum token count for anime and manga
max_tokens_anime = max(anime_max_tokens.values())
max_tokens_manga = max(manga_max_tokens.values())

print(f"\nOverall maximum token count for Anime Dataset: {max_tokens_anime}")
print(f"Overall maximum token count for Manga Dataset: {max_tokens_manga}")
