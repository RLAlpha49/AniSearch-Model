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
import sys
import os
import time
import warnings
import argparse
from typing import Dict, Any
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModel

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import common  # pylint: disable=wrong-import-position


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

from sentence_transformers import (  # pylint: disable=wrong-import-position, wrong-import-order  # noqa: E402
    SentenceTransformer,
    models,
)


# Parse command-line arguments
def parse_args() -> argparse.Namespace:
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
        help="The model name to use (e.g., 'all-mpnet-base-v1' or path to fine-tuned model).",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["anime", "manga"],
        required=True,
        help="Type of dataset to generate embeddings for: 'anime' or 'manga'.",
    )
    return parser.parse_args()


# Function to get SBERT embeddings
def get_sbert_embeddings(
    dataframe: pd.DataFrame,
    sbert_model: SentenceTransformer,
    batch_size: int,
    column_name: str,
    model_name: str,
    device: str,
) -> np.ndarray:
    """
    Generate SBERT embeddings for a given DataFrame column using batched processing.

    Args:
        dataframe (pandas.DataFrame): The input DataFrame containing the text data.
        sbert_model (SentenceTransformer): The SBERT model to use for generating embeddings.
        batch_size (int): The number of texts to process in each batch.
        column_name (str): The name of the DataFrame column containing the text data.
        model_name (str): The name of the model being used.
        device (str): The device to use for computation ('cpu' or 'cuda').

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
                model_name == "sentence-transformers/sentence-t5-xxl"
                and device == "cuda"
            ):
                # Use mixed precision for this specific model
                with torch.no_grad():
                    with torch.amp.autocast("cuda"):  # type: ignore
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


# Run by test_
def main() -> None:
    """
    Main function to execute the embedding generation process.
    """
    args = parse_args()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Parameters
    model_name = args.model
    dataset_type = args.type

    # Load the merged dataset based on type
    if dataset_type == "anime":
        dataset_path = "model/merged_anime_dataset.csv"
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
        embeddings_save_dir = f"model/anime/{model_name.split('/')[-1]}"
    elif dataset_type == "manga":
        dataset_path = "model/merged_manga_dataset.csv"
        synopsis_columns = [
            "synopsis",
            "Synopsis jikan Dataset",
            "Synopsis data Dataset",
        ]
        embeddings_save_dir = f"model/manga/{model_name.split('/')[-1]}"
    else:
        raise ValueError("Invalid dataset type specified. Use 'anime' or 'manga'.")

    # Load the merged dataset
    df = common.load_dataset(dataset_path)

    # Preprocess each synopsis or description column
    for col in synopsis_columns:
        df[f"Processed_{col}"] = df[col].fillna("").apply(common.preprocess_text)

    if device == "cuda":
        batch_size = 448
        if model_name in [
            "sentence-transformers/gtr-t5-xl",
            "sentence-transformers/sentence-t5-xl",
            "sentence-transformers/sentence-t5-xxl",
        ]:
            batch_size = 8
            # Limited by GPU memory, must not go past Dedicated GPU memory (Will Freeze/Slow Down).
            # Change as needed.
            if model_name == "sentence-transformers/sentence-t5-xxl":
                batch_size = 1
                device = "cpu"
    else:
        batch_size = 128
        if model_name == "sentence-transformers/sentence-t5-xxl":
            batch_size = 1
            device = "cpu"

    # Create directory for model-specific embeddings
    os.makedirs(embeddings_save_dir, exist_ok=True)

    # Load the underlying Hugging Face model to access config
    if (
        model_name == "fine_tuned_sbert_model_anime"
        or model_name == "fine_tuned_sbert_model_manga"
    ):
        model_name = f"model/{model_name}"
    hf_model = AutoModel.from_pretrained(model_name)

    # Check if the model is a path to a fine-tuned model
    if os.path.exists(model_name):
        # Load the fine-tuned model from the specified directory
        model = SentenceTransformer(model_name, device=device)
    else:
        # Load a pre-trained model from Hugging Face
        if not model_name.startswith("sentence-transformers/"):
            if model_name != "toobi/anime":
                model_name = f"sentence-transformers/{model_name}"

    # Define the maximum token counts for each model for both anime and manga
    max_token_counts = {
        "toobi/anime": {"anime": 733, "manga": 673},
        "sentence-transformers/all-distilroberta-v1": {"anime": 704, "manga": 654},
        "sentence-transformers/all-MiniLM-L6-v1": {"anime": 733, "manga": 673},
        "sentence-transformers/all-MiniLM-L12-v1": {"anime": 733, "manga": 673},
        "sentence-transformers/all-MiniLM-L6-v2": {"anime": 733, "manga": 673},
        "sentence-transformers/all-MiniLM-L12-v2": {"anime": 733, "manga": 673},
        "sentence-transformers/all-mpnet-base-v1": {"anime": 733, "manga": 673},
        "sentence-transformers/all-mpnet-base-v2": {"anime": 733, "manga": 673},
        "sentence-transformers/all-roberta-large-v1": {"anime": 704, "manga": 654},
        "sentence-transformers/gtr-t5-base": {"anime": 843, "manga": 765},
        "sentence-transformers/gtr-t5-large": {"anime": 843, "manga": 765},
        "sentence-transformers/gtr-t5-xl": {"anime": 843, "manga": 765},
        "sentence-transformers/multi-qa-distilbert-dot-v1": {
            "anime": 733,
            "manga": 673,
        },
        "sentence-transformers/multi-qa-mpnet-base-cos-v1": {
            "anime": 733,
            "manga": 673,
        },
        "sentence-transformers/multi-qa-mpnet-base-dot-v1": {
            "anime": 733,
            "manga": 673,
        },
        "sentence-transformers/paraphrase-distilroberta-base-v2": {
            "anime": 704,
            "manga": 654,
        },
        "sentence-transformers/paraphrase-mpnet-base-v2": {
            "anime": 733,
            "manga": 673,
        },
        "sentence-transformers/sentence-t5-base": {"anime": 843, "manga": 765},
        "sentence-transformers/sentence-t5-large": {"anime": 843, "manga": 765},
        "sentence-transformers/sentence-t5-xl": {"anime": 843, "manga": 765},
        "sentence-transformers/sentence-t5-xxl": {"anime": 843, "manga": 765},
        "model/fine_tuned_sbert_model_anime": {"anime": 843, "manga": 765},
        "model/fine_tuned_sbert_model_manga": {"anime": 843, "manga": 765},
    }

    # Initialize SBERT components
    word_embedding_model = models.Transformer(model_name)
    # Set the max_seq_length for the current model based on the context
    word_embedding_model.max_seq_length = max_token_counts.get(model_name, {}).get(
        dataset_type, None
    )  # Default to None if not found

    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
    )

    # Load pre-trained SBERT model
    model = SentenceTransformer(
        model_name,
        device=device,
        modules=[word_embedding_model, pooling_model],
    )

    model[0].max_seq_length = max_token_counts.get(model_name, {}).get(  # type: ignore
        dataset_type, 512
    )
    model[
        1
    ].word_embedding_dimension = word_embedding_model.get_word_embedding_dimension()  # type: ignore

    print(model)

    # Measure the time taken to generate embeddings for each column
    start_time = time.time()
    total_num_embeddings = 0
    for col in synopsis_columns:
        processed_col = f"Processed_{col}"
        embeddings = get_sbert_embeddings(
            df, model, batch_size, processed_col, model_name, device
        )

        # Save the embeddings for the current column
        if embeddings.size > 0:
            save_path = os.path.join(
                embeddings_save_dir, f"embeddings_{col.replace(' ', '_')}.npy"
            )
            np.save(save_path, embeddings)
            total_num_embeddings += embeddings.shape[0]

            # Clear memory
            del embeddings
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print(f"No embeddings generated for column: {col}")

    end_time = time.time()
    embedding_generation_time = end_time - start_time

    # Prepare evaluation data
    additional_info: Dict[str, Any] = {
        "dataset_info": {
            "num_samples": len(df),
            "preprocessing": "text normalization",
            "source": [dataset_path],
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
        "type": dataset_type,
        "device": device,
    }

    # Save evaluation data
    common.save_evaluation_data(
        model_name=model_name,
        batch_size=batch_size,
        num_embeddings=total_num_embeddings,
        additional_info=additional_info,
    )


if __name__ == "__main__":
    main()
