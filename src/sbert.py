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
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModel

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import common  # pylint: disable=wrong-import-position

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


# Function to get SBERT embeddings
def get_sbert_embeddings(
    dataframe, sbert_model, batch_size, column_name, model_name, device
):
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
def main():
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

    if not model_name.startswith("sentence-transformers/"):
        model_name = f"sentence-transformers/{model_name}"

    if device == "cuda":
        batch_size = 512
        if model_name in [
            "sentence-transformers/gtr-t5-xl",
            "sentence-transformers/sentence-t5-xl",
            "sentence-transformers/sentence-t5-xxl",
        ]:
            batch_size = 8
            # Limited by GPU memory, must not go past Dedicated GPU memory (Will Freeze/Slow Down).
            # Change as needed.
            if model_name == "sentence-transformers/sentence-t5-xxl":
                device = "cpu"
    else:
        batch_size = 128

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

    # Create directory for model-specific embeddings
    os.makedirs(embeddings_save_dir, exist_ok=True)

    # Load the merged dataset
    df = common.load_dataset(dataset_path)

    # Preprocess each synopsis or description column
    for col in synopsis_columns:
        df[f"Processed_{col}"] = df[col].fillna("").apply(common.preprocess_text)

    # Load the underlying Hugging Face model to access config
    hf_model = AutoModel.from_pretrained(model_name)

    # Initialize SBERT components
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=True,
    )

    # Load pre-trained SBERT model
    model = SentenceTransformer(
        model_name, device=device, modules=[word_embedding_model, pooling_model]
    )

    # Measure the time taken to generate embeddings for each column
    all_embeddings = {}
    start_time = time.time()
    for col in synopsis_columns:
        processed_col = f"Processed_{col}"
        embeddings = get_sbert_embeddings(
            df, model, batch_size, processed_col, model_name, device
        )
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

    # Calculate total number of embeddings
    total_num_embeddings = sum(
        emb.shape[0] for emb in all_embeddings.values() if emb.size > 0
    )

    # Save evaluation data
    common.save_evaluation_data(
        model_name=model_name,
        batch_size=batch_size,
        num_embeddings=total_num_embeddings,
        additional_info=additional_info,
    )


if __name__ == "__main__":
    main()
