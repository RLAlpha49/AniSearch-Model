"""
This module provides utility functions for loading datasets, preprocessing text,
and saving evaluation data for machine learning models.

Functions:
    load_dataset: Load and preprocess a dataset from a CSV file.
    preprocess_text: Clean and normalize text data for ML processing.
    save_evaluation_data: Save model evaluation results to JSON.
"""

# pylint: disable=E0401, E0611
import os
import re
import json
from datetime import datetime
from typing import Optional, Dict, Any
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import contractions
from unidecode import unidecode

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# Load the dataset
def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file and fill missing values in the 'Synopsis' column.

    Args:
        file_path (str): Path to the CSV file containing the dataset.

    Returns:
        pd.DataFrame: Loaded dataset with filled 'Synopsis' column.
    """
    df = pd.read_csv(file_path)
    df["synopsis"] = df["synopsis"].fillna("")
    return df


# Basic text preprocessing
def preprocess_text(text: Any) -> Any:
    """
    Preprocess text data by applying various cleaning and normalization steps.

    Steps include:
        - Converting to lowercase
        - Expanding contractions
        - Removing accents
        - Removing extra whitespace
        - Removing URLs
        - Removing source citations
        - Removing stopwords
        - Lemmatizing words

    Args:
        text (Any): Input text to preprocess. Can be string or other type.

    Returns:
        Any: Preprocessed text if input was string, otherwise returns input unchanged.
    """
    if text is None:
        return ""

    try:
        if isinstance(text, str):
            text = text.strip()  # Strip whitespace
            text = contractions.fix(text)  # Expand contractions
            text = unidecode(text)  # Remove accents
            text = re.sub(
                r"\s+", " ", text
            )  # Replace multiple spaces with a single space
            # Remove wrapping quotes
            if (text.startswith('"') and text.endswith('"')) or (
                text.startswith("'") and text.endswith("'")
            ):
                text = text[1:-1]
            text = re.sub(
                r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE
            )  # Remove URLs
            # Remove specific patterns
            text = re.sub(r"\[Written by .*?\].*$", "", text, flags=re.IGNORECASE)
            text = re.sub(
                r"<br><br>\s*\(source:.*?\).*$", "", text, flags=re.IGNORECASE
            )
            text = re.sub(r"\(source:.*?\).*$", "", text, flags=re.IGNORECASE)
            # Tokenize and remove stopwords
            words = text.split()
            words = [word for word in words if word not in stop_words]
            # Apply lemmatization
            words = [lemmatizer.lemmatize(word) for word in words]
            text = " ".join(words)
        else:
            return text
    except Exception:  # pylint: disable=broad-except
        return text

    return text


# Save evaluation data
def save_evaluation_data(
    model_name: str,
    batch_size: int,
    num_embeddings: int,
    additional_info: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save model evaluation data to a JSON file with timestamp and parameters.

    Creates or appends to 'model/evaluation_results.json', storing evaluation metrics
    and model configuration details.

    Args:
        model_name (str): Name/identifier of the model being evaluated.
        batch_size (int): Batch size used for generating embeddings.
        num_embeddings (int): Total number of embeddings generated.
        additional_info (Optional[Dict[str, Any]]): Additional evaluation metrics or parameters.
    """
    evaluation_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_parameters": {
            "model_name": model_name,
            "batch_size": batch_size,
            "num_embeddings": num_embeddings,
        },
    }

    if additional_info:
        evaluation_data.update(additional_info)

    # Path to the JSON file
    file_path = "model/evaluation_results.json"

    # Check if the file exists and is not empty
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        # Read the existing data
        with open(file_path, "r+", encoding="utf-8") as f:
            f.seek(0, os.SEEK_END)
            f.seek(f.tell() - 1, os.SEEK_SET)
            f.truncate()
            f.write(",\n")
            json.dump(evaluation_data, f, indent=4)
            f.write("\n]")
    else:
        # Create a new file with an array
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump([evaluation_data], f, indent=4)
