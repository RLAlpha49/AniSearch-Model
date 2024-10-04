"""
This module provides utility functions for loading datasets, preprocessing text,
and saving evaluation data for machine learning models.
"""

# pylint: disable=E0401, E0611
import re
import json
from datetime import datetime
import platform
import pandas as pd


# Load the dataset
def load_dataset(file_path):
    """
    Load dataset from a CSV file and fill missing values in the 'Synopsis' column.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded dataset with filled 'Synopsis' column.
    """
    df = pd.read_csv(file_path)
    df["Synopsis"] = df["Synopsis"].fillna("")
    return df


# Basic text preprocessing
def preprocess_text(text):
    """
    Preprocess the input text by converting it to lowercase and removing extra spaces.

    Args:
        text (str): The input text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove punctuation
    return text


# Save evaluation data
def save_evaluation_data(model_name, batch_size, num_embeddings, additional_info=None):
    """
    Save evaluation data including timestamp and model parameters to a JSON file.

    Args:
        model_name (str): The name of the model.
        batch_size (int): The batch size used for generating embeddings.
        num_embeddings (int): Number of embeddings generated.
        additional_info (dict, optional): Additional information to save.
    """
    evaluation_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_parameters": {
            "model_name": model_name,
            "batch_size": batch_size,
            "num_embeddings": num_embeddings,
        },
        "system_info": {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "processor": platform.processor(),
        },
    }

    if additional_info:
        evaluation_data.update(additional_info)

    with open("model/evaluation_results.json", "a", encoding="utf-8") as f:
        f.write(",\n")
        json.dump(evaluation_data, f, indent=4)
