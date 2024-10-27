"""
This module contains tests for the sbert.py script, which generates embeddings
for anime and manga datasets using the Sentence-BERT model.

The tests verify:
    - Successful execution of the script with valid command-line arguments
    - Creation and validation of embedding files for both anime and manga datasets
    - Proper saving and structure of evaluation results
    - Correct dimensionality of generated embeddings
    - Consistency between model parameters and evaluation data
"""

import subprocess
import sys
import os
import json
from typing import List
import numpy as np
import pytest


def run_sbert_command_and_verify(
    model_name: str, dataset_type: str, expected_files: List[str]
) -> None:
    """
    Run the SBERT command-line script and verify the generated embeddings and
    evaluation results.

    This function:
        1. Executes the SBERT script with specified parameters
        2. Verifies script execution success
        3. Checks for creation of expected embedding files
        4. Validates embedding dimensions
        5. Verifies evaluation results structure and content

    Args:
        model_name (str): The name of the model to be used (e.g.,
            'sentence-transformers/all-mpnet-base-v2')
        dataset_type (str): The type of dataset ('anime' or 'manga')
        expected_files (List[str]): List of expected embedding file names to be
            generated

    Raises:
        AssertionError: If any of the following conditions are not met:
            - Script execution fails
            - Expected embedding files are not created
            - Embeddings have invalid dimensions
            - Evaluation results are missing or malformed
            - Model parameters don't match input parameters
    """
    command = [
        sys.executable,
        "-m",
        "src.sbert",
        "--model",
        model_name,
        "--type",
        dataset_type,
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"

    embeddings_dir = (
        f"model/{dataset_type}/{model_name.replace('sentence-transformers/', '')}"
    )

    for file_name in expected_files:
        file_path = os.path.join(embeddings_dir, file_name)
        assert os.path.exists(
            file_path
        ), f"Embeddings file was not created at {file_path}."

        embeddings = np.load(file_path)
        assert embeddings.shape[1] > 0, "Embeddings should have a non-zero dimension."

    evaluation_results_path = os.path.join("model/", "evaluation_results.json")
    assert os.path.exists(
        evaluation_results_path
    ), f"Evaluation results were not saved at {evaluation_results_path}."

    with open(evaluation_results_path, "r", encoding="utf-8") as f:
        evaluation_data = json.load(f)

    if isinstance(evaluation_data, list):
        evaluation_data = evaluation_data[-1]

    assert (
        evaluation_data["model_parameters"]["model_name"] == model_name
    ), "Model name mismatch in evaluation data."
    assert (
        evaluation_data["type"] == dataset_type
    ), "Dataset type mismatch in evaluation data."


@pytest.mark.parametrize(
    "dataset_type, expected_files",
    [
        (
            "anime",
            [
                "embeddings_synopsis.npy",
                "embeddings_Synopsis_anime_270_Dataset.npy",
                "embeddings_Synopsis_Anime_data_Dataset.npy",
                "embeddings_Synopsis_anime_dataset_2023.npy",
                "embeddings_Synopsis_anime2_Dataset.npy",
                "embeddings_Synopsis_Anime-2022_Dataset.npy",
                "embeddings_Synopsis_anime4500_Dataset.npy",
                "embeddings_Synopsis_animes_dataset.npy",
                "embeddings_Synopsis_mal_anime_Dataset.npy",
                "embeddings_Synopsis_wykonos_Dataset.npy",
            ],
        ),
        (
            "manga",
            [
                "embeddings_synopsis.npy",
                "embeddings_Synopsis_data_Dataset.npy",
                "embeddings_Synopsis_jikan_Dataset.npy",
            ],
        ),
    ],
)
@pytest.mark.order(10)
def test_run_sbert_command_line(
    model_name: str, dataset_type: str, expected_files: List[str]
) -> None:
    """
    Test the SBERT command line script by running it with different dataset types
    and verifying the outputs.

    This test:
        1. Tests both anime and manga datasets
        2. Verifies generation of dataset-specific embedding files
        3. Validates embedding file structure and content
        4. Checks evaluation results for each dataset type

    Args:
        model_name (str): The name of the model to be tested, provided by pytest fixture
        dataset_type (str): The type of dataset being tested ('anime' or 'manga')
        expected_files (List[str]): List of expected embedding files for the dataset type

    The test is parameterized to run separately for anime and manga datasets,
    with different expected output files for each type.
    """
    run_sbert_command_and_verify(model_name, dataset_type, expected_files)
