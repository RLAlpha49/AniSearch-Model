"""
This module contains tests for evaluating the performance of models on anime and manga datasets.
It includes fixtures and test functions to ensure that the models are correctly loaded,
similarities are calculated, and evaluation results are saved properly.

The tests verify:
- Model and embedding loading functionality
- Similarity calculation between new descriptions and existing content
- Proper saving and structure of evaluation results
- Consistent behavior across both anime and manga datasets
"""

import os
import json
from typing import List, Dict
import pytest
from src.test import (
    load_model_and_embeddings,
    calculate_similarities,
    save_evaluation_results,
)


@pytest.fixture
def new_description() -> str:
    """
    Fixture that provides a new description for testing similarity calculations.

    The description represents a common isekai anime/manga plot to test against the datasets.

    Returns:
        str: A test description about a character being reborn in another world as a slime.
    """
    return (
        "The main character is a 37 year old man who is stabbed and dies, "
        "but is reborn as a slime in a different world."
    )


@pytest.mark.order(11)
def test_anime_model(new_description: str, model_name: str) -> None:  # pylint: disable=redefined-outer-name
    """
    Test the anime model's ability to find similar content based on description.

    This test verifies:
    1. Proper loading of the model and anime embeddings
    2. Accurate calculation of similarities between new description and existing anime
    3. Correct structure and saving of evaluation results
    4. Expected number and format of top similar results

    Args:
        new_description (str): A test description to compare against the anime database.
        model_name (str): The identifier of the model being tested.

    Raises:
        AssertionError: If any of the test conditions fail, including file existence,
                       data structure, or expected result format.
    """
    dataset_type = "anime"
    top_n = 5

    model, df, synopsis_columns, embeddings_save_dir = load_model_and_embeddings(
        model_name, dataset_type
    )
    top_results: List[Dict[str, float]] = calculate_similarities(
        model, df, synopsis_columns, embeddings_save_dir, new_description, top_n
    )

    assert len(top_results) == top_n
    for result in top_results:
        assert "title" in result
        assert "synopsis" in result
        assert "similarity" in result

    evaluation_results = save_evaluation_results(
        "./model/evaluation_results_anime.json",
        model_name,
        dataset_type,
        new_description,
        top_results,
    )
    assert os.path.exists(evaluation_results)
    with open(evaluation_results, "r", encoding="utf-8") as f:
        evaluation_data = json.load(f)
    assert len(evaluation_data) > 0
    assert isinstance(evaluation_data, list), "evaluation_results should be a list"
    assert isinstance(
        evaluation_data[-1], dict
    ), "Last item in evaluation_results should be a dictionary"
    assert "model_name" in evaluation_data[-1]
    assert "dataset_type" in evaluation_data[-1]
    assert "new_description" in evaluation_data[-1]
    assert len(evaluation_data[-1]["top_similarities"]) == top_n


@pytest.mark.order(12)
def test_manga_model(new_description: str, model_name: str) -> None:  # pylint: disable=redefined-outer-name
    """
    Test the manga model's ability to find similar content based on description.

    This test verifies:
    1. Proper loading of the model and manga embeddings
    2. Accurate calculation of similarities between new description and existing manga
    3. Correct structure and saving of evaluation results
    4. Expected number and format of top similar results

    Args:
        new_description (str): A test description to compare against the manga database.
        model_name (str): The identifier of the model being tested.

    Raises:
        AssertionError: If any of the test conditions fail, including file existence,
                       data structure, or expected result format.
    """
    dataset_type = "manga"
    top_n = 5

    model, df, synopsis_columns, embeddings_save_dir = load_model_and_embeddings(
        model_name, dataset_type
    )
    top_results: List[Dict[str, float]] = calculate_similarities(
        model, df, synopsis_columns, embeddings_save_dir, new_description, top_n
    )

    assert len(top_results) == top_n
    for result in top_results:
        assert "title" in result
        assert "synopsis" in result
        assert "similarity" in result

    evaluation_results = save_evaluation_results(
        "./model/evaluation_results_manga.json",
        model_name,
        dataset_type,
        new_description,
        top_results,
    )
    assert os.path.exists(evaluation_results)
    with open(evaluation_results, "r", encoding="utf-8") as f:
        evaluation_data = json.load(f)
    assert len(evaluation_data) > 0
    assert isinstance(evaluation_data, list), "evaluation_results should be a list"
    assert isinstance(
        evaluation_data[-1], dict
    ), "Last item in evaluation_results should be a dictionary"
    assert "model_name" in evaluation_data[-1]
    assert "dataset_type" in evaluation_data[-1]
    assert "new_description" in evaluation_data[-1]
    assert len(evaluation_data[-1]["top_similarities"]) == top_n
