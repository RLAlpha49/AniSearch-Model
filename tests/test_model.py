"""
This module contains tests for evaluating the performance of models on anime and manga datasets.
It includes fixtures and test functions to ensure that the models are correctly loaded,
similarities are calculated, and evaluation results are saved properly.
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
    Fixture that provides a new description for testing.

    Returns:
        str: The new description to compare against existing embeddings.
    """
    return (
        "The main character is a 37 year old man who is stabbed and dies, "
        "but is reborn as a slime in a different world."
    )


@pytest.mark.order(11)
def test_anime_model(new_description: str, model_name: str) -> None:  # pylint: disable=redefined-outer-name
    """
    Test the anime model by loading the model and embeddings, calculating similarities,
    and saving the evaluation results. The test ensures that the top results have the
    expected structure and that the evaluation results are saved correctly.

    Args:
        new_description (str): The new description to compare against existing embeddings.
        model_name (str): The name of the model to be tested.
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
    Test the manga model by loading the model and embeddings, calculating similarities,
    and saving the evaluation results. The test ensures that the top results have the
    expected structure and that the evaluation results are saved correctly.

    Args:
        new_description (str): The new description to compare against existing embeddings.
        model_name (str): The name of the model to be tested.
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
