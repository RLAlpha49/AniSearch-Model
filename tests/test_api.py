"""
This module contains tests for the Flask API endpoints in the src.api module.

The tests verify the functionality of the /anisearchmodel/manga endpoint,
ensuring it handles valid inputs, missing fields, and internal server errors
correctly. The tests use a mock for the get_similarities function to simulate
different scenarios.

The test suite includes:
- Testing successful manga similarity search with valid inputs
- Testing error handling for invalid inputs (missing fields, invalid model names)
- Testing internal server error handling
- Parameterized tests for different invalid input scenarios

The tests use pytest fixtures for the Flask test client and model name configuration.
"""

import time
from unittest.mock import patch
from typing import Generator
import pytest
from flask.testing import FlaskClient
from src.api import app


@pytest.fixture
def client() -> Generator[FlaskClient, None, None]:
    """
    Fixture to create a test client for the Flask application.

    Returns:
        Generator[FlaskClient, None, None]: A Flask test client instance that can be used
            to make requests to the application endpoints.
    """
    app.config["TESTING"] = True
    with app.test_client() as client:  # pylint: disable=W0621
        yield client


@pytest.mark.order(13)
def test_get_manga_similarities_success(client: FlaskClient, model_name: str) -> None:  # pylint: disable=W0621
    """
    Test the /anisearchmodel/manga endpoint with valid input.

    Verifies that the endpoint returns a 200 status code and the expected
    list of similarities when provided with a valid model and description.

    Args:
        client (FlaskClient): Flask test client fixture
        model_name (str): Model name fixture from command line options

    The test:
        1. Creates a payload with valid model name and description
        2. Mocks the get_similarities function to return predefined results
        3. Verifies the response status code and structure of returned data
    """
    # Sample payload
    payload = {
        "model": model_name,
        "description": "A hero reincarnated as a slime.",
    }

    # Mock the get_similarities function
    with patch("src.api.get_similarities") as mock_get_similarities:
        mock_get_similarities.return_value = [
            {"name": "A slime with unique powers.", "similarity": 0.95},
            {"name": "Reincarnation in a fantasy world.", "similarity": 0.90},
        ]

        response = client.post("/anisearchmodel/manga", json=payload)
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["similarity"] == 0.95
    time.sleep(1)


@pytest.mark.parametrize(
    "payload, expected_error",
    [
        (
            {"description": "A hero reincarnated as a slime."},
            "Model name and description are required",
        ),
        (
            {
                "model": "invalid_model",
                "description": "A hero reincarnated as a slime.",
            },
            "Invalid model name",
        ),
        ({"model": "valid_model"}, "Model name and description are required"),
    ],
)
@pytest.mark.order(14)
def test_get_manga_similarities_invalid_input(
    client: FlaskClient,  # pylint: disable=W0621
    payload: dict,
    expected_error: str,
) -> None:
    """
    Test the /anisearchmodel/manga endpoint with invalid inputs.

    Verifies that the endpoint returns a 400 status code and an error message
    when the input is invalid.

    Args:
        client (FlaskClient): Flask test client fixture
        payload (dict): Test payload with invalid input combinations
        expected_error (str): Expected error message for the given invalid input

    The test cases verify:
        1. Missing model name
        2. Invalid model name
        3. Missing description
    """
    response = client.post("/anisearchmodel/manga", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert data["error"] == expected_error
    time.sleep(1)


@pytest.mark.order(15)
def test_get_manga_similarities_internal_error(
    client: FlaskClient,  # pylint: disable=W0621
    model_name: str,
) -> None:
    """
    Test the /anisearchmodel/manga endpoint for internal server errors.

    Verifies that the endpoint returns a 500 status code and an error message
    when an exception occurs during processing.

    Args:
        client (FlaskClient): Flask test client fixture
        model_name (str): Model name fixture from command line options

    The test:
        1. Creates a valid payload
        2. Mocks get_similarities to raise an exception
        3. Verifies the 500 status code and error message
    """
    payload = {
        "model": model_name,
        "description": "A hero reincarnated as a slime.",
    }

    with patch(
        "src.api.get_similarities", side_effect=Exception("Database connection failed")
    ):
        response = client.post("/anisearchmodel/manga", json=payload)
        assert response.status_code == 500
        data = response.get_json()
        assert "error" in data
        assert data["error"] == "Internal server error"
    time.sleep(1)
