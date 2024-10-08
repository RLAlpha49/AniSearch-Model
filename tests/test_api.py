"""
This module contains tests for the Flask API endpoints in the src.api module.

The tests verify the functionality of the /anisearchmodel/manga endpoint,
ensuring it handles valid inputs, missing fields, and internal server errors
correctly. The tests use a mock for the get_similarities function to simulate
different scenarios.
"""

import time
from unittest.mock import patch
import pytest
from src.api import app


@pytest.fixture
def client():
    """
    Fixture to create a test client for the Flask application.
    """
    app.config["TESTING"] = True
    with app.test_client() as client:  # pylint: disable=W0621
        yield client


@pytest.mark.order(13)
def test_get_manga_similarities_success(client, model_name):  # pylint: disable=W0621
    """
    Test the /anisearchmodel/manga endpoint with valid input.

    Verifies that the endpoint returns a 200 status code and the expected
    list of similarities when provided with a valid model and description.
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
def test_get_manga_similarities_invalid_input(client, payload, expected_error):  # pylint: disable=W0621
    """
    Test the /anisearchmodel/manga endpoint with invalid inputs.

    Verifies that the endpoint returns a 400 status code and an error message
    when the input is invalid.
    """
    response = client.post("/anisearchmodel/manga", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert data["error"] == expected_error
    time.sleep(1)


@pytest.mark.order(15)
def test_get_manga_similarities_internal_error(client, model_name):  # pylint: disable=W0621
    """
    Test the /anisearchmodel/manga endpoint for internal server errors.

    Verifies that the endpoint returns a 500 status code and an error message
    when an exception occurs during processing.
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
