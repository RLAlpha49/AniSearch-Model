"""
This module contains tests for the Flask API endpoints in the src.api module.

The tests verify the functionality of the /anisearchmodel/manga endpoint,
ensuring it handles valid inputs, missing fields, and internal server errors
correctly. The tests use a mock for the get_similarities function to simulate
different scenarios.
"""

from unittest.mock import patch
import time
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


@pytest.mark.order(10)
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
            {"description": "A slime with unique powers.", "score": 0.95},
            {"description": "Reincarnation in a fantasy world.", "score": 0.90},
        ]

        response = client.post("/anisearchmodel/manga", json=payload)
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["score"] == 0.95
    time.sleep(1)


@pytest.mark.order(11)
def test_get_manga_similarities_missing_model(client):  # pylint: disable=W0621
    """
    Test the /anisearchmodel/manga endpoint with a missing model.

    Verifies that the endpoint returns a 400 status code and an error message
    when the model is not provided in the request payload.
    """
    payload = {"description": "A hero reincarnated as a slime."}

    response = client.post("/anisearchmodel/manga", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert data["error"] == "Model name and description are required"
    time.sleep(1)


@pytest.mark.order(12)
def test_get_manga_similarities_missing_description(client, model_name):  # pylint: disable=W0621
    """
    Test the /anisearchmodel/manga endpoint with a missing description.

    Verifies that the endpoint returns a 400 status code and an error message
    when the description is not provided in the request payload.
    """
    payload = {"model": model_name}

    response = client.post("/anisearchmodel/manga", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert data["error"] == "Model name and description are required"
    time.sleep(1)


@pytest.mark.order(13)
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
