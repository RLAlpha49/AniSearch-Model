"""
This module configures pytest options and fixtures for testing.

It includes a command line option to specify the model name for tests and a fixture
to retrieve the model name from the command line options.
"""

import pytest


def pytest_addoption(parser):
    """
    Add a command line option to specify the model name for tests.

    Args:
        parser (Parser): The parser for command line arguments.
    """
    parser.addoption(
        "--model",
        action="store",
        default="sentence-transformers/all-MiniLM-L6-v1",
        help="Model name to use for tests",
    )


@pytest.fixture
def model_name(request):
    """
    Fixture to retrieve the model name from the command line options.

    Args:
        request (FixtureRequest): The request object providing access to the test context.

    Returns:
        str: The model name specified in the command line options.
    """
    return request.config.getoption("--model")
