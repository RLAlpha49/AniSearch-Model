"""
This module configures pytest options and fixtures for testing.

It includes a command line option to specify the model name for tests and a fixture
to retrieve the model name from the command line options. The model name is used
for loading pre-trained sentence transformer models during testing.

The default model is 'sentence-transformers/all-MiniLM-L6-v1', which provides a good
balance between performance and resource usage for testing purposes.
"""

import pytest
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest


def pytest_addoption(parser: Parser) -> None:
    """
    Add a command line option to specify the model name for tests.

    This function adds the '--model' option to pytest's command line interface,
    allowing users to specify which sentence transformer model to use during testing.

    Args:
        parser (Parser): The pytest command line argument parser.

    Example:
        pytest --model "sentence-transformers/all-mpnet-base-v2"
    """
    parser.addoption(
        "--model",
        action="store",
        default="sentence-transformers/all-MiniLM-L6-v1",
        help="Model name to use for tests",
    )


@pytest.fixture
def model_name(request: FixtureRequest) -> str:
    """
    Fixture to retrieve the model name from the command line options.

    This fixture provides access to the model name specified via the --model
    command line option. If no model is specified, it returns the default
    'sentence-transformers/all-MiniLM-L6-v1'.

    Args:
        request (FixtureRequest): The request object providing access to the test context
            and command line options.

    Returns:
        str: The model name specified in the command line options or the default value.
            The model name is typically in the format 'sentence-transformers/model-name'.
    """
    return str(request.config.getoption("--model"))
