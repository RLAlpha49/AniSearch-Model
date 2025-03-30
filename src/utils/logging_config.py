"""
Logging configuration for the anime search model application.
"""

import logging


def setup_logging() -> None:
    """
    Configure logging for the application.

    Sets up the logging format and level for the application.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
