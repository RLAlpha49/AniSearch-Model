"""
Logging configuration for the anime search model application.
"""

import logging
from src.utils.error_handling import handle_exceptions


@handle_exceptions(log_exceptions=True, include_exc_info=True)
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
