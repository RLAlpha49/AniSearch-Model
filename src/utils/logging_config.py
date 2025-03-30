"""
# Logging Configuration

Centralized logging setup for the anime/manga search application.

This module provides a standardized logging configuration to ensure consistent
log formatting, appropriate log levels, and proper output handling across all
application components. It implements a simple, reusable logging setup that
can be called during application initialization.

## Features

- Consistent timestamp and log level formatting
- Console output through StreamHandler
- Error handling for logging setup via handle_exceptions decorator
- INFO level logging by default for appropriate verbosity

## Usage Context

The logging configuration is typically initialized at application startup:

1. Called in the main entry point before any other operations
2. Used by error handling utilities to log exceptions
3. Available for all modules to use for consistent logging

Having a centralized logging configuration ensures that all components produce
logs in a consistent format, making debugging and monitoring easier.
"""

import logging
from src.utils.error_handling import handle_exceptions


@handle_exceptions(log_exceptions=True, include_exc_info=True)
def setup_logging() -> None:
    """
    Configure logging for the application with standardized formatting.

    This function initializes the Python logging system with consistent formatting,
    appropriate log levels, and console output. It sets up:

    - INFO level logging for moderate verbosity
    - Timestamp, level, and message formatting
    - Console output through a StreamHandler

    The function is decorated with handle_exceptions to ensure that any issues
    during logging configuration are properly captured and reported.

    Args:
        None

    Returns:
        None: This function configures the global logging system but doesn't
            return any value.

    Example:
        ```python
        # Initialize logging at application startup
        from src.utils.logging_config import setup_logging

        def main():
            # Set up logging first
            setup_logging()

            # Now all subsequent log calls will use this configuration
            logging.info("Application starting")
            # ...
        ```

    Notes:
        - This function should be called once at application startup
        - The log format includes timestamp, log level, and message
        - The default level (INFO) can be overridden through environment variables
          if the standard logging configuration mechanisms are used
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
