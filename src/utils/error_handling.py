"""
# Error Handling Utilities

Standardized error handling functionality for the anime/manga search application.

This module provides reusable decorators and utilities to maintain consistent error
handling across the application. It implements a centralized approach to catching,
logging, and reporting exceptions, allowing for more maintainable and user-friendly
error handling.

## Features

- Decorator-based exception handling for consistent behavior
- Configurable error logging with severity control
- Customizable user-friendly error messages for CLI applications
- Type-safe implementation with proper generic typing

## Usage Context

These utilities are primarily used for:

1. Wrapping IO-heavy functions that may encounter file or network issues
2. Handling user input validation in CLI commands
3. Gracefully managing expected exceptions in model loading and inference
4. Providing informative error messages in both CLI and logging contexts

By centralizing error handling logic, the application maintains consistent behavior
across different components and provides better debugging information when issues occur.
"""

# pylint: disable=broad-exception-caught

import functools
import logging
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast, Union, Type

# Type variable for generic function signatures
F = TypeVar("F", bound=Callable[..., Any])

# Create a logger for this module
logger = logging.getLogger(__name__)

# Common exception types and their user-friendly messages
COMMON_EXCEPTIONS: Dict[Type[Exception], str] = {
    ValueError: "Invalid value or parameter",
    KeyError: "Missing data in results",
    ImportError: "Failed to import required module",
    RuntimeError: "Runtime error during operation",
    MemoryError: "Insufficient memory to process this operation",
    FileNotFoundError: "Required model or data file not found",
    PermissionError: "Permission denied when accessing files",
    TimeoutError: "Operation timed out",
    ConnectionError: "Network connection error",
}


def handle_exceptions(
    cli_mode: bool = False,
    exceptions: Optional[List[Type[Exception]]] = None,
    log_exceptions: bool = True,
    include_exc_info: bool = False,
    reraise: bool = True,
) -> Callable[[F], F]:
    """
    Decorator for handling exceptions in a standardized way across the application.

    This decorator wraps functions to provide consistent exception handling, including:

    1. Catching specified exceptions or all common exceptions by default
    2. Logging exceptions with configurable verbosity
    3. Presenting user-friendly error messages in CLI mode
    4. Optionally re-raising exceptions after handling

    The decorator can be configured for different contexts (CLI vs. background processing)
    and adjusted for different levels of verbosity and strictness.

    Args:
        cli_mode: Whether to print user-friendly messages to the console.
            When True, error messages are formatted for end users and printed to stdout.
            When False, errors are only logged (if log_exceptions is True).
            Default is False.

        exceptions: List of exception types to catch explicitly.
            If None (default), uses all exceptions defined in COMMON_EXCEPTIONS.
            Specify a subset of exceptions to handle only specific error types.

        log_exceptions: Whether to log the exceptions to the application logger.
            When True, exceptions are logged using the module's logger.
            When False, exceptions are not logged (useful when handled elsewhere).
            Default is True.

        include_exc_info: Whether to include exception traceback in logs.
            When True, full exception traceback is included in log messages.
            When False, only the exception message is logged.
            Default is False.

        reraise: Whether to re-raise the exception after handling.
            When True, the exception is re-raised after logging/printing.
            When False, the function returns None instead of re-raising.
            Default is True.

    Returns:
        Callable[[F], F]: A decorator function that wraps the target function
            with the specified exception handling behavior.

    Examples:
        Basic usage with default settings (logs exceptions and reraises):
        ```python
        @handle_exceptions()
        def load_data(filepath):
            with open(filepath) as f:
                return json.load(f)
        ```

        CLI-friendly error handling without reraising:
        ```python
        @handle_exceptions(cli_mode=True, reraise=False)
        def process_user_input(user_input):
            # Process user input, potentially raising exceptions
            return validated_input
        ```

        Handling only specific exceptions:
        ```python
        @handle_exceptions(
            exceptions=[FileNotFoundError, PermissionError],
            cli_mode=True
        )
        def read_config_file(filepath):
            with open(filepath) as f:
                return f.read()
        ```

    Notes:
        - When an exception is caught but not reraised (reraise=False), the function
          returns None. Callers should handle this case appropriately.
        - In CLI mode, caught exceptions generate user-friendly error messages based
          on the COMMON_EXCEPTIONS mapping.
        - Unexpected exceptions (not in the exceptions list) are always logged with
          full traceback information, regardless of the include_exc_info setting.
    """
    if exceptions is None:
        exceptions = list(COMMON_EXCEPTIONS.keys())

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:  # Catch all exceptions first to determine type
                exception_type = type(e)

                # Handle specific exception types
                if exception_type in exceptions:
                    # Get the exception message, defaulting to the exception type name
                    message = COMMON_EXCEPTIONS.get(
                        exception_type, str(exception_type.__name__)
                    )
                    error_detail = f"{message}: {str(e)}"

                    # Handle based on mode (CLI or logging)
                    if cli_mode:
                        print(f"Error: {error_detail}")

                    if log_exceptions:
                        context = f"Error in {func.__name__}"
                        logger.error(
                            "%s: %s", context, error_detail, exc_info=include_exc_info
                        )
                # Handle unexpected exceptions
                else:
                    if cli_mode:
                        print(f"Unexpected error: {str(e)}")
                        print("Please report this issue to the developers")

                    if log_exceptions:
                        logger.error(
                            "Unexpected error in %s: %s",
                            func.__name__,
                            str(e),
                            exc_info=True,
                        )

                if reraise:
                    raise
                return None

        return cast(F, wrapper)

    return decorator
