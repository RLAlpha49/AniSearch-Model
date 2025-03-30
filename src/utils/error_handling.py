"""
Error handling utilities for the anime/manga search application.

This module provides reusable error handling functionality to maintain consistent
error handling across the application.
"""

# pylint: disable=broad-exception-caught

import functools
import logging
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

# Type variable for generic function signatures
F = TypeVar("F", bound=Callable[..., Any])

# Create a logger for this module
logger = logging.getLogger(__name__)

# Common exception types and their user-friendly messages
COMMON_EXCEPTIONS: Dict[type, str] = {
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
    exceptions: Optional[List[type]] = None,
    log_exceptions: bool = True,
    include_exc_info: bool = False,
    reraise: bool = True,
) -> Callable[[F], F]:
    """
    Decorator for handling exceptions in a standardized way.

    Args:
        cli_mode: Whether to print user-friendly messages (for CLI)
                 instead of just logging
        exceptions: List of exception types to catch explicitly
                   (defaults to COMMON_EXCEPTIONS keys)
        log_exceptions: Whether to log the exceptions
        include_exc_info: Whether to include exception info in logs
        reraise: Whether to re-raise the exception after handling

    Returns:
        Decorated function with standardized exception handling
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
