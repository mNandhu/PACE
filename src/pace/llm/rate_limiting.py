"""
Rate limiting and retry utilities for PACE tests and LLM interface.

This module provides utilities for handling rate limits, retries, and
reducing verbose error output during testing.
"""

import time
import logging
import warnings
from typing import Callable, Any, Optional
from functools import wraps

logger = logging.getLogger(__name__)


class RateLimitHandler:
    """Handles rate limiting with exponential backoff."""

    def __init__(
        self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def wait_with_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        delay = min(self.base_delay * (2**attempt), self.max_delay)
        return delay

    def handle_rate_limit(self, func: Callable, *args, **kwargs) -> Any:
        """
        Handle function execution with rate limit retry logic.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If all retries are exhausted
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                # Check if it's a rate limit error by examining the error message
                if "rate limit" in str(e).lower() or "429" in str(e):
                    last_exception = e

                    if attempt >= self.max_retries:
                        logger.error(
                            f"Rate limit exceeded after {self.max_retries} retries"
                        )
                        raise e

                    delay = self.wait_with_backoff(attempt)

                    # Clean console output for rate limiting
                    print(
                        f"⚠️  Rate limit reached. Waiting {delay:.1f}s before retry {attempt + 1}/{self.max_retries}..."
                    )
                    logger.warning(
                        f"Rate limit hit, waiting {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})"
                    )

                    time.sleep(delay)

                else:
                    # For non-rate-limit errors, fail immediately
                    logger.error(f"Non-rate-limit error occurred: {str(e)}")
                    raise e

        # This should never be reached, but just in case
        raise last_exception or Exception("Unknown error occurred")


def with_rate_limit_handling(
    max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0
):
    """
    Decorator for adding rate limit handling to functions.

    Args:
        max_retries: Maximum number of retries
        base_delay: Base delay for exponential backoff
        max_delay: Maximum delay between retries
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = RateLimitHandler(max_retries, base_delay, max_delay)
            return handler.handle_rate_limit(func, *args, **kwargs)

        return wrapper

    return decorator


def suppress_all_warnings():
    """Completely suppress all warnings during test execution."""
    warnings.filterwarnings("ignore")

    # Suppress specific warnings that tend to be verbose
    import urllib3

    urllib3.disable_warnings()
    # Suppress pydantic warnings
    try:
        import importlib.util

        if importlib.util.find_spec("pydantic"):
            warnings.filterwarnings(
                "ignore", category=DeprecationWarning, module="pydantic"
            )
    except ImportError:
        pass


def suppress_litellm_warnings():
    """Suppress verbose litellm warnings and debug output."""
    # Suppress litellm's verbose logging
    logging.getLogger("litellm").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("httpcore").setLevel(logging.ERROR)

    # Also suppress HTTP-related warnings
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("requests").setLevel(logging.ERROR)

    # Suppress warnings module output
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)


def create_safe_llm_call(llm_interface, max_retries: int = 3):
    """
    Create a safe LLM call function with rate limit handling.

    Args:
        llm_interface: The LLM interface instance
        max_retries: Maximum number of retries for rate limits

    Returns:
        A function that safely calls the LLM with rate limit handling
    """

    @with_rate_limit_handling(max_retries=max_retries)
    def safe_call(prompt: str, system_message: Optional[str] = None):
        return llm_interface.get_llm_response(prompt, system_message=system_message)

    return safe_call
