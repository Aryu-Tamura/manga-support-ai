"""Custom exception types for the Manga Support AI application."""


class LLMUnavailableError(RuntimeError):
    """Raised when an LLM call cannot be completed (rate limit, quota, etc.)."""

