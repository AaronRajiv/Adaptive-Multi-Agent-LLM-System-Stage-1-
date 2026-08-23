"""
LLM Provider Abstraction Interface.

Defines the generic contract for language model interactions across the multi-agent system.
Agents depend exclusively on this interface, decoupling agent logic from any specific LLM provider.
"""

from abc import ABC, abstractmethod
from typing import Optional, Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """Abstract base class defining the provider-agnostic interface for LLM calls."""

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate raw text response from the model.

        Args:
            prompt: The user or agent prompt text.
            system_prompt: Optional system instruction setting agent persona or constraints.

        Returns:
            The model's textual response string.

        Raises:
            LLMProviderError: If the model invocation fails or returns empty/invalid data.
            LLMConfigurationError: If API credentials or model settings are missing or invalid.
        """
        pass

    @abstractmethod
    async def generate_structured(
        self, prompt: str, schema: Type[T], system_prompt: Optional[str] = None
    ) -> T:
        """
        Generate structured output validated against a Pydantic schema.

        Args:
            prompt: The user or agent prompt text.
            schema: The target Pydantic model class to validate and instantiate.
            system_prompt: Optional system instruction setting agent persona or constraints.

        Returns:
            An instantiated and validated Pydantic model of type T.

        Raises:
            LLMStructuredOutputError: If output cannot be parsed or fails schema validation.
            LLMProviderError: If the model invocation fails.
            LLMConfigurationError: If credentials are missing.
        """
        pass


class LLMError(Exception):
    """Base exception for all LLM provider errors."""
    pass


class LLMConfigurationError(LLMError):
    """Raised when LLM credentials or configuration are missing or invalid."""
    pass


class LLMProviderError(LLMError):
    """Raised when the downstream LLM API returns an error or communication fails."""
    pass


class LLMStructuredOutputError(LLMError):
    """Raised when the model output cannot be parsed into the expected Pydantic schema."""
    pass
