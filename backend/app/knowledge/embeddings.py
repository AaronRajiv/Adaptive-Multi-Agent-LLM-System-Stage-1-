"""Provider-agnostic text embedding interfaces and implementations."""

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import List

import httpx


class EmbeddingError(RuntimeError):
    """Raised when text cannot be embedded by the configured provider."""


class EmbeddingProvider(ABC):
    """Provider-agnostic contract for converting text into numeric vectors."""

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate one embedding vector for non-empty text."""

    async def embed_many(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings in input order; providers may override for batching."""
        return [await self.embed(text) for text in texts]


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic hashed-token embeddings for offline tests and local development."""

    def __init__(self, dimensions: int = 64):
        if dimensions < 1:
            raise ValueError("Embedding dimensions must be at least 1.")
        self.dimensions = dimensions

    async def embed(self, text: str) -> List[float]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            raise EmbeddingError("Cannot embed empty text.")

        vector = [0.0] * self.dimensions
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        magnitude = math.sqrt(sum(value * value for value in vector))
        return [value / magnitude for value in vector] if magnitude else vector


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Gemini REST embedding provider using the same API-key style as GeminiProvider."""

    def __init__(
        self, api_key: str, model: str = "gemini-embedding-001", output_dimensions: int | None = None
    ):
        if not api_key or not api_key.strip():
            raise EmbeddingError("Gemini embedding API key is missing.")
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.output_dimensions = output_dimensions
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    async def embed(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text.")

        url = f"{self.base_url}/{self.model}:embedContent"
        payload = {"content": {"parts": [{"text": text.strip()}]}}
        if self.output_dimensions is not None:
            payload["config"] = {"outputDimensionality": self.output_dimensions}
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
                )
        except httpx.RequestError as exc:
            raise EmbeddingError(f"Network error connecting to Gemini embedding API: {exc}") from exc

        if response.status_code != 200:
            raise EmbeddingError(f"Gemini embedding API error (HTTP {response.status_code}): {response.text}")

        try:
            values = response.json()["embedding"]["values"]
            vector = [float(value) for value in values]
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingError("Gemini embedding API returned an invalid embedding response.") from exc
        if not vector:
            raise EmbeddingError("Gemini embedding API returned an empty vector.")
        return vector


def get_embedding_provider(config):
    """Create the configured embedding provider without coupling retrieval to an LLM provider."""
    provider_name = config.embedding_provider.lower().strip()
    if provider_name == "mock":
        return MockEmbeddingProvider(dimensions=config.knowledge_embedding_dimensions)
    if provider_name == "gemini":
        api_key = config.embedding_api_key or config.llm_api_key
        return GeminiEmbeddingProvider(
            api_key=api_key,
            model=config.embedding_model,
            output_dimensions=config.knowledge_embedding_dimensions,
        )
    raise EmbeddingError(
        f"Unsupported embedding provider '{config.embedding_provider}'. Supported: 'gemini', 'mock'."
    )
