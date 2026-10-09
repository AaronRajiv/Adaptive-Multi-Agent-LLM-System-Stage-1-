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
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        output_dimensions: int | None = None,
        batch_size: int = 64,
        max_concurrency: int = 5,
    ):
        if not api_key or not api_key.strip():
            raise EmbeddingError("Gemini embedding API key is missing.")
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.output_dimensions = output_dimensions
        self.batch_size = max(1, min(batch_size, 100))
        self.max_concurrency = max(1, max_concurrency)
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    async def embed(self, text: str) -> List[float]:
        results = await self.embed_many([text])
        if not results:
            raise EmbeddingError("Failed to generate embedding.")
        return results[0]

    async def embed_many(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        import asyncio

        # Partition texts into batches of size self.batch_size
        batches = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _process_batch(batch_texts: List[str]) -> List[List[float]]:
            async with semaphore:
                return await self._embed_batch_with_retry(batch_texts)

        results_by_batch = await asyncio.gather(*[_process_batch(b) for b in batches])

        all_vectors: List[List[float]] = []
        for batch_vectors in results_by_batch:
            all_vectors.extend(batch_vectors)

        if len(all_vectors) != len(texts):
            raise EmbeddingError(f"Embedding count mismatch: expected {len(texts)}, got {len(all_vectors)}.")

        return all_vectors

    async def _embed_batch_with_retry(
        self, batch_texts: List[str], max_retries: int = 3
    ) -> List[List[float]]:
        import asyncio

        url = f"{self.base_url}/{self.model}:batchEmbedContents"
        formatted_model = f"models/{self.model}" if not self.model.startswith("models/") else self.model

        requests_payload = []
        for text in batch_texts:
            req: dict = {
                "model": formatted_model,
                "content": {"parts": [{"text": text.strip() or " "}]},
            }
            if self.output_dimensions is not None:
                req["outputDimensionality"] = self.output_dimensions
            requests_payload.append(req)

        payload = {"requests": requests_payload}

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        url,
                        json=payload,
                        headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
                    )

                if response.status_code == 200:
                    data = response.json()
                    raw_embeddings = data.get("embeddings", [])
                    vectors = []
                    for item in raw_embeddings:
                        values = item.get("values", [])
                        vectors.append([float(v) for v in values])

                    if len(vectors) == len(batch_texts):
                        return vectors
                    raise EmbeddingError(
                        f"Expected {len(batch_texts)} embedding vectors, but received {len(vectors)}."
                    )

                # Transient rate-limit or temporary server issues: retry with exponential backoff
                if response.status_code in (429, 500, 503) and attempt < max_retries - 1:
                    await asyncio.sleep(1.0 * (2 ** attempt))
                    continue

                # Permanent error or retries exhausted
                err_text = response.text
                try:
                    err_json = response.json()
                    err_text = err_json.get("error", {}).get("message", err_text)
                except Exception:
                    pass
                raise EmbeddingError(
                    f"Gemini embedding API failed (HTTP {response.status_code}): {err_text}"
                )

            except httpx.RequestError as exc:
                if attempt < max_retries - 1:
                    await asyncio.sleep(1.0 * (2 ** attempt))
                    continue
                raise EmbeddingError(
                    f"Network error connecting to Gemini embedding API after {max_retries} attempts: {exc}"
                ) from exc


def get_embedding_provider(config):
    """Create the configured embedding provider without coupling retrieval to an LLM provider."""
    provider_name = config.embedding_provider.lower().strip()
    llm_provider_name = config.llm_provider.lower().strip()

    if provider_name == "mock" or llm_provider_name == "mock":
        api_key = config.embedding_api_key or config.llm_api_key
        if provider_name == "mock" or not api_key:
            return MockEmbeddingProvider(dimensions=config.knowledge_embedding_dimensions)

    if provider_name == "gemini":
        api_key = config.embedding_api_key or config.llm_api_key
        if not api_key:
            raise EmbeddingError("Gemini embedding API key is missing. Set GEMINI_API_KEY or KNOWLEDGE_EMBEDDING_API_KEY.")
        return GeminiEmbeddingProvider(
            api_key=api_key,
            model=config.embedding_model,
            output_dimensions=config.knowledge_embedding_dimensions,
            batch_size=getattr(config, "embedding_batch_size", 64),
            max_concurrency=getattr(config, "embedding_max_concurrency", 5),
        )

    raise EmbeddingError(
        f"Unsupported embedding provider '{config.embedding_provider}'. Supported: 'gemini', 'mock'."
    )
