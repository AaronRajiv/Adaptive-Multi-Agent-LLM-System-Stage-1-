"""
Search Provider Abstraction with Credit Tracking and Rate Limiting.
"""

from abc import ABC, abstractmethod
import asyncio
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel, Field

from app.logging_config import get_logger

logger = get_logger("app.tools.search_provider")


class TavilyQuotaExceededError(RuntimeError):
    """Raised when Tavily monthly credit allowance or hard stop is reached."""


class SearchResultItem(BaseModel):
    """Structured search hit item."""

    title: str = Field(..., description="Title of result")
    url: str = Field(..., description="Target URL")
    snippet: str = Field(default="", description="Snippet text")
    score: Optional[float] = Field(default=None, description="Relevance score")


class SearchResponse(BaseModel):
    """Normalized search response across providers."""

    query: str
    provider: str
    results: List[SearchResultItem] = Field(default_factory=list)
    total_results: int = 0
    retrieval_timestamp: str
    raw_response: Dict[str, Any] = Field(default_factory=dict, description="Raw provider response preserved for replay")


class CreditTracker:
    """
    Persists monthly Tavily credit usage and enforces configurable hard stops.
    Tavily rules:
    - basic search = 1 credit
    - extract = 1 credit per 5 URLs
    """

    def __init__(
        self,
        storage_path: str = "data/tavily_credits.json",
        monthly_limit: int = 1000,
        hard_stop: int = 800,
    ):
        self.storage_path = Path(storage_path)
        self.monthly_limit = monthly_limit
        self.hard_stop = hard_stop
        self._lock = asyncio.Lock()
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            self._write_state({"month": self._current_month(), "credits_used": 0})

    @staticmethod
    def _current_month() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m")

    def _read_state(self) -> Dict[str, Any]:
        if not self.storage_path.exists():
            return {"month": self._current_month(), "credits_used": 0}
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"month": self._current_month(), "credits_used": 0}

    def _write_state(self, state: Dict[str, Any]) -> None:
        temp_file = self.storage_path.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        temp_file.replace(self.storage_path)

    async def get_credits_used(self) -> int:
        async with self._lock:
            state = self._read_state()
            if state.get("month") != self._current_month():
                return 0
            return int(state.get("credits_used", 0))

    async def check_and_consume(self, cost: int) -> int:
        """
        Atomically verify that cost does not exceed hard stop, then consume credits.
        Raises TavilyQuotaExceededError if hard stop is reached.
        """
        async with self._lock:
            state = self._read_state()
            current_month = self._current_month()
            if state.get("month") != current_month:
                state = {"month": current_month, "credits_used": 0}

            used = int(state.get("credits_used", 0))
            if used + cost > self.hard_stop:
                raise TavilyQuotaExceededError(
                    f"Tavily credit hard stop reached: attempting to use {cost} credit(s) "
                    f"would exceed monthly hard stop of {self.hard_stop}/{self.monthly_limit} "
                    f"(currently used: {used}). Requests blocked to prevent overage."
                )

            used += cost
            state["credits_used"] = used
            self._write_state(state)
            return used


class RateLimiter:
    """
    Sliding window per-minute rate limiter with exponential backoff.
    """

    def __init__(self, max_per_minute: int = 60, max_retries: int = 3, base_backoff_sec: float = 0.5):
        self.max_per_minute = max_per_minute
        self.max_retries = max_retries
        self.base_backoff_sec = base_backoff_sec
        self._timestamps: List[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire rate limit slot, waiting with backoff if rate limit is reached."""
        for attempt in range(self.max_retries + 1):
            async with self._lock:
                now = time.monotonic()
                # Discard timestamps older than 60 seconds
                self._timestamps = [ts for ts in self._timestamps if now - ts < 60.0]

                if len(self._timestamps) < self.max_per_minute:
                    self._timestamps.append(now)
                    return

                # Need to wait
                oldest = self._timestamps[0]
                wait_time = max(0.1, 60.0 - (now - oldest))

            backoff = min(wait_time, self.base_backoff_sec * (2 ** attempt))
            logger.warning("Rate limit hit (%d/%d rpm). Backing off for %.2fs (attempt %d)",
                           len(self._timestamps), self.max_per_minute, backoff, attempt + 1)
            await asyncio.sleep(backoff)

        raise RuntimeError(f"Rate limit exceeded after {self.max_retries} backoff retries.")


class SearchProvider(ABC):
    """Vendor-agnostic interface for web search implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier."""

    @abstractmethod
    async def search(self, query: str, max_results: int) -> SearchResponse:
        """Execute bounded search and return normalized SearchResponse."""


class TavilySearchProvider(SearchProvider):
    """Production Tavily provider: search_depth=basic only, credit bounded, rate limited."""

    ENDPOINT = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: str,
        credit_tracker: Optional[CreditTracker] = None,
        rate_limiter: Optional[RateLimiter] = None,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self.api_key = api_key.strip()
        self.credit_tracker = credit_tracker or CreditTracker()
        self.rate_limiter = rate_limiter or RateLimiter()
        self._client = client

    @property
    def name(self) -> str:
        return "tavily"

    async def search(self, query: str, max_results: int) -> SearchResponse:
        if not self.api_key:
            raise ValueError("Tavily API key is not configured.")

        # 1. Enforce credit hard stop (basic search = 1 credit)
        await self.credit_tracker.check_and_consume(cost=1)

        # 2. Enforce per-minute rate limiter
        await self.rate_limiter.acquire()

        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",  # Mandated basic search only
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }

        should_close = False
        client = self._client
        if client is None:
            client = httpx.AsyncClient(timeout=10.0)
            should_close = True

        try:
            resp = await client.post(
                self.ENDPOINT,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            raw_data = resp.json()
        finally:
            if should_close:
                await client.aclose()

        timestamp = datetime.now(timezone.utc).isoformat()
        results: List[SearchResultItem] = []
        for item in raw_data.get("results", [])[:max_results]:
            url = str(item.get("url", "")).strip()
            title = str(item.get("title", "")).strip() or url
            snippet = str(item.get("content", "")).strip()
            score = item.get("score")
            if url:
                results.append(
                    SearchResultItem(
                        title=title,
                        url=url,
                        snippet=snippet,
                        score=float(score) if score is not None else None,
                    )
                )

        return SearchResponse(
            query=query,
            provider=self.name,
            results=results,
            total_results=len(results),
            retrieval_timestamp=timestamp,
            raw_response=raw_data,
        )


class MockSearchProvider(SearchProvider):
    """Deterministic mock provider for offline tests and fixture replay."""

    def __init__(self, predefined_results: Optional[Dict[str, List[Dict[str, Any]]]] = None):
        self.predefined = predefined_results or {}

    @property
    def name(self) -> str:
        return "mock"

    async def search(self, query: str, max_results: int) -> SearchResponse:
        timestamp = datetime.now(timezone.utc).isoformat()
        if query in self.predefined:
            hits = [
                SearchResultItem(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("snippet", ""),
                    score=item.get("score"),
                )
                for item in self.predefined[query][:max_results]
            ]
        else:
            hits = [
                SearchResultItem(
                    title=f"Mock Result 1 for {query}",
                    url="https://example.com/mock-1",
                    snippet=f"Deterministic mock snippet 1 for {query}.",
                    score=0.95,
                ),
                SearchResultItem(
                    title=f"Mock Result 2 for {query}",
                    url="https://example.com/mock-2",
                    snippet=f"Deterministic mock snippet 2 for {query}.",
                    score=0.85,
                ),
            ][:max_results]

        raw = {
            "query": query,
            "results": [h.model_dump() for h in hits],
            "response_time": 0.01,
        }

        return SearchResponse(
            query=query,
            provider=self.name,
            results=hits,
            total_results=len(hits),
            retrieval_timestamp=timestamp,
            raw_response=raw,
        )
