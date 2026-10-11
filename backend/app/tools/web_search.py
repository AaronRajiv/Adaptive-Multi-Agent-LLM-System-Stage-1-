"""
Controlled Web Search Tool using SearchProvider interface.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel, Field

from app.config import settings
from app.logging_config import get_logger
from app.orchestration.tools import ControlledTool, ToolResult
from app.tools.search_provider import (
    CreditTracker,
    MockSearchProvider,
    RateLimiter,
    SearchProvider,
    SearchResultItem,
    SearchResponse,
    TavilyQuotaExceededError,
    TavilySearchProvider,
)

logger = get_logger("app.tools.web_search")


class WebSearchInput(BaseModel):
    """Schema for bounded web search query."""

    query: str = Field(..., min_length=1, description="Search query string")
    max_results: int = Field(default=5, ge=1, le=20, description="Max search results to retrieve (1-20)")


class WebSearchPayload(BaseModel):
    """Structured payload returned on successful search execution."""

    query: str = Field(..., description="Original search query")
    provider: str = Field(..., description="Search provider used (tavily, mock)")
    results: List[SearchResultItem] = Field(default_factory=list, description="Ranked search results")
    total_results: int = Field(default=0, description="Count of returned items")
    retrieval_timestamp: str = Field(..., description="ISO 8601 UTC timestamp")


class WebSearchTool(ControlledTool):
    """ControlledTool that executes bounded queries against web search engines."""

    name = "web_search"
    description = (
        "Search the public web for real-time information, articles, documentation, or facts. "
        "Returns top ranked result titles, URLs, and snippets."
    )
    input_model = WebSearchInput

    def __init__(
        self,
        custom_settings: Optional[Any] = None,
        provider: Optional[SearchProvider] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        credit_tracker: Optional[CreditTracker] = None,
        rate_limiter: Optional[RateLimiter] = None,
        budget_tracker: Optional[Any] = None,
        cache_manager: Optional[Any] = None,
    ):
        from app.tools.web_cache import WebCacheManager
        self.cfg = custom_settings or settings
        self.credit_tracker = credit_tracker
        self.rate_limiter = rate_limiter
        self.budget_tracker = budget_tracker
        self.cache_manager = cache_manager or WebCacheManager(custom_settings=self.cfg)
        self._provider = provider

        if self._provider is None:
            prov_type = getattr(self.cfg, "web_search_provider", "tavily").lower()
            if prov_type == "mock":
                self._provider = MockSearchProvider()
            else:
                api_key = getattr(self.cfg, "tavily_api_key", "").strip()
                self._provider = TavilySearchProvider(
                    api_key=api_key,
                    credit_tracker=self.credit_tracker,
                    rate_limiter=self.rate_limiter,
                    client=http_client,
                )

    async def execute(self, tool_input: WebSearchInput) -> ToolResult:
        """Run search query against configured SearchProvider with caching & replay."""
        from app.tools.web_cache import ReplayCacheMissError
        query = tool_input.query.strip()
        if not query:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="Search query must not be empty or whitespace.",
                metadata={"provider": self._provider.name if self._provider else "unknown"},
            )

        max_results = max(1, min(tool_input.max_results, 20))
        provider_name = self._provider.name if self._provider else "tavily"

        # 1. Check disk cache / replay fixture
        try:
            cached_resp = self.cache_manager.get_search(
                provider=provider_name, query=query, max_results=max_results
            )
            if cached_resp is not None:
                payload = WebSearchPayload(
                    query=cached_resp.query,
                    provider=cached_resp.provider,
                    results=cached_resp.results,
                    total_results=cached_resp.total_results,
                    retrieval_timestamp=cached_resp.retrieval_timestamp,
                )
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    output=payload.model_dump(),
                    metadata={
                        "query": query,
                        "provider": cached_resp.provider,
                        "result_count": cached_resp.total_results,
                        "timestamp": cached_resp.retrieval_timestamp,
                        "raw_response": cached_resp.raw_response,
                        "from_cache": True,
                    },
                )
        except ReplayCacheMissError as miss_err:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"REPLAY cache miss: {miss_err}",
                metadata={"cache_miss": True, "provider": provider_name, "replay_mode": True},
            )

        # 2. Check budget only when cache miss (network call required)
        if self.budget_tracker is not None:
            try:
                await self.budget_tracker.reserve_search()
            except Exception as b_err:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=f"Web search budget exceeded: {b_err}",
                    metadata={"budget_exceeded": True, "provider": provider_name},
                )

        # Check for unconfigured Tavily API key upfront if using Tavily
        if isinstance(self._provider, TavilySearchProvider) and not self._provider.api_key:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="Tavily API key is not configured. Set TAVILY_API_KEY environment variable.",
                metadata={"provider": "tavily", "configured": False},
            )

        try:
            response: SearchResponse = await self._provider.search(query=query, max_results=max_results)
            # Save fixture to cache
            self.cache_manager.save_search(response, max_results=max_results)

            payload = WebSearchPayload(
                query=query,
                provider=response.provider,
                results=response.results,
                total_results=response.total_results,
                retrieval_timestamp=response.retrieval_timestamp,
            )
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=payload.model_dump(),
                metadata={
                    "query": query,
                    "provider": response.provider,
                    "result_count": response.total_results,
                    "timestamp": response.retrieval_timestamp,
                    "raw_response": response.raw_response,  # Stored for replay
                    "from_cache": False,
                },
            )
        except TavilyQuotaExceededError as q_err:
            logger.error("Tavily credit hard stop reached: %s", q_err)
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Credit quota exceeded: {q_err}",
                metadata={"query": query, "provider": self._provider.name, "quota_exceeded": True},
            )
        except httpx.TimeoutException as exc:
            logger.warning("Web search timed out for query '%s': %s", query, exc)
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="Web search request timed out.",
                metadata={"query": query, "provider": self._provider.name, "timeout": True},
            )
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            error_body = exc.response.text[:200]
            logger.error("Search provider HTTP %d error: %s", status_code, error_body)
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Search provider error (HTTP {status_code}): {error_body}",
                metadata={"query": query, "status_code": status_code, "provider": self._provider.name},
            )
        except Exception as exc:
            logger.error("Web search failed unexpectedly: %s", exc, exc_info=True)
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Web search failed: {str(exc)}",
                metadata={"query": query, "provider": self._provider.name},
            )
