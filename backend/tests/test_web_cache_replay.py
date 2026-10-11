"""
Comprehensive tests for Phase W2: Content-Addressed Caching and REPLAY Mode.
- Search key normalization & content-addressing
- URL canonicalization & content-addressing
- Record mode (saving disk fixtures)
- Replay mode (serving from fixtures and failing loudly on cache miss)
- Live mode cache TTL expiry
- Zero-credit and zero-budget consumption on cache hits
"""

import json
from pathlib import Path
import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.config import Settings
from app.orchestration.resource_budget import AtomicWebBudgetTracker, WebBudget
from app.tools.fetch_page import FetchPageInput, FetchPagePayload, FetchPageTool
from app.tools.search_provider import CreditTracker, MockSearchProvider, SearchResponse, SearchResultItem
from app.tools.web_cache import (
    ReplayCacheMissError,
    WebCacheManager,
    compute_page_cache_key,
    compute_search_cache_key,
    normalize_canonical_url,
    normalize_search_query,
)
from app.tools.web_search import WebSearchInput, WebSearchTool


# ==============================================================================
# 1. Normalization and Content-Addressing Tests
# ==============================================================================

def test_search_cache_key_normalization():
    """Verify search keys are invariant to casing and whitespace variations."""
    key1 = compute_search_cache_key("Tavily", "quantum   computing  applications", 5)
    key2 = compute_search_cache_key("tavily", "Quantum Computing Applications", 5)
    key3 = compute_search_cache_key("tavily", "quantum computing applications", 5)

    assert key1 == key2 == key3
    assert len(key1) == 64  # SHA-256 hex digest

    # Different max_results or different query produces different key
    diff_key = compute_search_cache_key("tavily", "quantum computing applications", 10)
    assert diff_key != key1


def test_url_canonicalization_and_cache_key():
    """Verify URL canonicalization normalizes casing, default ports, query order, and trailing slashes."""
    url1 = "HTTPS://Example.COM:443/research/article/?beta=2&alpha=1"
    url2 = "https://example.com/research/article?alpha=1&beta=2"

    assert normalize_canonical_url(url1) == "https://example.com/research/article?alpha=1&beta=2"
    assert normalize_canonical_url(url2) == "https://example.com/research/article?alpha=1&beta=2"

    key1 = compute_page_cache_key(url1)
    key2 = compute_page_cache_key(url2)
    assert key1 == key2
    assert len(key1) == 64


# ==============================================================================
# 2. Record Mode Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_search_cache_record_mode_persists_fixture(tmp_path):
    """Verify WebCacheManager in record mode writes search fixture to disk."""
    cache_mgr = WebCacheManager(mode="record", cache_dir=str(tmp_path / "cache"))

    resp = SearchResponse(
        query="solid state batteries",
        provider="mock",
        results=[
            SearchResultItem(
                title="Solid State Battery Energy Density",
                url="https://battery-tech.org/solid-state",
                snippet="Solid state cells achieve 450 Wh/kg in 2026 pilot lines.",
                score=0.97,
            )
        ],
        total_results=1,
        retrieval_timestamp="2026-10-10T15:00:00Z",
        raw_response={"query": "solid state batteries", "raw_field": 123},
    )

    key = cache_mgr.save_search(resp, max_results=5)
    fixture_file = tmp_path / "cache" / "searches" / f"{key}.json"

    assert fixture_file.exists()
    with open(fixture_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["query"] == "solid state batteries"
    assert data["provider"] == "mock"
    assert data["raw_response"] == {"query": "solid state batteries", "raw_field": 123}
    assert len(data["results"]) == 1


@pytest.mark.asyncio
async def test_page_cache_record_mode_persists_fixture(tmp_path):
    """Verify WebCacheManager in record mode writes page fetch fixture to disk."""
    cache_mgr = WebCacheManager(mode="record", cache_dir=str(tmp_path / "cache"))

    payload = FetchPagePayload(
        url="https://example.com/deep-dive",
        final_url="https://example.com/deep-dive",
        title="Deep Dive Technical Analysis",
        text="Substantive clean article content extracted from HTML.",
        retrieval_timestamp="2026-10-10T16:00:00Z",
        content_hash="deadbeef0123456789abcdef",
        bytes_downloaded=1420,
        content_type="text/html",
        redirect_count=0,
        raw_content="<html><body><p>Raw downloaded body</p></body></html>",
    )

    key = cache_mgr.save_page(payload)
    fixture_file = tmp_path / "cache" / "pages" / f"{key}.json"

    assert fixture_file.exists()
    with open(fixture_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["title"] == "Deep Dive Technical Analysis"
    assert data["content_hash"] == "deadbeef0123456789abcdef"
    assert data["raw_content"] == "<html><body><p>Raw downloaded body</p></body></html>"


# ==============================================================================
# 3. Replay Mode Tests (Deterministic Playback & Loud Failure on Miss)
# ==============================================================================

@pytest.mark.asyncio
async def test_replay_mode_serves_from_disk_fixtures(tmp_path):
    """Verify replay mode returns pre-recorded disk fixtures accurately."""
    cache_dir = tmp_path / "replay_cache"
    cache_mgr = WebCacheManager(mode="replay", cache_dir=str(cache_dir))

    # Pre-seed fixture
    resp = SearchResponse(
        query="aerodynamics ground effect",
        provider="tavily",
        results=[
            SearchResultItem(
                title="Ground Effect Venturi Tunnels",
                url="https://f1technical.net/venturi",
                snippet="Venturi tunnels generate underfloor downforce.",
                score=0.99,
            )
        ],
        total_results=1,
        retrieval_timestamp="2026-10-10T10:00:00Z",
        raw_response={"seeded": True},
    )
    cache_mgr.save_search(resp, max_results=3)

    # In replay mode, get_search must return the exact seeded fixture
    cached = cache_mgr.get_search("tavily", "Aerodynamics  Ground Effect", 3)
    assert cached is not None
    assert cached.query == "aerodynamics ground effect"
    assert cached.results[0].title == "Ground Effect Venturi Tunnels"
    assert cached.raw_response == {"seeded": True}


@pytest.mark.asyncio
async def test_replay_mode_raises_loud_error_on_missing_search_fixture(tmp_path):
    """Verify replay mode raises ReplayCacheMissError on search cache miss."""
    cache_mgr = WebCacheManager(mode="replay", cache_dir=str(tmp_path / "empty_cache"))

    with pytest.raises(ReplayCacheMissError, match="No recorded search fixture found"):
        cache_mgr.get_search("tavily", "unrecorded query string", 5)


@pytest.mark.asyncio
async def test_replay_mode_raises_loud_error_on_missing_page_fixture(tmp_path):
    """Verify replay mode raises ReplayCacheMissError on page fetch cache miss."""
    cache_mgr = WebCacheManager(mode="replay", cache_dir=str(tmp_path / "empty_cache"))

    with pytest.raises(ReplayCacheMissError, match="No recorded page fixture found"):
        cache_mgr.get_page("https://unrecorded-domain.com/missing-page")


# ==============================================================================
# 4. Tool Integration with Zero-Credit & Zero-Budget on Cache Hit
# ==============================================================================

@pytest.mark.asyncio
async def test_web_search_tool_cache_hit_bypasses_credits_and_budget(tmp_path):
    """Verify cache hit skips network provider, consumes 0 credits, and uses 0 budget slots."""
    cache_dir = tmp_path / "tool_cache"
    cache_mgr = WebCacheManager(mode="live", cache_dir=str(cache_dir))

    # Pre-seed cache fixture
    seeded_resp = SearchResponse(
        query="carbon fiber composites",
        provider="mock",
        results=[
            SearchResultItem(
                title="Carbon Fiber Tensile Strength",
                url="https://composites.org/tensile",
                snippet="High-modulus carbon fiber exceeds 500 GPa tensile modulus.",
                score=0.96,
            )
        ],
        total_results=1,
        retrieval_timestamp="2026-10-10T12:00:00Z",
        raw_response={"mock_cached": True},
    )
    cache_mgr.save_search(seeded_resp, max_results=2)

    # Set up credit tracker with 0 initial usage and strict budget of 1 search
    credit_tracker = CreditTracker(storage_path=str(tmp_path / "credits.json"), monthly_limit=1000, hard_stop=800)
    budget = WebBudget(max_searches=1, max_fetches=1)
    budget_tracker = AtomicWebBudgetTracker(budget)

    # Provider mock that would raise if called
    mock_provider = MockSearchProvider()
    mock_provider.search = AsyncMock(side_effect=RuntimeError("Provider should NOT be called on cache hit!"))

    tool = WebSearchTool(
        provider=mock_provider,
        credit_tracker=credit_tracker,
        budget_tracker=budget_tracker,
        cache_manager=cache_mgr,
    )

    result = await tool.execute(WebSearchInput(query="Carbon Fiber Composites", max_results=2))

    assert result.success is True
    assert result.metadata["from_cache"] is True
    assert result.output["results"][0]["title"] == "Carbon Fiber Tensile Strength"

    # Verify 0 credits consumed and 0 budget reserved
    assert await credit_tracker.get_credits_used() == 0
    assert budget_tracker.searches == 0


@pytest.mark.asyncio
async def test_fetch_page_tool_cache_hit_bypasses_budget(tmp_path):
    """Verify page fetch cache hit skips outbound HTTP request and consumes 0 fetch budget."""
    cache_dir = tmp_path / "page_tool_cache"
    cache_mgr = WebCacheManager(mode="live", cache_dir=str(cache_dir))

    # Pre-seed cache fixture
    seeded_payload = FetchPagePayload(
        url="https://example.com/cached-article",
        final_url="https://example.com/cached-article",
        title="Pre-cached Article",
        text="Cached content text for deterministic testing.",
        retrieval_timestamp="2026-10-10T14:00:00Z",
        content_hash="hash12345678",
        bytes_downloaded=800,
        content_type="text/html",
        redirect_count=0,
    )
    cache_mgr.save_page(seeded_payload)

    budget = WebBudget(max_fetches=1, max_bytes=1000)
    budget_tracker = AtomicWebBudgetTracker(budget)

    cfg = Settings(fetch_page_respect_robots_txt=False)
    tool = FetchPageTool(
        custom_settings=cfg,
        budget_tracker=budget_tracker,
        cache_manager=cache_mgr,
    )

    # Execute fetch
    result = await tool.execute(FetchPageInput(url="https://example.com/cached-article"))

    assert result.success is True
    assert result.metadata["from_cache"] is True
    assert result.output["title"] == "Pre-cached Article"

    # Verify 0 fetch budget was used
    assert budget_tracker.fetches == 0
    assert budget_tracker.bytes_downloaded == 0


# ==============================================================================
# 5. Live Mode TTL Expiration Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_live_mode_ttl_expiration(tmp_path):
    """Verify live mode cache entry expires after configured ttl_seconds."""
    cache_mgr = WebCacheManager(mode="live", cache_dir=str(tmp_path / "ttl_cache"), ttl_seconds=2.0)

    resp = SearchResponse(
        query="temporary facts",
        provider="mock",
        results=[],
        total_results=0,
        retrieval_timestamp="2026-10-10T00:00:00Z",
        raw_response={},
    )
    key = cache_mgr.save_search(resp, max_results=1)

    # Immediately: valid
    assert cache_mgr.get_search("mock", "temporary facts", 1) is not None

    # Manually back-date the cached_at_unix in fixture
    fixture_path = tmp_path / "ttl_cache" / "searches" / f"{key}.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["cached_at_unix"] = time.time() - 100.0  # 100 seconds in past (> 2.0s TTL)
    with open(fixture_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    # Now: expired -> returns None
    assert cache_mgr.get_search("mock", "temporary facts", 1) is None
