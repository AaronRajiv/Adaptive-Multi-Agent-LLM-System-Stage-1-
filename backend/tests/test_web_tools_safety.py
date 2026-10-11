"""
Comprehensive safety and functionality tests for Controlled Web Tools:
- Network safety, SSRF defenses, and IP denylists
- DNS resolution, rebinding, and redirect defenses
- Content-type validation, max bytes bounding, and timeout enforcement
- Robots.txt enforcement
- FetchPageTool and WebSearchTool contracts and error handling
"""

import asyncio
from datetime import datetime
import ipaddress
import json
import socket
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import ValidationError

from app.config import Settings
from app.orchestration.tools import ToolRegistry, ToolRequest
from app.tools.fetch_page import FetchPageInput, FetchPageTool
from app.tools.network_safety import (
    ALLOWED_CONTENT_TYPES,
    IPCheckResult,
    RobotsPolicyEnforcer,
    SecurityViolationError,
    resolve_and_validate_host,
    validate_content_type,
    validate_ip_address,
    validate_url_scheme_and_domain,
)
from app.tools.web_search import (
    SearchResultItem,
    WebSearchInput,
    WebSearchPayload,
    WebSearchTool,
)


# ==============================================================================
# 1. IP Denylist & SSRF Protection Tests
# ==============================================================================

@pytest.mark.parametrize(
    "blocked_ip, reason_contains",
    [
        ("127.0.0.1", "loopback"),
        ("127.0.1.1", "loopback"),
        ("::1", "loopback"),
        ("10.0.0.1", "private network"),
        ("10.254.254.254", "private network"),
        ("172.16.0.1", "private network"),
        ("172.24.0.1", "private network"),
        ("172.31.255.255", "private network"),
        ("192.168.0.1", "private network"),
        ("192.168.1.100", "private network"),
        ("fc00::1", "private network"),
        ("fd12:3456:789a::1", "private network"),
        ("169.254.169.254", "cloud metadata"),
        ("169.254.1.1", "link-local"),
        ("fe80::1", "link-local"),
        ("100.100.100.200", "cloud metadata"),
        ("fd00:ec2::254", "cloud metadata"),
        ("100.64.0.1", "carrier-grade nat"),
        ("100.127.255.255", "carrier-grade nat"),
        ("0.0.0.0", "non-routable/reserved"),
        ("224.0.0.1", "multicast/reserved"),
        ("255.255.255.255", "multicast/reserved"),
    ],
)
def test_validate_ip_address_blocks_prohibited_ranges(blocked_ip: str, reason_contains: str):
    res = validate_ip_address(blocked_ip)
    assert not res.is_allowed
    assert reason_contains.lower() in res.reason.lower()


@pytest.mark.parametrize(
    "allowed_ip",
    [
        "8.8.8.8",
        "1.1.1.1",
        "93.184.216.34",  # example.com
        "140.82.121.3",   # github.com
        "2606:4700:4700::1111",  # cloudflare dns ipv6
    ],
)
def test_validate_ip_address_allows_public_ips(allowed_ip: str):
    res = validate_ip_address(allowed_ip)
    assert res.is_allowed
    assert res.ip == allowed_ip


# ==============================================================================
# 2. URL Scheme & Domain Allow/Denylist Tests
# ==============================================================================

@pytest.mark.parametrize(
    "invalid_scheme_url",
    [
        "file:///etc/passwd",
        "ftp://ftp.example.com/file",
        "gopher://evil.com/1",
        "data:text/html,<html>Hello</html>",
        "javascript:alert(1)",
        "chrome://settings",
    ],
)
def test_validate_url_scheme_blocks_non_http(invalid_scheme_url: str):
    with pytest.raises(SecurityViolationError, match="Prohibited URL scheme"):
        validate_url_scheme_and_domain(invalid_scheme_url)


def test_validate_url_scheme_blocks_denylisted_domain():
    denylist = {"malware.org", "evil.com"}
    with pytest.raises(SecurityViolationError, match="blocked by domain denylist"):
        validate_url_scheme_and_domain("https://evil.com/payload", denylist=denylist)

    # Subdomain check
    with pytest.raises(SecurityViolationError, match="blocked by domain denylist"):
        validate_url_scheme_and_domain("https://sub.evil.com/payload", denylist=denylist)


def test_validate_url_scheme_enforces_allowlist_when_configured():
    allowlist = {"trusted.org", "example.com"}

    # Allowed
    parsed = validate_url_scheme_and_domain("https://example.com/article", allowlist=allowlist)
    assert parsed.hostname == "example.com"
    assert parsed.scheme == "https"

    # Disallowed because not in allowlist
    with pytest.raises(SecurityViolationError, match="not in the configured domain allowlist"):
        validate_url_scheme_and_domain("https://wikipedia.org/wiki/Python", allowlist=allowlist)


# ==============================================================================
# 3. DNS Host Resolution & Rebinding Defenses
# ==============================================================================

@pytest.mark.asyncio
async def test_resolve_and_validate_host_blocks_private_dns_result():
    # Simulate a domain that resolves to 192.168.1.50
    mock_addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.50", 80))]

    with patch("asyncio.get_running_loop") as mock_get_loop:
        mock_loop = AsyncMock()
        mock_loop.getaddrinfo = AsyncMock(return_value=mock_addrinfo)
        mock_get_loop.return_value = mock_loop

        with pytest.raises(SecurityViolationError, match="prohibited IP address"):
            await resolve_and_validate_host("internal.local", 80)


@pytest.mark.asyncio
async def test_resolve_and_validate_host_succeeds_for_public_ip():
    mock_addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    with patch("asyncio.get_running_loop") as mock_get_loop:
        mock_loop = AsyncMock()
        mock_loop.getaddrinfo = AsyncMock(return_value=mock_addrinfo)
        mock_get_loop.return_value = mock_loop

        resolved_ips = await resolve_and_validate_host("example.com", 443)
        assert "93.184.216.34" in resolved_ips


# ==============================================================================
# 4. Robots.txt Policy Enforcement Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_robots_txt_enforcement_blocks_and_allows():
    robots_content = (
        "User-agent: *\n"
        "Disallow: /admin/\n"
        "Disallow: /private\n"
        "Allow: /public\n"
    )

    enforcer = RobotsPolicyEnforcer(user_agent="AdaptiveAgentBot/1.0")

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = httpx.Response(
        status_code=200,
        text=robots_content,
        request=httpx.Request("GET", "https://example.com/robots.txt"),
    )
    mock_client.get = AsyncMock(return_value=mock_resp)

    # Patch DNS check inside RobotsPolicyEnforcer to allow example.com
    with patch("app.tools.network_safety.resolve_and_validate_host", new_callable=AsyncMock) as mock_dns:
        mock_dns.return_value = ["93.184.216.34"]

        # Disallowed path
        allowed = await enforcer.can_fetch("https://example.com/admin/settings", client=mock_client)
        assert allowed is False

        # Disallowed exact prefix
        allowed = await enforcer.can_fetch("https://example.com/private", client=mock_client)
        assert allowed is False

        # Allowed path
        allowed = await enforcer.can_fetch("https://example.com/public/article", client=mock_client)
        assert allowed is True


# ==============================================================================
# 5. Content-Type Validation Tests
# ==============================================================================

def test_validate_content_type_accepts_clean_types():
    assert validate_content_type("text/html; charset=utf-8") == "text/html"
    assert validate_content_type("text/plain") == "text/plain"
    assert validate_content_type("application/pdf") == "application/pdf"
    assert validate_content_type("application/json") == "application/json"


@pytest.mark.parametrize(
    "bad_content_type",
    [
        "application/octet-stream",
        "application/x-executable",
        "application/x-msdos-program",
        "image/png",
        "video/mp4",
        "audio/mpeg",
    ],
)
def test_validate_content_type_rejects_disallowed_types(bad_content_type: str):
    with pytest.raises(SecurityViolationError, match="is not permitted"):
        validate_content_type(bad_content_type)


# ==============================================================================
# 6. FetchPageTool Execution and Edge Cases
# ==============================================================================

@pytest.mark.asyncio
async def test_fetch_page_blocks_private_ip_url_immediately():
    tool = FetchPageTool()
    res = await tool.execute(FetchPageInput(url="http://127.0.0.1:8000/api/status"))
    assert not res.success
    assert "Security violation" in res.error
    assert "prohibited address" in res.error.lower()


@pytest.mark.asyncio
async def test_fetch_page_blocks_file_scheme():
    tool = FetchPageTool()
    res = await tool.execute(FetchPageInput(url="file:///etc/passwd"))
    assert not res.success
    assert "Security violation" in res.error
    assert "Prohibited URL scheme" in res.error


@pytest.mark.asyncio
async def test_fetch_page_detects_ssrf_via_redirect_to_private():
    """Verify that a 302 redirecting to a private IP is blocked on the subsequent hop."""
    cfg = Settings(fetch_page_respect_robots_txt=False)
    tool = FetchPageTool(custom_settings=cfg)

    # First hop: 302 to http://192.168.1.1/secret
    def mock_handler(request: httpx.Request):
        return httpx.Response(
            302,
            headers={"Location": "http://192.168.1.1/secret"},
            request=request,
        )

    mock_transport = httpx.MockTransport(mock_handler)

    # Allow the initial public IP DNS resolution
    real_resolve = resolve_and_validate_host

    async def selective_dns(host, port=80):
        if host in ("public-domain.com", "93.184.216.34"):
            return ["93.184.216.34"]
        return await real_resolve(host, port)

    with patch("app.tools.fetch_page.resolve_and_validate_host", side_effect=selective_dns):
        with patch("httpx.AsyncClient", return_value=httpx.AsyncClient(transport=mock_transport)):
            res = await tool.execute(FetchPageInput(url="http://public-domain.com/start"))

    assert not res.success
    assert "Security violation" in res.error
    assert "prohibited address '192.168.1.1'" in res.error


@pytest.mark.asyncio
async def test_fetch_page_enforces_max_bytes_limit():
    """Verify that responses exceeding fetch_page_max_bytes are rejected."""
    cfg = Settings(
        fetch_page_max_bytes=1024,  # Minimum allowable config value (1 KB)
        fetch_page_respect_robots_txt=False,
    )
    tool = FetchPageTool(custom_settings=cfg)

    # Response with 2000 bytes
    large_body = b"A" * 2000
    mock_transport = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            content=large_body,
            headers={"Content-Type": "text/html", "Content-Length": str(len(large_body))},
            request=req,
        )
    )

    with patch("app.tools.fetch_page.resolve_and_validate_host", new_callable=AsyncMock) as mock_dns:
        mock_dns.return_value = ["93.184.216.34"]
        with patch("httpx.AsyncClient", return_value=httpx.AsyncClient(transport=mock_transport)):
            res = await tool.execute(FetchPageInput(url="http://example.com/large"))

    assert not res.success
    assert "exceeded maximum allowed limit" in res.error


@pytest.mark.asyncio
async def test_fetch_page_successful_html_extraction():
    cfg = Settings(fetch_page_respect_robots_txt=False)
    tool = FetchPageTool(custom_settings=cfg)

    html_content = (
        "<!DOCTYPE html>"
        "<html><head><title>Safe Research Page</title></head>"
        "<body>"
        "<header><nav><a href='#'>Menu</a></nav></header>"
        "<main><article>"
        "<h1>Safe Research Article</h1>"
        "<p>This is substantive paragraph content verified for LLM research agents.</p>"
        "<p>Another paragraph detailing multi-agent safety mechanisms.</p>"
        "</article></main>"
        "<footer><p>Copyright 2026</p></footer>"
        "</body></html>"
    )

    mock_transport = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            content=html_content.encode("utf-8"),
            headers={"Content-Type": "text/html; charset=utf-8"},
            request=req,
        )
    )

    with patch("app.tools.fetch_page.resolve_and_validate_host", new_callable=AsyncMock) as mock_dns:
        mock_dns.return_value = ["93.184.216.34"]
        with patch("httpx.AsyncClient", return_value=httpx.AsyncClient(transport=mock_transport)):
            res = await tool.execute(FetchPageInput(url="https://example.com/article"))

    assert res.success
    assert res.tool_name == "fetch_page"
    assert "Safe Research" in res.output["title"]
    assert "multi-agent safety mechanisms" in res.output["text"]
    assert len(res.output["content_hash"]) == 64
    assert res.output["bytes_downloaded"] > 0


# ==============================================================================
# 7. WebSearchTool Execution and Integration Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_web_search_unconfigured_api_key_returns_clear_error(tmp_path):
    cfg = Settings(web_search_provider="tavily", tavily_api_key="", web_cache_dir=str(tmp_path / "cache"))
    tool = WebSearchTool(custom_settings=cfg)

    res = await tool.execute(WebSearchInput(query="renewable energy storage"))
    assert not res.success
    assert "Tavily API key is not configured" in res.error
    assert res.metadata["configured"] is False


@pytest.mark.asyncio
async def test_web_search_empty_whitespace_query_validation(tmp_path):
    cfg = Settings(web_cache_dir=str(tmp_path / "cache"))
    tool = WebSearchTool(custom_settings=cfg)
    res = await tool.execute(WebSearchInput(query="   "))
    assert not res.success
    assert "must not be empty" in res.error


@pytest.mark.asyncio
async def test_web_search_mock_provider_fallback(tmp_path):
    cfg = Settings(web_search_provider="mock", web_cache_dir=str(tmp_path / "cache"))
    tool = WebSearchTool(custom_settings=cfg)

    res = await tool.execute(WebSearchInput(query="quantum computing advances", max_results=2))
    assert res.success
    assert res.tool_name == "web_search"
    assert res.output["provider"] == "mock"
    assert len(res.output["results"]) == 2
    assert "quantum computing advances" in res.output["results"][0]["title"]


@pytest.mark.asyncio
async def test_web_search_tavily_api_client_parsing(tmp_path):
    cfg = Settings(web_search_provider="tavily", tavily_api_key="tvly-testkey-123", web_cache_dir=str(tmp_path / "cache"))
    tracker = CreditTracker(storage_path=str(tmp_path / "credits.json"), monthly_limit=1000, hard_stop=800)

    fake_tavily_response = {
        "query": "solar photovoltaic efficiency 2026",
        "results": [
            {
                "title": "Perovskite Solar Cells Reach New Efficiency Milestone",
                "url": "https://science.org/perovskite-milestone",
                "content": "Researchers achieved 34% tandem cell efficiency under standard testing conditions.",
                "score": 0.98,
            },
            {
                "title": "Commercial Silicon Solar PV Trends",
                "url": "https://cleanenergy.gov/silicon-trends",
                "content": "Commercial TOPCon panels exceed 24% module efficiency in volume manufacturing.",
                "score": 0.91,
            },
        ],
        "response_time": 0.35,
    }

    mock_transport = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json=fake_tavily_response,
            headers={"Content-Type": "application/json"},
            request=req,
        )
    )

    async with httpx.AsyncClient(transport=mock_transport) as client:
        tool = WebSearchTool(custom_settings=cfg, http_client=client, credit_tracker=tracker)
        res = await tool.execute(WebSearchInput(query="solar photovoltaic efficiency 2026", max_results=5))

    assert res.success
    assert res.tool_name == "web_search"
    assert res.output["total_results"] == 2
    assert res.output["results"][0]["title"] == "Perovskite Solar Cells Reach New Efficiency Milestone"
    assert res.output["results"][0]["score"] == 0.98
    assert "science.org" in res.output["results"][0]["url"]


@pytest.mark.asyncio
async def test_web_search_tool_registry_invocation(tmp_path):
    """Verify registry can register, inspect schema, and invoke WebSearchTool."""
    cfg = Settings(web_search_provider="mock", web_cache_dir=str(tmp_path / "cache"))
    search_tool = WebSearchTool(custom_settings=cfg)

    registry = ToolRegistry()
    registry.register(search_tool)

    definition = registry.resolve("web_search").definition
    assert definition.name == "web_search"
    assert "query" in definition.input_schema["properties"]
    assert "max_results" in definition.input_schema["properties"]

    req = ToolRequest(name="web_search", arguments={"query": "battery storage grid", "max_results": 1})
    result = await registry.invoke(req)

    assert result.success
    assert result.tool_name == "web_search"
    assert len(result.output["results"]) == 1


# ==============================================================================
# 8. Amendment Verification Tests (Phase W1 User Requirements 1-7)
# ==============================================================================

from app.tools.search_provider import (
    CreditTracker,
    MockSearchProvider,
    RateLimiter,
    SearchProvider,
    TavilyQuotaExceededError,
    TavilySearchProvider,
)
from app.knowledge.models import KnowledgeDocument
from app.knowledge.chunking import WebPageChunker
from app.orchestration.evidence_ledger import EvidenceLedger
from app.orchestration.grounding_verifier import GroundingVerifier
from app.orchestration.resource_budget import AtomicWebBudgetTracker, WebBudget, ResourceBudgetExceededError


@pytest.mark.asyncio
async def test_credit_tracker_enforces_hard_stop(tmp_path):
    """Verify monthly credit tracker enforces 800-credit hard stop and persists state."""
    credits_file = tmp_path / "test_credits.json"
    tracker = CreditTracker(storage_path=str(credits_file), monthly_limit=1000, hard_stop=800)

    # 1. Consume 799 credits
    used = await tracker.check_and_consume(799)
    assert used == 799

    # 2. Attempting to consume 2 more should exceed 800 hard stop
    with pytest.raises(TavilyQuotaExceededError, match="credit hard stop reached"):
        await tracker.check_and_consume(2)

    # 3. Exactly 1 more reaches 800
    used_exact = await tracker.check_and_consume(1)
    assert used_exact == 800

    # 4. Next 1 is blocked
    with pytest.raises(TavilyQuotaExceededError):
        await tracker.check_and_consume(1)


@pytest.mark.asyncio
async def test_rate_limiter_allows_and_tracks():
    """Verify sliding-window rate limiter tracks requests."""
    limiter = RateLimiter(max_per_minute=5, max_retries=1, base_backoff_sec=0.01)
    for _ in range(5):
        await limiter.acquire()
    assert len(limiter._timestamps) == 5


@pytest.mark.asyncio
async def test_search_provider_interface_with_tavily(tmp_path):
    """Verify TavilySearchProvider complies with SearchProvider interface."""
    credits_file = tmp_path / "tavily_test_credits.json"
    tracker = CreditTracker(storage_path=str(credits_file), monthly_limit=1000, hard_stop=800)

    fake_response = {
        "query": "formula 1 active aero",
        "results": [{"title": "Active Aero in 2026", "url": "https://f1.com/aero", "content": "FIA confirms straight-line mode.", "score": 0.99}],
    }
    mock_transport = httpx.MockTransport(lambda req: httpx.Response(200, json=fake_response, request=req))

    async with httpx.AsyncClient(transport=mock_transport) as client:
        provider = TavilySearchProvider(
            api_key="tvly-mock-key",
            credit_tracker=tracker,
            client=client,
        )
        assert isinstance(provider, SearchProvider)
        assert provider.name == "tavily"

        resp = await provider.search("formula 1 active aero", max_results=3)
        assert resp.total_results == 1
        assert resp.results[0].title == "Active Aero in 2026"
        assert resp.raw_response == fake_response  # Raw response preserved for replay

    # Verify 1 credit consumed
    assert await tracker.get_credits_used() == 1


@pytest.mark.asyncio
async def test_web_citations_never_parsed_as_document_citations_even_with_art_in_text():
    """
    Requirement 3 Verification:
    Web citations whose text/domain contains 'art', 'article', 'page', or 'p.'
    (e.g. smart-example.com, particle-news.org) are never parsed as document citations.
    """
    ledger = EvidenceLedger()
    content_hash = "a1b2c3d4e5f67890123456789abcdef0123456789abcdef0123456789abcdef0"
    item = ledger.add_web_evidence(
        url="https://smart-example.com/article/part1",
        title="Smart Article on Battery Storage",
        text="Solar battery round-trip efficiency reaches 92% in 2026 commercial deployments.",
        retrieval_timestamp="2026-10-10T00:00:00Z",
        content_hash=content_hash,
    )
    assert item.citation == "[web:a1b2c3d4]"

    # Text containing domains/words with 'art', 'article', 'page', 'p.'
    claim_text = (
        "According to smart-example.com, solar battery round-trip efficiency reaches 92% [web:a1b2c3d4]. "
        "The smart particle article on page 5 also noted these figures."
    )

    result = GroundingVerifier.verify(claim_text, evidence_ledger=ledger)

    assert result.total_citations == 1
    assert result.valid_citations == 1
    assert result.passed is True
    # Verify it was categorized strictly as web, never as a document citation
    assert result.details["verifications"][0]["citation_type"] == "web"
    assert result.details["verifications"][0]["citation"] == "[web:a1b2c3d4]"


def test_web_page_chunker_preserves_title_and_paragraphs():
    """Requirement 4: WebPageChunker uses paragraph/heading boundaries and preserves page title."""
    chunker = WebPageChunker(target_tokens=20, max_tokens=35)

    web_text = (
        "# Overview of Formula 1 2026 Regulations\n\n"
        "The 2026 regulations represent a major shift in powertrain and aerodynamic philosophy across all teams.\n\n"
        "## Sustainable Fuels\n\n"
        "Formula 1 will transition to 100% advanced sustainable drop-in fuels from 2026 onward globally.\n\n"
        "## Power Unit Changes\n\n"
        "The MGU-H is abolished. Electrical power output increases from 120kW to 350kW with strict battery limits."
    )

    doc = KnowledgeDocument(
        id="web:https://fia.com/2026-overview",
        source="https://fia.com/2026-overview",
        content=web_text,
        metadata={"title": "FIA 2026 Overview", "content_hash": "b2c3d4e5"},
    )

    chunks = chunker.chunk(doc)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.metadata["source_type"] == "web"
        assert c.metadata["page_title"] == "FIA 2026 Overview"
        assert "embedding_text" in c.metadata
        assert "FIA 2026 Overview" in c.metadata["embedding_text"]


@pytest.mark.asyncio
async def test_atomic_web_budget_tracker_enforced_under_concurrency(tmp_path):
    """
    Requirement 5: Web budgets (searches, fetches, bytes) must be enforced atomically
    at call time under concurrency. Verifies parallel tool calls exceeding the limit.
    """
    budget = WebBudget(max_searches=3, max_fetches=3, max_bytes=5000, max_wall_time_seconds=10.0)
    tracker = AtomicWebBudgetTracker(budget)

    # 1. Parallel search calls: 10 parallel tasks attempting to execute searches
    async def try_search(task_id: int):
        try:
            await tracker.reserve_search()
            return "SUCCESS"
        except ResourceBudgetExceededError:
            return "BLOCKED"

    results = await asyncio.gather(*[try_search(i) for i in range(10)])
    success_count = sum(1 for r in results if r == "SUCCESS")
    blocked_count = sum(1 for r in results if r == "BLOCKED")

    # Exactly 3 must succeed, exactly 7 must be blocked atomically
    assert success_count == 3
    assert blocked_count == 7
    assert tracker.searches == 3

    # 2. Parallel fetch tool calls with FetchPageTool
    cfg = Settings(fetch_page_respect_robots_txt=False, web_cache_dir=str(tmp_path / "cache"))
    tool = FetchPageTool(custom_settings=cfg, budget_tracker=tracker)

    mock_transport = httpx.MockTransport(
        lambda req: httpx.Response(200, content=b"<html>Safe page</html>", headers={"Content-Type": "text/html"}, request=req)
    )

    async def try_fetch(idx: int):
        with patch("app.tools.fetch_page.resolve_and_validate_host", new_callable=AsyncMock) as m_dns:
            m_dns.return_value = ["93.184.216.34"]
            with patch("httpx.AsyncClient", return_value=httpx.AsyncClient(transport=mock_transport)):
                return await tool.execute(FetchPageInput(url=f"http://example.com/page{idx}"))

    fetch_results = await asyncio.gather(*[try_fetch(i) for i in range(8)])
    fetch_success = sum(1 for r in fetch_results if r.success)
    fetch_blocked = sum(1 for r in fetch_results if not r.success and "budget exceeded" in r.error.lower())

    # Exactly 3 fetches succeed (max_fetches=3), remaining 5 blocked
    assert fetch_success == 3
    assert fetch_blocked == 5
    assert tracker.fetches == 3


def test_deterministic_conflict_detection():
    """
    Requirement 7: Deterministic best-effort document-vs-web conflict detection.
    Detects numeric/value mismatch on matching topic.
    """
    from app.knowledge.models import RetrievedChunk

    ledger = EvidenceLedger()

    # Document chunk states 798kg minimum mass
    doc_chunk = RetrievedChunk(
        id="c-doc-4.1",
        document_id="doc-fia-2025",
        source="fia_2025.pdf",
        content="Article 4.1 Minimum Mass: The mass of the car without fuel must not be less than 798kg at all times.",
        similarity_score=0.95,
        chunk_index=0,
        metadata={"article": "4.1", "label": "FIA 2025"},
    )
    ledger.add_chunk(doc_chunk)

    # Web chunk states 800kg mass
    web_text = "Analysis indicates the vehicle minimum mass was revised to 800kg for subsequent trials."
    ledger.add_web_evidence(
        url="https://motorsport.org/weights",
        title="Weight Regulations Update",
        text=web_text,
        retrieval_timestamp="2026-10-10T12:00:00Z",
        content_hash="c3d4e5f678901234",
    )

    conflicts = ledger.detect_conflicts()
    assert len(conflicts) >= 1
    f = conflicts[0]
    assert f.topic in ("mass", "minimum")
    assert "798kg" in f.document_value or "798" in f.document_value
    assert "800kg" in f.web_value or "800" in f.web_value
    assert f.web_citation == "[web:c3d4e5f6]"
