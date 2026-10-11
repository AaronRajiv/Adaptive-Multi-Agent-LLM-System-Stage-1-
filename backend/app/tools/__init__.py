"""
Tools package providing controlled tools for the multi-agent system.
"""

from app.tools.fetch_page import FetchPageInput, FetchPagePayload, FetchPageTool
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
from app.tools.search_provider import (
    CreditTracker,
    MockSearchProvider,
    RateLimiter,
    SearchProvider,
    SearchResponse,
    TavilyQuotaExceededError,
    TavilySearchProvider,
)
from app.tools.web_cache import (
    ReplayCacheMissError,
    WebCacheManager,
    compute_page_cache_key,
    compute_search_cache_key,
    normalize_canonical_url,
    normalize_search_query,
)
from app.tools.web_search import (
    SearchResultItem,
    WebSearchInput,
    WebSearchPayload,
    WebSearchTool,
)

__all__ = [
    "FetchPageInput",
    "FetchPagePayload",
    "FetchPageTool",
    "SearchResultItem",
    "WebSearchInput",
    "WebSearchPayload",
    "WebSearchTool",
    "SearchProvider",
    "TavilySearchProvider",
    "MockSearchProvider",
    "CreditTracker",
    "RateLimiter",
    "TavilyQuotaExceededError",
    "SearchResponse",
    "WebCacheManager",
    "ReplayCacheMissError",
    "compute_page_cache_key",
    "compute_search_cache_key",
    "normalize_canonical_url",
    "normalize_search_query",
    "SecurityViolationError",
    "RobotsPolicyEnforcer",
    "resolve_and_validate_host",
    "validate_ip_address",
    "validate_url_scheme_and_domain",
    "validate_content_type",
    "ALLOWED_CONTENT_TYPES",
    "IPCheckResult",
]
