"""
Content-Addressed Web Cache and Deterministic Fixture REPLAY Manager.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional
import urllib.parse

from pydantic import BaseModel, Field

from app.config import settings
from app.logging_config import get_logger
from app.tools.fetch_page import FetchPagePayload
from app.tools.search_provider import SearchResponse, SearchResultItem

logger = get_logger("app.tools.web_cache")


class ReplayCacheMissError(RuntimeError):
    """Raised when a request cannot be satisfied from disk fixtures in REPLAY mode."""


def normalize_search_query(query: str) -> str:
    """Normalize query text by lowercasing and collapsing whitespace."""
    return re.sub(r"\s+", " ", query.strip().lower())


def compute_search_cache_key(provider: str, query: str, max_results: int) -> str:
    """
    Generate deterministic content-addressed hash for search request.
    Keyed by normalized (provider, query, max_results).
    """
    prov_norm = provider.strip().lower()
    query_norm = normalize_search_query(query)
    seed = f"{prov_norm}|{query_norm}|{max_results}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def normalize_canonical_url(url: str) -> str:
    """
    Normalize URL to canonical representation:
    - Lowercase scheme and domain
    - Strip default ports (80 for http, 443 for https)
    - Normalize empty path to '/' and strip trailing slashes (except root)
    - Sort query parameters alphabetically
    """
    parsed = urllib.parse.urlsplit(url.strip())
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # Strip default ports
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    elif scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]

    path = parsed.path
    if not path:
        path = "/"
    elif len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    # Sort query parameters
    query_parts = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query_parts.sort(key=lambda kv: (kv[0], kv[1]))
    normalized_query = urllib.parse.urlencode(query_parts)

    return urllib.parse.urlunsplit((scheme, netloc, path, normalized_query, ""))


def compute_page_cache_key(url: str) -> str:
    """
    Generate deterministic content-addressed hash for webpage fetch request.
    Keyed by normalized canonical URL.
    """
    canonical_url = normalize_canonical_url(url)
    return hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()


class WebCacheManager:
    """
    Unified manager for content-addressed disk caching and REPLAY fixtures.
    Modes:
    - 'live': Normal network calls; reads from disk cache if unexpired and writes back.
    - 'record': Executes live network calls and records raw fixtures to disk.
    - 'replay': Strictly serves from pre-recorded disk fixtures; fails with ReplayCacheMissError on miss.
    """

    def __init__(
        self,
        mode: Optional[str] = None,
        cache_dir: Optional[str] = None,
        ttl_seconds: Optional[float] = None,
        custom_settings: Optional[Any] = None,
    ):
        cfg = custom_settings or settings
        self.mode = (mode or getattr(cfg, "web_cache_mode", "live")).strip().lower()
        base_dir = cache_dir or getattr(cfg, "web_cache_dir", "data/web_cache")
        self.cache_dir = Path(base_dir)
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else getattr(cfg, "web_cache_ttl_seconds", 86400.0)

        self.search_dir = self.cache_dir / "searches"
        self.pages_dir = self.cache_dir / "pages"
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        self.search_dir.mkdir(parents=True, exist_ok=True)
        self.pages_dir.mkdir(parents=True, exist_ok=True)

    def is_replay_mode(self) -> bool:
        return self.mode == "replay"

    def is_record_mode(self) -> bool:
        return self.mode == "record"

    # --------------------------------------------------------------------------
    # Search Cache Methods
    # --------------------------------------------------------------------------

    def get_search(
        self, provider: str, query: str, max_results: int
    ) -> Optional[SearchResponse]:
        """
        Retrieve search response from disk cache or replay fixture.
        In 'replay' mode, raises ReplayCacheMissError if not found.
        """
        key = compute_search_cache_key(provider, query, max_results)
        fixture_path = self.search_dir / f"{key}.json"

        if not fixture_path.exists():
            if self.is_replay_mode():
                raise ReplayCacheMissError(
                    f"No recorded search fixture found for query '{query}' "
                    f"(provider='{provider}', max_results={max_results}, key='{key}') in REPLAY mode."
                )
            return None

        try:
            with open(fixture_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning("Failed to read search fixture %s: %s", fixture_path, exc)
            if self.is_replay_mode():
                raise ReplayCacheMissError(f"Corrupt search fixture for key '{key}': {exc}") from exc
            return None

        # Check TTL in live mode
        if self.mode == "live" and self.ttl_seconds > 0:
            created_at = data.get("cached_at_unix", 0)
            if time.time() - created_at > self.ttl_seconds:
                logger.debug("Search cache expired for query '%s' (key=%s)", query, key)
                return None

        # Reconstruct SearchResponse
        results = [SearchResultItem(**item) for item in data.get("results", [])]
        return SearchResponse(
            query=data.get("query", query),
            provider=data.get("provider", provider),
            results=results,
            total_results=data.get("total_results", len(results)),
            retrieval_timestamp=data.get("retrieval_timestamp", datetime.now(timezone.utc).isoformat()),
            raw_response=data.get("raw_response", {}),
        )

    def save_search(
        self, response: SearchResponse, max_results: int
    ) -> str:
        """Save search response to disk cache / fixture. Returns cache key."""
        key = compute_search_cache_key(response.provider, response.query, max_results)
        fixture_path = self.search_dir / f"{key}.json"

        record = {
            "key": key,
            "query": response.query,
            "provider": response.provider,
            "max_results": max_results,
            "total_results": response.total_results,
            "results": [r.model_dump() for r in response.results],
            "retrieval_timestamp": response.retrieval_timestamp,
            "raw_response": response.raw_response,
            "cached_at_unix": time.time(),
        }

        temp_path = fixture_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        temp_path.replace(fixture_path)
        logger.debug("Saved search fixture for '%s' -> %s", response.query, fixture_path)
        return key

    # --------------------------------------------------------------------------
    # Page Cache Methods
    # --------------------------------------------------------------------------

    def get_page(self, url: str) -> Optional[FetchPagePayload]:
        """
        Retrieve webpage content from disk cache or replay fixture.
        In 'replay' mode, raises ReplayCacheMissError if not found.
        """
        key = compute_page_cache_key(url)
        fixture_path = self.pages_dir / f"{key}.json"

        if not fixture_path.exists():
            if self.is_replay_mode():
                raise ReplayCacheMissError(
                    f"No recorded page fixture found for URL '{url}' "
                    f"(canonical_url='{normalize_canonical_url(url)}', key='{key}') in REPLAY mode."
                )
            return None

        try:
            with open(fixture_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning("Failed to read page fixture %s: %s", fixture_path, exc)
            if self.is_replay_mode():
                raise ReplayCacheMissError(f"Corrupt page fixture for key '{key}': {exc}") from exc
            return None

        # Check TTL in live mode
        if self.mode == "live" and self.ttl_seconds > 0:
            created_at = data.get("cached_at_unix", 0)
            if time.time() - created_at > self.ttl_seconds:
                logger.debug("Page cache expired for URL '%s' (key=%s)", url, key)
                return None

        return FetchPagePayload(
            url=data.get("url", url),
            final_url=data.get("final_url", url),
            title=data.get("title", ""),
            text=data.get("text", ""),
            retrieval_timestamp=data.get("retrieval_timestamp", datetime.now(timezone.utc).isoformat()),
            content_hash=data.get("content_hash", ""),
            bytes_downloaded=data.get("bytes_downloaded", 0),
            content_type=data.get("content_type", "text/html"),
            redirect_count=data.get("redirect_count", 0),
            raw_content=data.get("raw_content"),
        )

    def save_page(self, payload: FetchPagePayload) -> str:
        """Save fetched webpage payload to disk cache / fixture. Returns cache key."""
        key = compute_page_cache_key(payload.url)
        fixture_path = self.pages_dir / f"{key}.json"

        record = {
            "key": key,
            "url": payload.url,
            "canonical_url": normalize_canonical_url(payload.url),
            "final_url": payload.final_url,
            "title": payload.title,
            "text": payload.text,
            "retrieval_timestamp": payload.retrieval_timestamp,
            "content_hash": payload.content_hash,
            "bytes_downloaded": payload.bytes_downloaded,
            "content_type": payload.content_type,
            "redirect_count": payload.redirect_count,
            "raw_content": payload.raw_content,
            "cached_at_unix": time.time(),
        }

        temp_path = fixture_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        temp_path.replace(fixture_path)
        logger.debug("Saved page fixture for '%s' -> %s", payload.url, fixture_path)
        return key
