"""
Bounded, Safe Page Fetching ControlledTool with SSRF and Injection Defense.
"""

from datetime import datetime, timezone
import hashlib
import io
import re
from typing import Any, Dict, Optional
import urllib.parse

from bs4 import BeautifulSoup
import httpx
from pydantic import BaseModel, Field, HttpUrl
import trafilatura

from app.config import settings
from app.logging_config import get_logger
from app.orchestration.tools import ControlledTool, ToolResult
from app.tools.network_safety import (
    ALLOWED_CONTENT_TYPES,
    RobotsPolicyEnforcer,
    SecurityViolationError,
    resolve_and_validate_host,
    validate_content_type,
    validate_url_scheme_and_domain,
)

logger = get_logger("app.tools.fetch_page")


class FetchPageInput(BaseModel):
    """Schema for safe webpage fetch request."""

    url: str = Field(..., description="Target webpage URL (http or https)")


class FetchPagePayload(BaseModel):
    """Clean extracted content and provenance metadata."""

    url: str = Field(..., description="Original requested URL")
    final_url: str = Field(..., description="Final URL after safe redirect resolution")
    title: str = Field(..., description="Extracted document or webpage title")
    text: str = Field(..., description="Clean extracted main text content")
    retrieval_timestamp: str = Field(..., description="ISO 8601 UTC retrieval timestamp")
    content_hash: str = Field(..., description="SHA-256 hash of extracted text")
    bytes_downloaded: int = Field(default=0, description="Total raw bytes downloaded")
    content_type: str = Field(default="text/html", description="MIME content type")
    redirect_count: int = Field(default=0, description="Number of redirects followed")
    raw_content: Optional[str] = Field(default=None, description="Raw downloaded response body for replay")


class FetchPageTool(ControlledTool):
    """ControlledTool that safely fetches and extracts webpage text with SSRF protection."""

    name = "fetch_page"
    description = "Safely fetch and extract clean text from a web page URL with SSRF protection."
    input_model = FetchPageInput

    def __init__(
        self,
        custom_settings: Optional[Any] = None,
        budget_tracker: Optional[Any] = None,
        cache_manager: Optional[Any] = None,
    ):
        from app.tools.web_cache import WebCacheManager
        self.cfg = custom_settings or settings
        self.budget_tracker = budget_tracker
        self.cache_manager = cache_manager or WebCacheManager(custom_settings=self.cfg)
        self.robots_enforcer = RobotsPolicyEnforcer(
            user_agent=getattr(self.cfg, "fetch_page_user_agent", "AdaptiveAgentBot/1.0 (+http://localhost:8000; web-research)")
        )

    async def execute(self, tool_input: FetchPageInput) -> ToolResult:
        """Execute safe fetch with DNS/IP resolution, redirect inspection, and content extraction."""
        from app.tools.web_cache import ReplayCacheMissError
        target_url = tool_input.url.strip()
        timestamp = datetime.now(timezone.utc).isoformat()

        # 1. Check disk cache / replay fixture
        try:
            cached_page = self.cache_manager.get_page(target_url)
            if cached_page is not None:
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    output=cached_page.model_dump(),
                    metadata={
                        "final_url": cached_page.final_url,
                        "content_hash": cached_page.content_hash,
                        "bytes": cached_page.bytes_downloaded,
                        "title": cached_page.title,
                        "from_cache": True,
                    },
                )
        except ReplayCacheMissError as miss_err:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"REPLAY cache miss: {miss_err}",
                metadata={"url": target_url, "cache_miss": True, "replay_mode": True},
            )

        # 2. Check budget only when network call is required
        if self.budget_tracker is not None:
            try:
                await self.budget_tracker.reserve_fetch()
            except Exception as b_err:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=f"Web fetch budget exceeded: {b_err}",
                    metadata={"url": target_url, "budget_exceeded": True},
                )

        try:
            payload = await self.fetch_and_extract(target_url, timestamp)
            if self.budget_tracker is not None:
                await self.budget_tracker.record_downloaded_bytes(payload.bytes_downloaded)

            # Save fixture to cache
            self.cache_manager.save_page(payload)

            return ToolResult(
                tool_name=self.name,
                success=True,
                output=payload.model_dump(),
                metadata={
                    "final_url": payload.final_url,
                    "content_hash": payload.content_hash,
                    "bytes": payload.bytes_downloaded,
                    "title": payload.title,
                    "from_cache": False,
                },
            )
        except SecurityViolationError as sec_err:
            logger.warning("fetch_page security violation for '%s': %s", target_url, sec_err)
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Security violation: {sec_err}",
                metadata={"url": target_url, "violation_type": "security"},
            )
        except httpx.TimeoutException as time_err:
            logger.warning("fetch_page timeout for '%s': %s", target_url, time_err)
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Request timed out after {getattr(self.cfg, 'fetch_page_timeout_seconds', 10.0)}s",
                metadata={"url": target_url, "violation_type": "timeout"},
            )
        except Exception as exc:
            logger.error("fetch_page failed for '%s': %s", target_url, exc, exc_info=True)
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Failed to fetch page: {str(exc)}",
                metadata={"url": target_url, "violation_type": "general"},
            )

    async def fetch_and_extract(self, url: str, timestamp: str) -> FetchPagePayload:
        """Internal worker executing safe network traversal and parsing."""
        max_redirects = getattr(self.cfg, "fetch_page_max_redirects", 5)
        max_bytes = getattr(self.cfg, "fetch_page_max_bytes", 2 * 1024 * 1024)
        timeout_sec = getattr(self.cfg, "fetch_page_timeout_seconds", 10.0)
        user_agent = getattr(self.cfg, "fetch_page_user_agent", "AdaptiveAgentBot/1.0 (+http://localhost:8000; web-research)")
        allowlist = getattr(self.cfg, "fetch_page_domain_allowlist_set", set())
        denylist = getattr(self.cfg, "fetch_page_domain_denylist_set", set())
        respect_robots = getattr(self.cfg, "fetch_page_respect_robots_txt", True)

        current_url = url
        redirect_count = 0
        final_content_bytes = bytearray()
        content_type = "text/html"

        headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,text/plain,application/pdf,*/*;q=0.8",
        }

        async with httpx.AsyncClient(
            headers=headers,
            timeout=timeout_sec,
            follow_redirects=False,
        ) as client:
            while True:
                # 1. Scheme and Domain Validation
                parsed = validate_url_scheme_and_domain(current_url, allowlist=allowlist, denylist=denylist)

                # 2. DNS Resolution and Prohibited IP Inspection
                port = parsed.port or (443 if parsed.scheme == "https" else 80)
                await resolve_and_validate_host(parsed.hostname or "", port=port)

                # 3. Robots.txt Compliance Check
                if respect_robots:
                    allowed_by_robots = await self.robots_enforcer.can_fetch(current_url, client=client)
                    if not allowed_by_robots:
                        raise SecurityViolationError(
                            f"Access to URL '{current_url}' is disallowed by site's robots.txt policy."
                        )

                # 4. Outbound Request Execution
                response = await client.get(current_url)

                # 5. Handle Redirects Manually with Re-Validation
                if response.status_code in (301, 302, 303, 307, 308):
                    redirect_count += 1
                    if redirect_count > max_redirects:
                        raise SecurityViolationError(
                            f"Exceeded maximum allowed redirects ({max_redirects})."
                        )
                    location = response.headers.get("Location")
                    if not location:
                        raise SecurityViolationError("Redirect response missing 'Location' header.")

                    current_url = urllib.parse.urljoin(current_url, location)
                    logger.debug("Following safe redirect %d to '%s'", redirect_count, current_url)
                    continue

                if response.status_code >= 400:
                    raise RuntimeError(f"HTTP error {response.status_code}: {response.reason_phrase}")

                # 6. Content-Type and Length Validation
                content_type = validate_content_type(response.headers.get("Content-Type"))

                # 7. Bounded Body Read
                body_bytes = response.content
                if len(body_bytes) > max_bytes:
                    raise SecurityViolationError(
                        f"Response body ({len(body_bytes)} bytes) exceeded maximum allowed limit ({max_bytes} bytes)."
                    )
                final_content_bytes = bytearray(body_bytes)
                break

        # 8. Clean Content & Title Extraction
        raw_text_decoded = None
        try:
            raw_text_decoded = bytes(final_content_bytes).decode("utf-8", errors="replace")
        except Exception:
            pass

        extracted_text, title = self._extract_content(bytes(final_content_bytes), content_type, current_url)

        max_chars = getattr(self.cfg, "fetch_page_max_chars", 8000)
        if len(extracted_text) > max_chars:
            extracted_text = extracted_text[:max_chars].rstrip() + "\n\n[...content truncated for model rate-limit safety...]"

        content_hash = hashlib.sha256(extracted_text.encode("utf-8")).hexdigest()

        return FetchPagePayload(
            url=url,
            final_url=current_url,
            title=title,
            text=extracted_text,
            retrieval_timestamp=timestamp,
            content_hash=content_hash,
            bytes_downloaded=len(final_content_bytes),
            content_type=content_type,
            redirect_count=redirect_count,
            raw_content=raw_text_decoded,
        )

    def _extract_content(self, data: bytes, content_type: str, url: str) -> tuple[str, str]:
        """Extract clean readable text and page title from HTML, PDF, or Plain text."""
        parsed_url = urllib.parse.urlparse(url)
        default_title = parsed_url.netloc

        if "pdf" in content_type:
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(data))
                text_parts = []
                for p in reader.pages:
                    t = p.extract_text()
                    if t:
                        text_parts.append(t)
                pdf_text = "\n\n".join(text_parts).strip()
                title = default_title
                if reader.metadata and reader.metadata.title:
                    title = str(reader.metadata.title)
                return pdf_text or "No readable text extracted from PDF.", title
            except Exception as e:
                logger.warning("PDF extraction failed: %s", e)
                return "Failed to parse PDF document.", default_title

        # Try HTML text extraction
        decoded_html = ""
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                decoded_html = data.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if not decoded_html:
            decoded_html = data.decode("utf-8", errors="replace")

        # Extract title using BeautifulSoup first
        title = default_title
        try:
            soup = BeautifulSoup(decoded_html, "html.parser")
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            elif soup.find("h1"):
                h1 = soup.find("h1")
                if h1 and h1.text:
                    title = h1.text.strip()
        except Exception:
            pass

        # Extract main article content via trafilatura
        extracted = None
        try:
            extracted = trafilatura.extract(
                decoded_html,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
            )
        except Exception as e:
            logger.debug("Trafilatura extraction exception: %s", e)

        if extracted and extracted.strip():
            return extracted.strip(), title

        # Fallback to BeautifulSoup if trafilatura yields empty
        try:
            for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                element.decompose()
            clean_bs4 = soup.get_text(separator="\n")
            lines = [line.strip() for line in clean_bs4.splitlines() if line.strip()]
            bs4_text = "\n".join(lines)
            if bs4_text.strip():
                return bs4_text.strip(), title
        except Exception:
            pass

        # Final plain text fallback
        return decoded_html.strip() or "Empty page content.", title
