"""
Network and SSRF Safety Module for Bounded Web Tools.

Enforces:
1. Scheme allowlist (http, https only).
2. DNS resolution and strict blocking of loopback, private, link-local, multicast,
   cloud metadata (e.g. AWS/GCP 169.254.169.254, IPv6 fd00:ec2::254), and carrier-grade NAT.
3. Domain allowlists and denylists.
4. Asynchronous robots.txt validation with per-origin caching.
5. Content-Type inspection against permitted text and document MIME types.
"""

import asyncio
import ipaddress
import socket
from typing import Dict, List, Optional, Set
import urllib.parse
import urllib.robotparser

import httpx

from app.logging_config import get_logger

logger = get_logger("app.tools.network_safety")


class SecurityViolationError(RuntimeError):
    """Raised when an outbound network request violates security policies or SSRF rules."""


# Explicitly blocked CIDRs for Cloud Metadata and special-use address blocks
PROHIBITED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),         # IPv4 Loopback
    ipaddress.ip_network("10.0.0.0/8"),          # RFC 1918 Private
    ipaddress.ip_network("172.16.0.0/12"),       # RFC 1918 Private
    ipaddress.ip_network("192.168.0.0/16"),      # RFC 1918 Private
    ipaddress.ip_network("169.254.0.0/16"),      # IPv4 Link-local / Cloud Metadata (169.254.169.254)
    ipaddress.ip_network("100.64.0.0/10"),       # Shared Address Space / CGNAT
    ipaddress.ip_network("192.0.0.0/24"),        # IETF Protocol Assignments
    ipaddress.ip_network("192.0.2.0/24"),        # TEST-NET-1
    ipaddress.ip_network("198.18.0.0/15"),       # Network Benchmark Tests
    ipaddress.ip_network("198.51.100.0/24"),     # TEST-NET-2
    ipaddress.ip_network("203.0.113.0/24"),      # TEST-NET-3
    ipaddress.ip_network("224.0.0.0/4"),         # Multicast
    ipaddress.ip_network("240.0.0.0/4"),         # Reserved / Future use
    ipaddress.ip_network("0.0.0.0/8"),           # "This" network
    ipaddress.ip_network("::1/128"),             # IPv6 Loopback
    ipaddress.ip_network("::/128"),              # IPv6 Unspecified
    ipaddress.ip_network("fc00::/7"),            # IPv6 Unique Local Address (ULA)
    ipaddress.ip_network("fe80::/10"),           # IPv6 Link-local
    ipaddress.ip_network("ff00::/8"),            # IPv6 Multicast
    ipaddress.ip_network("fd00:ec2::/64"),       # AWS IPv6 Metadata
]

# Explicit single metadata IPs for rapid exact matching
CLOUD_METADATA_IPS = {
    ipaddress.ip_address("169.254.169.254"),     # AWS / GCP / Azure metadata
    ipaddress.ip_address("100.100.100.200"),     # Alibaba Cloud metadata
}

# Permitted MIME types for document and webpage content extraction
ALLOWED_CONTENT_TYPES: Set[str] = {
    "text/html",
    "application/xhtml+xml",
    "text/plain",
    "text/markdown",
    "text/csv",
    "application/pdf",
    "application/json",
    "text/xml",
    "application/xml",
}


from dataclasses import dataclass

@dataclass(frozen=True)
class IPCheckResult:
    """Outcome of validating an IP address against safety policies."""

    ip: str
    is_allowed: bool
    reason: str = ""


def validate_ip_address(
    ip_input: str | ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> IPCheckResult:
    """Validate whether an IP address is permissible for outbound requests or prohibited."""
    try:
        if isinstance(ip_input, (ipaddress.IPv4Address, ipaddress.IPv6Address)):
            ip = ip_input
        else:
            ip = ipaddress.ip_address(str(ip_input).strip())
    except ValueError as exc:
        return IPCheckResult(ip=str(ip_input), is_allowed=False, reason=f"Invalid IP address: {exc}")

    # 1. Cloud metadata exact matches and subnets
    if ip in CLOUD_METADATA_IPS or ip in ipaddress.ip_network("fd00:ec2::/64"):
        return IPCheckResult(ip=str(ip), is_allowed=False, reason="Prohibited cloud metadata address")

    # 2. Loopback
    if ip.is_loopback or ip in ipaddress.ip_network("127.0.0.0/8") or ip == ipaddress.ip_address("::1"):
        return IPCheckResult(ip=str(ip), is_allowed=False, reason="Prohibited loopback address")

    # 3. Carrier-grade NAT (100.64.0.0/10)
    if ip in ipaddress.ip_network("100.64.0.0/10"):
        return IPCheckResult(ip=str(ip), is_allowed=False, reason="Prohibited carrier-grade NAT")

    # 4. Link-local (169.254.0.0/16, fe80::/10)
    if ip.is_link_local or ip in ipaddress.ip_network("169.254.0.0/16") or ip in ipaddress.ip_network("fe80::/10"):
        return IPCheckResult(ip=str(ip), is_allowed=False, reason="Prohibited link-local address")

    # 5. Non-routable "this" network (0.0.0.0/8) or unspecified (::)
    if ip in ipaddress.ip_network("0.0.0.0/8") or ip.is_unspecified:
        return IPCheckResult(ip=str(ip), is_allowed=False, reason="Prohibited non-routable/reserved address")

    # 6. Multicast / Broadcast / Reserved (224.0.0.0/4, 255.255.255.255, ff00::/8)
    if (
        ip.is_multicast
        or ip in ipaddress.ip_network("224.0.0.0/4")
        or ip in ipaddress.ip_network("ff00::/8")
        or str(ip) == "255.255.255.255"
    ):
        return IPCheckResult(ip=str(ip), is_allowed=False, reason="Prohibited multicast/reserved address")

    # 7. Standard RFC 1918 and IPv6 ULA private networks
    if ip.is_private or ip in ipaddress.ip_network("fc00::/7"):
        return IPCheckResult(ip=str(ip), is_allowed=False, reason="Prohibited private network address")

    # 8. Remaining prohibited ranges
    for net in PROHIBITED_NETWORKS:
        if ip in net:
            return IPCheckResult(ip=str(ip), is_allowed=False, reason="Prohibited reserved address block")

    if ip.is_reserved:
        return IPCheckResult(ip=str(ip), is_allowed=False, reason="Prohibited non-routable/reserved address")

    return IPCheckResult(ip=str(ip), is_allowed=True, reason="Public IP address")


def is_ip_prohibited(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Check if an IP address belongs to loopback, private, link-local, multicast, or metadata."""
    return not validate_ip_address(ip).is_allowed


async def resolve_and_validate_host(hostname: str, port: int = 80) -> List[str]:
    """
    Resolve hostname via DNS and ensure none of the resolved IP addresses are prohibited.
    Returns list of validated IP string addresses.
    """
    clean_host = hostname.strip().strip("[]").lower()
    if not clean_host:
        raise SecurityViolationError("Target hostname is empty.")

    # Direct IP literal check
    try:
        ip = ipaddress.ip_address(clean_host)
        if is_ip_prohibited(ip):
            raise SecurityViolationError(
                f"Direct IP access to prohibited address '{ip}' is blocked."
            )
        return [str(ip)]
    except ValueError:
        pass

    # Domain name resolution
    loop = asyncio.get_running_loop()
    try:
        addrinfo = await loop.getaddrinfo(
            clean_host,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise SecurityViolationError(f"DNS resolution failed for host '{clean_host}': {exc}") from exc

    resolved_ips: List[str] = []
    for entry in addrinfo:
        sockaddr = entry[4]
        ip_str = sockaddr[0]
        try:
            ip_obj = ipaddress.ip_address(ip_str)
            if is_ip_prohibited(ip_obj):
                raise SecurityViolationError(
                    f"Host '{clean_host}' resolved to prohibited IP address '{ip_obj}'."
                )
            if ip_str not in resolved_ips:
                resolved_ips.append(ip_str)
        except ValueError as exc:
            raise SecurityViolationError(f"Invalid IP address returned from DNS: {ip_str}") from exc

    if not resolved_ips:
        raise SecurityViolationError(f"DNS resolution returned no valid IP addresses for host '{clean_host}'.")

    return resolved_ips


def validate_url_scheme_and_domain(
    url: str,
    allowlist: Optional[Set[str]] = None,
    denylist: Optional[Set[str]] = None,
) -> urllib.parse.ParseResult:
    """
    Validate that URL uses http/https and adheres to domain allow/deny lists.
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as exc:
        raise SecurityViolationError(f"Malformed URL '{url}': {exc}") from exc

    if parsed.scheme.lower() not in ("http", "https"):
        raise SecurityViolationError(
            f"Prohibited URL scheme '{parsed.scheme}'. Only 'http' and 'https' are allowed."
        )

    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        raise SecurityViolationError(f"URL '{url}' does not specify a valid hostname.")

    # Denylist check
    if denylist:
        for denied in denylist:
            denied_norm = denied.strip().lower()
            if denied_norm and (hostname == denied_norm or hostname.endswith(f".{denied_norm}")):
                raise SecurityViolationError(f"Host '{hostname}' is blocked by domain denylist.")

    # Allowlist check
    if allowlist and len(allowlist) > 0:
        allowed_match = False
        for allowed in allowlist:
            allowed_norm = allowed.strip().lower()
            if allowed_norm and (hostname == allowed_norm or hostname.endswith(f".{allowed_norm}")):
                allowed_match = True
                break
        if not allowed_match:
            raise SecurityViolationError(f"Host '{hostname}' is not in the configured domain allowlist.")

    return parsed


def validate_content_type(content_type_header: Optional[str]) -> str:
    """
    Ensure response Content-Type matches allowed text/document MIME types.
    """
    if not content_type_header:
        # Default assumption is text/html if not specified, but return default
        return "text/html"

    main_type = content_type_header.split(";")[0].strip().lower()
    if main_type not in ALLOWED_CONTENT_TYPES:
        raise SecurityViolationError(
            f"Content-Type '{main_type}' is not permitted. Allowed types: {sorted(ALLOWED_CONTENT_TYPES)}"
        )
    return main_type


class RobotsPolicyEnforcer:
    """
    Asynchronously checks and caches robots.txt rules per origin to respect crawler guidelines.
    """

    def __init__(self, user_agent: str, cache_ttl_seconds: float = 3600.0):
        self.user_agent = user_agent
        self.cache_ttl = cache_ttl_seconds
        self._parsers: Dict[str, urllib.robotparser.RobotFileParser] = {}
        self._fetch_timestamps: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def can_fetch(self, target_url: str, client: Optional[httpx.AsyncClient] = None) -> bool:
        """
        Check if user agent is permitted to fetch target_url according to the site's robots.txt.
        """
        parsed = urllib.parse.urlparse(target_url)
        origin = f"{parsed.scheme}://{parsed.netloc}"

        async with self._lock:
            cached_parser = self._parsers.get(origin)

        if cached_parser is None:
            parser = urllib.robotparser.RobotFileParser()
            robots_url = f"{origin}/robots.txt"

            # Validate robots.txt host safety before fetching
            try:
                validate_url_scheme_and_domain(robots_url)
                await resolve_and_validate_host(parsed.hostname or "", parsed.port or (443 if parsed.scheme == "https" else 80))
            except SecurityViolationError:
                # If robots.txt origin is prohibited, can_fetch is strictly False
                return False

            try:
                if client is not None:
                    resp = await client.get(robots_url, timeout=5.0)
                else:
                    async with httpx.AsyncClient(timeout=5.0) as local_client:
                        resp = await local_client.get(robots_url)

                if resp.status_code == 200:
                    parser.parse(resp.text.splitlines())
                elif resp.status_code in (401, 403):
                    # Disallowed if robots.txt itself is restricted
                    parser.parse(["User-agent: *", "Disallow: /"])
                else:
                    # 404 or other: allow all per RFC 9309
                    parser.parse(["User-agent: *", "Allow: /"])
            except Exception as exc:
                logger.debug("Failed to fetch robots.txt for '%s': %s. Defaulting to allow.", origin, exc)
                parser.parse(["User-agent: *", "Allow: /"])

            async with self._lock:
                self._parsers[origin] = parser

            cached_parser = parser

        return cached_parser.can_fetch(self.user_agent, target_url)
