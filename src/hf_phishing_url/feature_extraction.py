from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
import re
from typing import TYPE_CHECKING, Iterable
from urllib.parse import urlparse
from typing import TypedDict

if TYPE_CHECKING:
    import pandas as pd


_DEFAULT_BRANDS = {
    "adobe",
    "airbnb",
    "alibaba",
    "aliexpress",
    "amazon",
    "americanexpress",
    "amex",
    "apple",
    "bankofamerica",
    "binance",
    "bitbucket",
    "chase",
    "coinbase",
    "dhl",
    "discord",
    "dropbox",
    "ebay",
    "facebook",
    "fedex",
    "github",
    "gitlab",
    "gmail",
    "google",
    "icloud",
    "instagram",
    "linkedin",
    "microsoft",
    "netflix",
    "office",
    "outlook",
    "paypal",
    "reddit",
    "roblox",
    "salesforce",
    "samsung",
    "shopify",
    "slack",
    "snapchat",
    "spotify",
    "steam",
    "stripe",
    "telegram",
    "tiktok",
    "twitter",
    "ups",
    "usps",
    "venmo",
    "whatsapp",
    "yahoo",
    "zoom",
}


_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
_HAS_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


class UrlTokens(TypedDict):
    normalized: str
    hostname: str
    raw_lower: str
    host_lower: str
    path_lower: str
    raw_tokens: list[str]
    host_tokens: list[str]
    path_tokens: list[str]
    subdomain_tokens: list[str]
    tld: str


@dataclass(frozen=True)
class UrlTokenizer:
    """
    Tokenize a URL into parts.
    """

    def tokenize(self, url: str) -> UrlTokens:
        normalized = _normalize_url(url)
        parsed = urlparse(normalized)

        hostname = (parsed.hostname or "").lower()
        path = parsed.path or ""

        raw_lower = normalized.lower()
        host_lower = hostname.lower()
        path_lower = path.lower()

        tld = _infer_tld(host_lower)
        subdomain = _infer_subdomain(host_lower)

        return {
            "normalized": normalized,
            "hostname": hostname,
            "raw_lower": raw_lower,
            "host_lower": host_lower,
            "path_lower": path_lower,
            "raw_tokens": _tokens(raw_lower),
            "host_tokens": _tokens(host_lower),
            "path_tokens": _tokens(path_lower),
            "subdomain_tokens": _tokens(subdomain),
            "tld": tld if tld else "",
        }


@dataclass(frozen=True)
class UrlFeatureExtractor:
    """
    URL-only feature extraction to match the feature names listed in `notes.md`.

    The output includes `url` (string) plus numeric/boolean features. Booleans are
    encoded as 0/1 (F,T) ints.
    """

    brands: set[str] = field(default_factory=lambda: set(_DEFAULT_BRANDS))

    def extract_one(self, url: str) -> dict[str, int | float | str]:
        tokenizer =  UrlTokenizer()

        tokens = tokenizer.tokenize(url)
        normalized = tokens.get('normalized')
        hostname = tokens.get('hostname')
        path_lower = tokens.get('path_lower')
        host_lower = tokens.get('host_lower')
        raw_lower = tokens.get('raw_lower')
        raw_tokens = tokens.get('raw_tokens')
        host_tokens = tokens.get('host_tokens')
        path_tokens = tokens.get('path_tokens')
        subdomain_tokens = tokens.get('subdomain_tokens')
        tld = tokens.get('tld')

        subdomain = _infer_subdomain(host_lower)

        digits_url = _count_digits(raw_lower)
        digits_host = _count_digits(host_lower)

        length_url = len(normalized)
        length_hostname = len(hostname)

        features: dict[str, int | float | str] = {
            "url": url,
            "length_url": length_url,
            "length_hostname": length_hostname,
            "ip": int(_is_ip(hostname)),
            "nb_dots": raw_lower.count("."),
            "nb_hyphens": raw_lower.count("-"),
            "nb_at": raw_lower.count("@"),
            "nb_qm": raw_lower.count("?"),
            "nb_and": raw_lower.count("&"),
            "nb_or": raw_lower.count("|"),
            "nb_eq": raw_lower.count("="),
            "nb_underscore": raw_lower.count("_"),
            "nb_tilde": raw_lower.count("~"),
            "nb_percent": raw_lower.count("%"),
            "nb_slash": raw_lower.count("/"),
            "nb_star": raw_lower.count("*"),
            "nb_colon": raw_lower.count(":"),
            "nb_comma": raw_lower.count(","),
            "nb_semicolumn": raw_lower.count(";"),
            "nb_dollar": raw_lower.count("$"),
            "nb_space": raw_lower.count(" "),
            "nb_www": int(sum(1 for t in raw_tokens if t == "www")),
            "nb_com": int(sum(1 for t in raw_tokens if t == "com")),
            "nb_dslash": raw_lower.count("//"),
            "nb_subdomains": _count_subdomains(host_lower),
            "ratio_digits_url": float(digits_url / length_url) if length_url else 0.0,
            "ratio_digits_host": float(digits_host / length_hostname) if length_hostname else 0.0,
            "http_in_path": int("http" in path_lower),
            "tld_in_path": int(bool(tld) and tld in path_tokens),
            "tld_in_subdomain": int(bool(tld) and tld in subdomain_tokens),
            "prefix_suffix": int("-" in host_lower),
            "path_extension": int(_has_path_extension(path_lower)),
            "shortest_words_raw": _min_token_len(raw_tokens),
            "shortest_word_host": _min_token_len(host_tokens),
            "shortest_word_path": _min_token_len(path_tokens),
            "longest_words_raw": _max_token_len(raw_tokens),
            "longest_word_host": _max_token_len(host_tokens),
            "longest_word_path": _max_token_len(path_tokens),
            "brand_in_subdomain": int(_contains_brand(subdomain_tokens, self.brands)),
            "brand_in_path": int(_contains_brand(path_tokens, self.brands)),
        }
        return features

    def extract_many(self, urls: Iterable[str]) -> pd.DataFrame:
        import pandas as pd

        rows = [self.extract_one(u) for u in urls]
        return pd.DataFrame.from_records(rows)


def _normalize_url(url: str) -> str:
    '''Returns URL with scheme if exists, else prepend 'http://'.'''
    s = (url or "").strip()
    if not s:
        return ""
    if _HAS_SCHEME_RE.search(s):
        return s
    return "http://" + s


def _is_ip(hostname: str) -> bool:
    '''Returns whether the hostname is an IP address.'''
    if not hostname:
        return False
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def _infer_tld(hostname: str) -> str:
    '''Doesn't support multipart suffixes.'''
    if not hostname or _is_ip(hostname):
        return ""
    parts = [p for p in hostname.split(".") if p]
    if len(parts) < 2:
        return ""
    return parts[-1]


def _infer_subdomain(hostname: str) -> str:
    '''Assumes host name structure of `subdomain.domain.tld`.'''
    if not hostname or _is_ip(hostname):
        return ""
    parts = [p for p in hostname.split(".") if p]
    if len(parts) <= 2:
        return ""
    return ".".join(parts[:-2])


def _count_subdomains(hostname: str) -> int:
    '''Assumes host name structure of `subdomain.domain.tld`.'''
    if not hostname or _is_ip(hostname):
        return 0
    parts = [p for p in hostname.split(".") if p]
    return max(0, len(parts) - 2)


def _tokens(s: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(s or "")]


def _count_digits(s: str) -> int:
    return sum(ch.isdigit() for ch in (s or ""))


def _has_path_extension(path: str) -> bool:
    if not path:
        return False
    seg = (path.rsplit("/", 1)[-1] or "").strip()
    if not seg or seg in {".", ".."}:
        return False
    if "." not in seg:
        return False
    ext = seg.rsplit(".", 1)[-1]
    if not ext:
        return False
    if len(ext) > 5:
        return False
    return bool(re.fullmatch(r"[a-zA-Z0-9]+", ext))


def _min_token_len(tokens: list[str]) -> int:
    if not tokens:
        return 0
    return min(len(t) for t in tokens if t) if any(tokens) else 0


def _max_token_len(tokens: list[str]) -> int:
    if not tokens:
        return 0
    return max(len(t) for t in tokens if t) if any(tokens) else 0


def _contains_brand(tokens: list[str], brands: set[str]) -> bool:
    if not brands or not tokens:
        return False
    for t in tokens:
        if t in brands:
            return True
    return False
