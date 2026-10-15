"""Rate-limited, cached JSON fetching shared by the source fetchers.

Failures are explicit. A non-200 status, an HTML page where JSON was expected (some sites
answer *200* with a firewall block page), or unparsable JSON all raise
``SourceUnavailableError``. Nothing is retried silently, and stale cache entries are never
served in place of a failed request.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from stock_research.data.cache import Cache
from stock_research.data.errors import MissingDataError


class SourceUnavailableError(MissingDataError):
    """The source could not be reached, refused the request or returned unusable content."""

    def __init__(self, source: str, url: str, actual: Any, expected: str) -> None:
        super().__init__(None, f"{source}:{url}", actual, expected)
        self.source = source
        self.url = url


class SourceRateLimitedError(SourceUnavailableError):
    """The source reports that its request quota is used up."""


class _Response(Protocol):
    status_code: int
    headers: Mapping[str, str]
    text: str


class _Session(Protocol):
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = ...,
        headers: Mapping[str, str] | None = ...,
        timeout: float = ...,
    ) -> _Response: ...


class RateLimiter:
    """Minimum spacing between requests plus an optional rolling one-hour cap."""

    def __init__(
        self,
        min_interval_seconds: float = 0.0,
        max_per_hour: int | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = min_interval_seconds
        self.max_per_hour = max_per_hour
        self._clock = clock
        self._sleep = sleep
        self._sent: deque[float] = deque()

    def wait(self) -> None:
        now = self._clock()
        if self._sent and self.min_interval > 0:
            gap = now - self._sent[-1]
            if gap < self.min_interval:
                self._sleep(self.min_interval - gap)
                now = self._clock()
        if self.max_per_hour is not None:
            while self._sent and now - self._sent[0] >= 3600:
                self._sent.popleft()
            if len(self._sent) >= self.max_per_hour:
                # Queue rather than skip: skipping companies would bias industry medians.
                self._sleep(3600 - (now - self._sent[0]))
                now = self._clock()
                self._sent.popleft()
        self._sent.append(now)


@dataclass(frozen=True, slots=True)
class JsonResult:
    data: Any
    fetched_at: datetime
    from_cache: bool


USER_AGENT = "stock-research/0.1 (personal research tool)"


class JsonClient:
    def __init__(
        self,
        source: str,
        session: _Session,
        limiter: RateLimiter,
        cache: Cache | None = None,
        *,
        timeout_seconds: float = 60.0,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.source = source
        self._session = session
        self._limiter = limiter
        self._cache = cache
        self._timeout = timeout_seconds
        self._headers = {"User-Agent": USER_AGENT, "Accept": "application/json", **(headers or {})}

    def get_json(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        *,
        max_age: timedelta | None,
        check: Callable[[Any], None] | None = None,
    ) -> JsonResult:
        """GET ``url`` as JSON, serving the cache when it is younger than ``max_age``.

        ``check`` validates the decoded body (e.g. an API's own error envelope) before it is
        cached, so error responses never enter the cache. Auth headers are not part of the
        cache key and are never stored.
        """
        params = dict(params or {})
        if self._cache is not None and max_age is not None:
            hit = self._cache.get_raw(self.source, url, params, max_age=max_age)
            if hit is not None:
                return JsonResult(json.loads(hit.payload), hit.fetched_at, from_cache=True)

        self._limiter.wait()
        try:
            response = self._session.get(
                url, params=params, headers=self._headers, timeout=self._timeout
            )
        except Exception as exc:  # network errors from the HTTP library
            raise SourceUnavailableError(
                self.source, url, f"{type(exc).__name__}: {exc}", "a response"
            ) from exc
        fetched_at = datetime.now(UTC)
        text = response.text

        if response.status_code != 200:
            data = _try_json(text)
            if check is not None and data is not None:
                check(data)  # lets the API's own error (e.g. quota) surface with detail
            raise SourceUnavailableError(
                self.source, url, f"HTTP {response.status_code}: {text[:200]!r}", "HTTP 200"
            )
        if _looks_like_html(text, response.headers):
            raise SourceUnavailableError(
                self.source, url, f"HTML page: {_summary(text)!r}", "a JSON body"
            )
        data = _try_json(text)
        if data is None:
            raise SourceUnavailableError(self.source, url, _summary(text), "valid JSON")
        if check is not None:
            check(data)
        if self._cache is not None:
            self._cache.put_raw(self.source, url, params, text, fetched_at=fetched_at)
        return JsonResult(data, fetched_at, from_cache=False)


def _looks_like_html(text: str, headers: Mapping[str, str]) -> bool:
    content_type = {k.lower(): v for k, v in headers.items()}.get("content-type", "")
    return "html" in content_type.lower() or text.lstrip()[:1] == "<"


def _try_json(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return None


def _summary(text: str) -> str:
    return " ".join(text.split())[:160]
