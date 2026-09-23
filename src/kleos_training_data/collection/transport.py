from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any

from kleos_training_data.errors import CaptureError, MissingDependencyError
from kleos_training_data.logging_utils import SafeSecret, get_logger

logger = get_logger(__name__)

RETRYABLE_STATUSES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})

NEVER_RETRY: frozenset[int] = frozenset({400, 401, 403, 404, 405, 422})


def require_httpx() -> Any:
    try:
        import httpx
    except ImportError as exc:
        raise MissingDependencyError(
            "httpx", extra="collect", purpose="capture from a real backend"
        ) from exc
    return httpx


@dataclass
class RetryPolicy:
    max_attempts: int = 4
    base_delay: float = 0.5
    max_delay: float = 20.0
    budget: int = 20
    _spent: int = field(default=0, init=False)

    @property
    def budget_remaining(self) -> int:
        return max(0, self.budget - self._spent)

    def should_retry(self, *, attempt: int, status: int | None) -> bool:
        if attempt >= self.max_attempts:
            return False
        if not self.budget_remaining:
            return False
        if status is None:
            return True
        if status in NEVER_RETRY:
            return False
        return status in RETRYABLE_STATUSES

    def delay_for(self, attempt: int, *, retry_after: str | None = None) -> float:
        if retry_after:
            explicit = _parse_retry_after(retry_after)
            if explicit is not None:
                return min(explicit, self.max_delay)
        ceiling = min(self.base_delay * (2**attempt), self.max_delay)
        return random.uniform(0, ceiling)

    def spend(self) -> None:
        self._spent += 1


def _parse_retry_after(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    from datetime import UTC, datetime

    return max(0.0, (when - datetime.now(UTC)).total_seconds())


@dataclass
class RateLimiter:
    requests_per_minute: float = 20.0
    burst: int = 5
    _tokens: float = field(default=0.0, init=False)
    _last: float = field(default_factory=time.monotonic, init=False)

    def __post_init__(self) -> None:
        self._tokens = float(self.burst)

    def acquire(self) -> float:
        rate = self.requests_per_minute / 60.0
        waited = 0.0
        while True:
            now = time.monotonic()
            self._tokens = min(self.burst, self._tokens + (now - self._last) * rate)
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return waited
            sleep_for = (1.0 - self._tokens) / rate if rate else 1.0
            time.sleep(sleep_for)
            waited += sleep_for


@dataclass
class TransportConfig:
    base_url: str
    token: SafeSecret | None = None
    timeout_seconds: float = 60.0
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    rate_limiter: RateLimiter = field(default_factory=RateLimiter)
    user_agent: str = "kleos-training-data/0.1"


@dataclass
class Response:
    status: int
    headers: dict[str, str]
    text: str
    latency_ms: int
    attempts: int
    retried_on: list[int]
    request_id: str


class HttpTransport:
    def __init__(self, config: TransportConfig) -> None:
        self.config = config
        self._client: Any | None = None

    def _headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": self.config.user_agent,
            "Accept": "text/event-stream, application/json",
        }
        if self.config.token is not None and self.config.token.is_set:
            headers["Authorization"] = f"Bearer {self.config.token.reveal()}"
        return headers

    def __enter__(self) -> HttpTransport:
        httpx = require_httpx()
        self._client = httpx.Client(
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
            follow_redirects=False,
        )
        return self

    def __exit__(self, *exc: object) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def request(
        self,
        method: str,
        path: str,
        *,
        request_id: str,
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Response:
        httpx = require_httpx()
        if self._client is None:
            raise CaptureError("HttpTransport must be used as a context manager.")

        headers = {**self._headers(), "X-Request-Id": request_id}
        started = time.perf_counter()
        retried_on: list[int] = []

        for attempt in range(self.config.retry.max_attempts):
            waited = self.config.rate_limiter.acquire()
            status: int | None = None
            try:
                response = self._client.request(
                    method, path, headers=headers, data=data, params=params
                )
                status = response.status_code
                if 200 <= status < 300:
                    return Response(
                        status=status,
                        headers=dict(response.headers),
                        text=response.text,
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        attempts=attempt + 1,
                        retried_on=retried_on,
                        request_id=request_id,
                    )
                retry_after = response.headers.get("Retry-After")
            except (httpx.TimeoutException, httpx.TransportError):
                retry_after = None

            logger.warning(
                "request_id=%s attempt=%d status=%s waited_ms=%d budget_left=%d",
                request_id,
                attempt + 1,
                status,
                int(waited * 1000),
                self.config.retry.budget_remaining,
            )

            if not self.config.retry.should_retry(attempt=attempt + 1, status=status):
                break
            if status is not None:
                retried_on.append(status)
            self.config.retry.spend()
            time.sleep(self.config.retry.delay_for(attempt, retry_after=retry_after))

        raise CaptureError(
            f"Request failed after {len(retried_on) + 1} attempt(s).",
            details={
                "request_id": request_id,
                "last_status": str(status),
                "retried_on": ", ".join(str(s) for s in retried_on) or "(none)",
                "budget_remaining": str(self.config.retry.budget_remaining),
            },
            suggestions=[
                "A 4xx other than 408/429 is not retried: the request or the "
                "credential is wrong, and repeating it only delays the real error.",
                "If the budget is exhausted, the backend is degraded — stop rather than piling on.",
            ],
        )

    def stream_lines(
        self, method: str, path: str, *, request_id: str, data: dict[str, Any] | None = None
    ) -> tuple[list[str], Response]:
        response = self.request(method, path, request_id=request_id, data=data)
        return response.text.splitlines(), response
