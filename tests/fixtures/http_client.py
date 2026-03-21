"""HTTP client with retry logic and response caching."""
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HttpResponse:
    """Represents an HTTP response."""
    status_code: int
    body: str
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> Any:
        import json
        return json.loads(self.body)


class RetryConfig:
    """Configuration for HTTP retry behavior."""

    def __init__(self, max_retries: int = 3, backoff_factor: float = 0.5):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def get_delay(self, attempt: int) -> float:
        return self.backoff_factor * (2 ** attempt)


class ResponseCache:
    """Simple in-memory cache for HTTP responses."""

    def __init__(self, ttl_seconds: int = 300):
        self._cache: dict[str, tuple[float, HttpResponse]] = {}
        self._ttl = ttl_seconds

    def get(self, url: str) -> HttpResponse | None:
        entry = self._cache.get(url)
        if entry is None:
            return None
        timestamp, response = entry
        if time.time() - timestamp > self._ttl:
            del self._cache[url]
            return None
        return response

    def put(self, url: str, response: HttpResponse) -> None:
        self._cache[url] = (time.time(), response)

    def invalidate(self, url: str) -> bool:
        return self._cache.pop(url, None) is not None

    def clear(self) -> None:
        self._cache.clear()
