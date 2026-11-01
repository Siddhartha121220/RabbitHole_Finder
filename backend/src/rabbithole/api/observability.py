"""JSON logs with a request id and latency per request; in-process token-bucket rate limit."""

import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request, Response

log = logging.getLogger("rabbithole.api")

# Fields a log call may attach via `extra=` that should appear in the JSON line.
EXTRA_FIELDS = ("request_id", "method", "path", "status", "latency_ms", "event")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        out |= {k: getattr(record, k) for k in EXTRA_FIELDS if hasattr(record, k)}
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger("rabbithole")
    root.handlers[:] = [handler]
    root.setLevel(level)
    root.propagate = False


async def request_context(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Tag each request with an id (taken from X-Request-ID if the caller sent one)."""
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    request.state.request_id = rid
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
    finally:
        log.info(
            "request",
            extra={
                "request_id": rid,
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            },
        )
    response.headers["X-Request-ID"] = rid
    return response


class RateLimiter:
    """Token bucket per client key: `per_minute` tokens, refilled continuously.
    ponytail: in-process only; one API instance. Move to Redis if it ever scales out."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.capacity = float(per_minute)
        self.rate = per_minute / 60.0
        self.clock = clock
        self.buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last time)

    def allow(self, key: str) -> bool:
        now = self.clock()
        tokens, last = self.buckets.get(key, (self.capacity, now))
        tokens = min(self.capacity, tokens + (now - last) * self.rate)
        allowed = tokens >= 1
        self.buckets[key] = (tokens - 1 if allowed else tokens, now)
        return allowed
