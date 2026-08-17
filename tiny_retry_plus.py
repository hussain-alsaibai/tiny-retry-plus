"""
tiny-retry-plus — Zero-dependency retry with jitter, adaptive backoff, and circuit-breaker integration.

A stdlib-only retry library that builds on the spirit of `tiny-retry` and adds:
  * Full / decorrelated / decaying jitter
  * Total-timeout + per-attempt timeout awareness
  * Adaptive backoff (exponential with cap)
  * Circuit-breaker integration
  * Rate-limit header awareness (e.g. Retry-After)
  * Sync + async decorators
  * Manual `AttemptContext` for fine-grained control

MIT License | https://github.com/hussain-alsaibai/tiny-retry-plus
"""

from __future__ import annotations

import asyncio
import functools
import http.client
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, Type, TypeVar, Union

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "retry",
    "retry_async",
    "RetryConfig",
    "RetryError",
    "CircuitBreakerOpen",
    "AttemptContext",
    "CircuitBreaker",
]


T = TypeVar("T")


# ──────────────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────────────
class RetryError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, message: str, attempts: int, last_error: Optional[BaseException]):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


class CircuitBreakerOpen(Exception):
    """Raised when the circuit breaker is open and the call is rejected."""

    pass


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    attempts: int = 3
    """Max total attempts (including the first try). Must be >= 1."""

    timeout: float = 30.0
    """Overall timeout in seconds across all attempts. Use math.inf to disable."""

    base_delay: float = 0.1
    """Initial delay in seconds before the second attempt."""

    max_delay: float = 30.0
    """Cap on any single delay, in seconds."""

    exponential_base: float = 2.0
    """Multiplier per attempt: delay_n = base_delay * (exponential_base ** n)."""

    jitter: str = "full"
    """One of: 'none', 'full', 'decorrelated', 'decaying'."""

    jitter_range: Tuple[float, float] = (0.5, 1.5)
    """Multiplier range used by the 'decaying' jitter strategy."""

    exceptions: Tuple[Type[BaseException], ...] = (Exception,)
    """Exception classes that trigger a retry."""

    retriable: Optional[Callable[[BaseException], bool]] = None
    """Optional custom filter — return False to stop retrying immediately."""

    on_retry: Optional[Callable[[BaseException, int, float], None]] = None
    """Callback invoked after each failed attempt. Args: (error, attempt_number, delay)."""

    on_timeout: Optional[Callable[[Optional[BaseException]], None]] = None
    """Callback invoked when the overall timeout is reached."""

    rate_limit_header: Optional[str] = None
    """HTTP header to consult for rate limit hints (default: 'Retry-After')."""

    rate_limit_statuses: Tuple[int, ...] = (429, 503)
    """HTTP status codes treated as 'rate limited' (default: 429, 503)."""

    circuit_breaker: Optional[Any] = None
    """Optional injected :class:`CircuitBreaker` instance."""

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("attempts must be >= 1")
        if self.timeout <= 0:
            raise ValueError("timeout must be > 0")
        if self.base_delay < 0:
            raise ValueError("base_delay must be >= 0")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")
        if self.jitter not in ("none", "full", "decorrelated", "decaying"):
            raise ValueError(f"unknown jitter mode: {self.jitter!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Backoff / jitter primitives
# ──────────────────────────────────────────────────────────────────────────────
def _exponential_delay(attempt: int, base: float, cap: float, exp_base: float) -> float:
    """Pure exponential backoff for the given attempt index (0-based)."""
    delay = base * (exp_base ** attempt)
    if delay > cap:
        return cap
    return delay


def _jitter_none(delay: float, last_delay: float, rng: Tuple[float, float]) -> float:
    return delay


def _jitter_full(delay: float, last_delay: float, rng: Tuple[float, float]) -> float:
    # AWS-style full jitter: delay * U(0, 1)
    return delay * random.random()


def _jitter_decorrelated(delay: float, last_delay: float, rng: Tuple[float, float]) -> float:
    # AWS decorrelated jitter:
    #   sleep_n = min(cap, random_between(base, sleep_{n-1} * 3))
    base = rng[0] if rng and rng[0] > 0 else 0.0
    if last_delay <= 0:
        last_delay = delay
    upper = last_delay * 3
    if base >= upper:
        return upper
    return base + random.random() * (upper - base)


def _jitter_decaying(delay: float, last_delay: float, rng: Tuple[float, float]) -> float:
    low, high = rng
    if high < low:
        low, high = high, low
    factor = low + random.random() * (high - low)
    return delay * factor


def _calculate_delay(config: RetryConfig, attempt: int, last_delay: float = 0.0) -> float:
    """Compute the next delay using config jitter strategy."""
    delay = _exponential_delay(
        attempt, config.base_delay, config.max_delay, config.exponential_base
    )
    rng = config.jitter_range
    if config.jitter == "full":
        out = _jitter_full(delay, last_delay, rng)
    elif config.jitter == "decorrelated":
        out = _jitter_decorrelated(delay, last_delay, rng)
    elif config.jitter == "decaying":
        out = _jitter_decaying(delay, last_delay, rng)
    else:
        out = _jitter_none(delay, last_delay, rng)
    # Always clamp to max_delay for sanity
    return min(max(out, 0.0), config.max_delay)


# ──────────────────────────────────────────────────────────────────────────────
# Sync @retry decorator
# ──────────────────────────────────────────────────────────────────────────────
def retry(
    config: Optional[RetryConfig] = None,
    **kwargs: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for synchronous retry with configurable backoff and jitter.

    Either pass a pre-built :class:`RetryConfig` or use keyword arguments that
    map directly to its fields::

        @retry(attempts=5, base_delay=0.5, jitter="full", timeout=10)
        def fetch(url): ...

        @retry(RetryConfig(attempts=3, exceptions=(urllib.error.URLError,)))
        def fetch(url): ...
    """
    cfg = config if config is not None else RetryConfig(**kwargs)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kw: Any) -> T:
            # Circuit breaker check
            if cfg.circuit_breaker is not None and not cfg.circuit_breaker.can_execute():
                raise CircuitBreakerOpen(
                    f"Circuit breaker is open for {func.__name__!r}"
                )

            start = time.monotonic()
            last_error: Optional[BaseException] = None
            last_delay = 0.0
            skip_backoff = False  # when True, next iteration skips the exponential backoff

            for attempt in range(cfg.attempts):
                elapsed = time.monotonic() - start
                if elapsed >= cfg.timeout:
                    if cfg.on_timeout is not None:
                        cfg.on_timeout(last_error)
                    raise TimeoutError(
                        f"Overall timeout ({cfg.timeout}s) exceeded after {attempt} attempt(s)"
                    )

                # Sleep before retrying (skip on first attempt; also skip when
                # the previous attempt was rate-limit-honored — we already slept
                # the server's requested Retry-After, no need to stack more).
                if attempt > 0 and not skip_backoff:
                    delay = _calculate_delay(cfg, attempt, last_delay)
                    last_delay = delay
                    if elapsed + delay >= cfg.timeout:
                        if cfg.on_timeout is not None:
                            cfg.on_timeout(last_error)
                        raise TimeoutError(
                            f"Timeout ({cfg.timeout}s) would be exceeded by next retry delay"
                        )
                    time.sleep(delay)
                skip_backoff = False  # only applies to the immediately-following attempt

                try:
                    result = func(*args, **kw)
                except cfg.exceptions as e:  # type: ignore[misc]
                    last_error = e
                    if cfg.retriable is not None and not cfg.retriable(e):
                        raise

                    # Rate limit hint from HTTP headers
                    retry_after = _extract_retry_after(e, cfg)
                    if retry_after > 0:
                        if elapsed + retry_after >= cfg.timeout:
                            if cfg.on_timeout is not None:
                                cfg.on_timeout(last_error)
                            raise TimeoutError(
                                f"Retry-After ({retry_after}s) would exceed overall timeout"
                            )
                        time.sleep(retry_after)
                        last_delay = retry_after
                        # Honor the server's exact wait — don't stack an
                        # exponential backoff on top of it.
                        skip_backoff = True
                        continue

                    # Notify circuit breaker of failure
                    if cfg.circuit_breaker is not None:
                        cfg.circuit_breaker.record_failure(e)

                    if attempt == cfg.attempts - 1:
                        raise RetryError(
                            f"All {cfg.attempts} attempts failed for {func.__name__!r}",
                            cfg.attempts,
                            last_error,
                        ) from last_error

                    if cfg.on_retry is not None:
                        cfg.on_retry(e, attempt + 1, last_delay)
                    continue
                else:
                    # Success — notify circuit breaker
                    if cfg.circuit_breaker is not None:
                        cfg.circuit_breaker.record_success()
                    return result

            # Defensive — should be unreachable
            raise RetryError(
                f"All {cfg.attempts} attempts failed for {func.__name__!r}",
                cfg.attempts,
                last_error,
            ) from last_error

        return wrapper

    return decorator


def _extract_retry_after(error: BaseException, cfg: RetryConfig) -> float:
    """Best-effort extraction of a Retry-After-style hint from an HTTP error."""
    header_name = cfg.rate_limit_header or "Retry-After"
    # urllib.error.HTTPError exposes .code and .headers
    if isinstance(error, urllib.error.HTTPError):
        if error.code in cfg.rate_limit_statuses:
            value = error.headers.get(header_name) if error.headers else None
            if value:
                try:
                    return max(0.0, float(value))
                except (TypeError, ValueError):
                    # Some servers send HTTP-date; we ignore that for simplicity.
                    pass
    # http.client.HTTPResponse, requests-like, etc. — best effort
    headers = getattr(error, "headers", None)
    code = getattr(error, "code", None) or getattr(error, "status", None) or getattr(error, "status_code", None)
    if headers is not None and code in cfg.rate_limit_statuses:
        value = headers.get(header_name) if hasattr(headers, "get") else None
        if value:
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                pass
    return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Async @retry_async decorator
# ──────────────────────────────────────────────────────────────────────────────
def retry_async(
    config: Optional[RetryConfig] = None,
    **kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Async equivalent of :func:`retry`."""
    cfg = config if config is not None else RetryConfig(**kwargs)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kw: Any) -> Any:
            if cfg.circuit_breaker is not None and not cfg.circuit_breaker.can_execute():
                raise CircuitBreakerOpen(
                    f"Circuit breaker is open for {func.__name__!r}"
                )

            start = time.monotonic()
            last_error: Optional[BaseException] = None
            last_delay = 0.0
            skip_backoff = False

            for attempt in range(cfg.attempts):
                elapsed = time.monotonic() - start
                if elapsed >= cfg.timeout:
                    if cfg.on_timeout is not None:
                        cfg.on_timeout(last_error)
                    raise TimeoutError(
                        f"Overall timeout ({cfg.timeout}s) exceeded after {attempt} attempt(s)"
                    )

                if attempt > 0 and not skip_backoff:
                    delay = _calculate_delay(cfg, attempt, last_delay)
                    last_delay = delay
                    if elapsed + delay >= cfg.timeout:
                        if cfg.on_timeout is not None:
                            cfg.on_timeout(last_error)
                        raise TimeoutError(
                            f"Timeout ({cfg.timeout}s) would be exceeded by next retry delay"
                        )
                    await asyncio.sleep(delay)
                skip_backoff = False

                try:
                    result = await func(*args, **kw)
                except cfg.exceptions as e:  # type: ignore[misc]
                    last_error = e
                    if cfg.retriable is not None and not cfg.retriable(e):
                        raise

                    retry_after = _extract_retry_after(e, cfg)
                    if retry_after > 0:
                        if elapsed + retry_after >= cfg.timeout:
                            if cfg.on_timeout is not None:
                                cfg.on_timeout(last_error)
                            raise TimeoutError(
                                f"Retry-After ({retry_after}s) would exceed overall timeout"
                            )
                        await asyncio.sleep(retry_after)
                        last_delay = retry_after
                        skip_backoff = True
                        continue

                    if cfg.circuit_breaker is not None:
                        cfg.circuit_breaker.record_failure(e)

                    if attempt == cfg.attempts - 1:
                        raise RetryError(
                            f"All {cfg.attempts} attempts failed for {func.__name__!r}",
                            cfg.attempts,
                            last_error,
                        ) from last_error

                    if cfg.on_retry is not None:
                        cfg.on_retry(e, attempt + 1, last_delay)
                    continue
                else:
                    if cfg.circuit_breaker is not None:
                        cfg.circuit_breaker.record_success()
                    return result

            raise RetryError(
                f"All {cfg.attempts} attempts failed for {func.__name__!r}",
                cfg.attempts,
                last_error,
            ) from last_error

        return wrapper

    return decorator


# ──────────────────────────────────────────────────────────────────────────────
# Manual AttemptContext
# ──────────────────────────────────────────────────────────────────────────────
class AttemptContext:
    """Context manager for manual retry loops.

    Usage::

        ctx = AttemptContext(attempts=5, base_delay=0.2, jitter="full")
        while not ctx.exhausted and not ctx.timed_out:
            with ctx:
                do_the_thing()
        if ctx.last_error:
            raise ctx.last_error
    """

    def __init__(
        self,
        config: Optional[RetryConfig] = None,
        **kwargs: Any,
    ) -> None:
        self.config = config if config is not None else RetryConfig(**kwargs)
        self.attempt = 0
        self._start = 0.0
        self._succeeded = False
        self.last_error: Optional[BaseException] = None

    @property
    def exhausted(self) -> bool:
        # Once we succeed, we're done — mark as exhausted so the user's
        # `while not ctx.exhausted` loop naturally terminates.
        return self._succeeded or self.attempt >= self.config.attempts

    @property
    def succeeded(self) -> bool:
        return self._succeeded

    @property
    def timed_out(self) -> bool:
        if self._start == 0.0:
            return False
        return (time.monotonic() - self._start) >= self.config.timeout

    def __enter__(self) -> "AttemptContext":
        # Each entry counts as a try, whether or not it succeeds. The loop
        # # exits naturally when `ctx.exhausted` becomes True.
        self.attempt += 1
        if self._start == 0.0:
            self._start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            self._succeeded = True  # success — mark so `exhausted` becomes True
            return False  # success — let it propagate naturally
        self.last_error = exc_val
        if not isinstance(exc_val, self.config.exceptions):
            return False  # non-retriable — let it propagate
        if self.config.retriable is not None and not self.config.retriable(exc_val):
            return False  # custom filter rejected it
        if self.attempt >= self.config.attempts:
            return False  # out of budget — let the exception escape
        elapsed = time.monotonic() - self._start
        if elapsed >= self.config.timeout:
            return False
        delay = _calculate_delay(self.config, self.attempt, 0.0)
        if elapsed + delay >= self.config.timeout:
            return False
        time.sleep(delay)
        if self.config.on_retry is not None:
            self.config.on_retry(exc_val, self.attempt + 1, delay)
        return True  # suppress — caller will loop again


# Convenience alias — `attempt` matches the context manager name in tests/examples
attempt = AttemptContext


# ──────────────────────────────────────────────────────────────────────────────
# Circuit Breaker (lightweight, no external state)
# ──────────────────────────────────────────────────────────────────────────────
class CircuitBreaker:
    """Minimal three-state circuit breaker (CLOSED → OPEN → HALF_OPEN).

    Designed to be passed into :class:`RetryConfig` via ``circuit_breaker=``.
    Thread-safety is intentionally NOT provided; if you need it, wrap calls in
    your own lock — this keeps the implementation zero-dep and tiny.
    """

    STATE_CLOSED = "CLOSED"
    STATE_OPEN = "OPEN"
    STATE_HALF_OPEN = "HALF_OPEN"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_attempts: int = 3,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be > 0")
        if half_open_attempts < 1:
            raise ValueError("half_open_attempts must be >= 1")
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_attempts = half_open_attempts

        self._failures = 0
        self._last_failure_time = 0.0
        self._state = self.STATE_CLOSED
        self._half_open_successes = 0
        self._half_open_in_flight = 0

    # ── state queries ────────────────────────────────────────────────────────
    def _live_state(self) -> str:
        """Internal: compute live state, transitioning OPEN → HALF_OPEN
        if the recovery timeout has elapsed since the last failure."""
        if self._state == self.STATE_OPEN:
            if (
                self.recovery_timeout > 0
                and time.monotonic() - self._last_failure_time >= self.recovery_timeout
            ):
                # Commit the transition so record_* methods see the right state
                self._state = self.STATE_HALF_OPEN
                self._half_open_successes = 0
                self._half_open_in_flight = 0
        return self._state

    @property
    def state(self) -> str:
        """Live state, accounting for elapsed recovery_timeout."""
        return self._live_state()

    @property
    def failure_count(self) -> int:
        return self._failures

    # ── decision ─────────────────────────────────────────────────────────────
    def can_execute(self) -> bool:
        live = self._live_state()
        if live == self.STATE_CLOSED:
            return True
        if live == self.STATE_HALF_OPEN:
            if self._half_open_in_flight < self.half_open_attempts:
                self._half_open_in_flight += 1
                return True
            return False
        # OPEN
        return False

    # ── bookkeeping ──────────────────────────────────────────────────────────
    def record_success(self) -> None:
        # Trigger any OPEN→HALF_OPEN transition first
        self._live_state()
        self._failures = 0
        if self._state == self.STATE_HALF_OPEN:
            self._half_open_successes += 1
            self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
            if self._half_open_successes >= self.half_open_attempts:
                self._state = self.STATE_CLOSED
                self._half_open_successes = 0
                self._half_open_in_flight = 0
        elif self._state == self.STATE_OPEN:
            # We shouldn't normally get here — but be defensive.
            pass

    def record_failure(self, error: Optional[BaseException] = None) -> None:
        self._failures += 1
        self._last_failure_time = time.monotonic()
        if self._state == self.STATE_HALF_OPEN:
            # Any failure in HALF_OPEN reopens the breaker
            self._state = self.STATE_OPEN
            self._half_open_successes = 0
            self._half_open_in_flight = 0
            return
        if self._failures >= self.failure_threshold:
            self._state = self.STATE_OPEN

    def reset(self) -> None:
        """Force the breaker back to CLOSED."""
        self._failures = 0
        self._state = self.STATE_CLOSED
        self._half_open_successes = 0
        self._half_open_in_flight = 0
        self._last_failure_time = 0.0

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(state={self.state!r}, failures={self._failures}, "
            f"threshold={self.failure_threshold}, recovery_timeout={self.recovery_timeout})"
        )


# Convenience constructor accepting a config-friendly name
def circuit(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    half_open_attempts: int = 3,
) -> CircuitBreaker:
    """Shorthand to build a :class:`CircuitBreaker`."""
    return CircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        half_open_attempts=half_open_attempts,
    )
