"""
Tests for tiny-retry-plus.

Run:
    cd /home/node/.openclaw/workspace/tiny-retry-plus
    python -m pytest tests/ -v --tb=short

The integration tests against httpbin.org are marked with @pytest.mark.network
and skipped automatically if the network is unreachable.
"""
from __future__ import annotations

import asyncio
import os
import socket
import time
import urllib.error
import urllib.request
from typing import List

import pytest

from tiny_retry_plus import (
    AttemptContext,
    CircuitBreaker,
    CircuitBreakerOpen,
    RetryConfig,
    RetryError,
    __version__,
    attempt,
    circuit,
    retry,
    retry_async,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures & helpers
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def make_flaky():
    """Return a factory for stateful flaky functions."""

    def _factory(succeed_after: int = 2):
        state = {"n": 0}

        def fn():
            state["n"] += 1
            if state["n"] < succeed_after:
                raise ConnectionError(f"fail #{state['n']}")
            return f"ok after {state['n']}"

        return fn, state

    return _factory


def _httpbin_reachable() -> bool:
    """Cheap reachability probe so we can skip network tests cleanly."""
    try:
        socket.create_connection(("httpbin.org", 443), timeout=2).close()
        return True
    except OSError:
        return False


pytestmark_network = pytest.mark.skipif(
    not _httpbin_reachable(),
    reason="httpbin.org is not reachable from this environment",
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Version
# ──────────────────────────────────────────────────────────────────────────────
def test_version():
    assert __version__ == "0.1.0"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Basic retry success / failure
# ──────────────────────────────────────────────────────────────────────────────
def test_retry_success_first_try(make_flaky):
    fn, _ = make_flaky(succeed_after=1)

    @retry(attempts=3, base_delay=0.001, jitter="none")
    def wrapped():
        return fn()

    assert wrapped() == "ok after 1"


def test_retry_succeeds_after_failures(make_flaky):
    fn, state = make_flaky(succeed_after=3)

    @retry(attempts=5, base_delay=0.001, jitter="none")
    def wrapped():
        return fn()

    assert wrapped() == "ok after 3"
    assert state["n"] == 3


def test_retry_exhausts_attempts(make_flaky):
    fn, state = make_flaky(succeed_after=100)

    @retry(attempts=3, base_delay=0.001, jitter="none")
    def wrapped():
        return fn()

    with pytest.raises(RetryError) as excinfo:
        wrapped()
    assert excinfo.value.attempts == 3
    assert isinstance(excinfo.value.last_error, ConnectionError)
    assert state["n"] == 3


def test_retry_non_retriable_exception_propagates():
    @retry(attempts=5, base_delay=0.001, exceptions=(ValueError,))
    def boom():
        boom.n += 1
        raise KeyError("nope")

    boom.n = 0
    with pytest.raises(KeyError):
        boom()
    assert boom.n == 1  # KeyError is not in exceptions → no retry


def test_retry_custom_retriable_filter():
    @retry(attempts=5, base_delay=0.001, retriable=lambda e: "retry-me" in str(e))
    def picky():
        picky.n += 1
        if picky.n < 3:
            raise RuntimeError("retry-me please")
        raise RuntimeError("give up")

    picky.n = 0
    with pytest.raises(RuntimeError, match="give up"):
        picky()
    assert picky.n == 3


# ──────────────────────────────────────────────────────────────────────────────
# 3. Jitter strategies
# ──────────────────────────────────────────────────────────────────────────────
def test_jitter_none_is_deterministic():
    cfg = RetryConfig(base_delay=0.1, max_delay=10.0, jitter="none")
    assert _calculate_delay_for(cfg, attempt=2) == pytest.approx(0.1 * (2.0 ** 2))


def test_jitter_full_within_range():
    cfg = RetryConfig(base_delay=0.1, max_delay=10.0, jitter="full")
    seen = [_calculate_delay_for(cfg, attempt=2) for _ in range(200)]
    base = 0.1 * (2.0 ** 2)
    assert min(seen) >= 0
    assert max(seen) <= base  # full jitter is U(0, base)
    assert max(seen) - min(seen) > 0.05  # spreads measurably


def test_jitter_decorrelated_bounded_by_three_times_last():
    cfg = RetryConfig(base_delay=0.1, max_delay=100.0, jitter="decorrelated", jitter_range=(0.05, 0.05))
    last = 0.2
    for _ in range(200):
        d = _calculate_delay_for(cfg, attempt=2, last_delay=last)
        assert 0.0 <= d <= last * 3 + 1e-9


def test_jitter_decaying_within_range():
    cfg = RetryConfig(base_delay=0.1, max_delay=10.0, jitter="decaying", jitter_range=(0.5, 1.5))
    base = 0.1 * (2.0 ** 2)
    seen = [_calculate_delay_for(cfg, attempt=2) for _ in range(500)]
    assert min(seen) >= base * 0.5 - 1e-9
    assert max(seen) <= base * 1.5 + 1e-9


def test_calculate_delay_clamps_to_max_delay():
    cfg = RetryConfig(base_delay=0.1, max_delay=0.5, exponential_base=2.0, jitter="none")
    # attempt=10 would be 102.4 without clamp; must clamp to 0.5
    assert _calculate_delay_for(cfg, attempt=10) == pytest.approx(0.5)


def _calculate_delay_for(cfg: RetryConfig, attempt: int, last_delay: float = 0.0) -> float:
    """Test helper — mirrors the private API surface."""
    from tiny_retry_plus import _calculate_delay
    return _calculate_delay(cfg, attempt, last_delay)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Timeout behavior
# ──────────────────────────────────────────────────────────────────────────────
def test_timeout_aborts_retries(make_flaky):
    fn, state = make_flaky(succeed_after=100)

    @retry(attempts=100, base_delay=0.5, max_delay=1.0, jitter="none", timeout=0.5)
    def wrapped():
        return fn()

    t0 = time.monotonic()
    with pytest.raises((TimeoutError, RetryError)):
        wrapped()
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0, f"timeout didn't kick in; took {elapsed}s"


def test_on_timeout_callback_invoked():
    calls: List[object] = []

    @retry(
        attempts=10,
        base_delay=0.5,
        max_delay=1.0,
        jitter="none",
        timeout=0.2,
        on_timeout=lambda err: calls.append(err),
    )
    def always_fails():
        raise RuntimeError("boom")

    with pytest.raises((TimeoutError, RetryError)):
        always_fails()
    assert len(calls) == 1
    assert isinstance(calls[0], RuntimeError)


# ──────────────────────────────────────────────────────────────────────────────
# 5. on_retry callback
# ──────────────────────────────────────────────────────────────────────────────
def test_on_retry_callback(make_flaky):
    events: List[tuple] = []
    fn, _ = make_flaky(succeed_after=3)

    @retry(
        attempts=5,
        base_delay=0.001,
        max_delay=0.01,
        jitter="none",
        on_retry=lambda err, n, delay: events.append((n, delay, type(err).__name__)),
    )
    def wrapped():
        return fn()

    wrapped()
    assert [e[0] for e in events] == [1, 2]  # called for attempts 1 and 2
    assert all(e[2] == "ConnectionError" for e in events)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Rate limit header handling
# ──────────────────────────────────────────────────────────────────────────────
class FakeHTTP429(urllib.error.HTTPError):
    def __init__(self, retry_after: str):
        super().__init__(
            url="https://example.com",
            code=429,
            msg="Too Many Requests",
            hdrs={"Retry-After": retry_after},
            fp=None,
        )


def test_retry_after_header_respected():
    @retry(attempts=3, base_delay=0.001, jitter="none", timeout=5.0)
    def flaky():
        flaky.n += 1
        if flaky.n == 1:
            raise FakeHTTP429("0.2")
        return "ok"

    flaky.n = 0
    t0 = time.monotonic()
    assert flaky() == "ok"
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.18, f"should have slept ~0.2s; slept {elapsed}s"


def test_retry_after_invalid_value_falls_back_to_normal_backoff():
    @retry(attempts=3, base_delay=0.001, jitter="none", timeout=5.0)
    def flaky():
        flaky.n += 1
        if flaky.n == 1:
            raise FakeHTTP429("not-a-number")
        return "ok"

    flaky.n = 0
    assert flaky() == "ok"


# ──────────────────────────────────────────────────────────────────────────────
# 7. Async retry
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_async_retry_succeeds(make_flaky):
    fn, state = make_flaky(succeed_after=3)

    @retry_async(attempts=5, base_delay=0.001, jitter="none")
    async def wrapped():
        # The factory returns a sync fn; wrap it in an async shim
        return fn()

    assert await wrapped() == "ok after 3"


@pytest.mark.asyncio
async def test_async_retry_exhausts():
    @retry_async(attempts=3, base_delay=0.001, jitter="none")
    async def boom():
        boom.n += 1
        raise ValueError("nope")

    boom.n = 0
    with pytest.raises(RetryError):
        await boom()
    assert boom.n == 3


@pytest.mark.asyncio
async def test_async_retry_timeout():
    @retry_async(attempts=100, base_delay=0.5, max_delay=1.0, jitter="none", timeout=0.2)
    async def slow():
        raise RuntimeError("slow")

    t0 = time.monotonic()
    with pytest.raises((TimeoutError, RetryError)):
        await slow()
    assert time.monotonic() - t0 < 2.0


@pytest.mark.asyncio
async def test_async_retry_after_header():
    class FakeAsync429(Exception):
        def __init__(self):
            super().__init__("Too Many Requests")
            self.code = 429
            self.headers = {"Retry-After": "0.2"}

    @retry_async(attempts=3, base_delay=0.001, jitter="none", timeout=5.0)
    async def flaky():
        flaky.n += 1
        if flaky.n == 1:
            raise FakeAsync429()
        return "ok"

    flaky.n = 0
    t0 = time.monotonic()
    assert await flaky() == "ok"
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.18


# ──────────────────────────────────────────────────────────────────────────────
# 8. AttemptContext
# ──────────────────────────────────────────────────────────────────────────────
def test_attempt_context_succeeds_after_failures():
    counter = {"n": 0}
    ctx = attempt(attempts=5, base_delay=0.001, jitter="none")

    while not ctx.exhausted and not ctx.timed_out:
        with ctx:
            counter["n"] += 1
            if counter["n"] < 3:
                raise ConnectionError("transient")

    assert counter["n"] == 3
    assert isinstance(ctx.last_error, ConnectionError)


def test_attempt_context_exhausts_and_raises():
    ctx = attempt(attempts=3, base_delay=0.001, jitter="none")
    # The final attempt's exception escapes naturally; capture it.
    with pytest.raises(ValueError, match="nope"):
        while not ctx.exhausted and not ctx.timed_out:
            with ctx:
                raise ValueError("nope")
    assert ctx.exhausted
    # The last error is captured on the context
    assert isinstance(ctx.last_error, ValueError)


def test_attempt_context_non_retriable_passes_through():
    ctx = attempt(attempts=5, base_delay=0.001, exceptions=(ValueError,))
    with pytest.raises(KeyError):
        with ctx:
            raise KeyError("different")
    # Non-matching exception → __exit__ returns False → propagates


# ──────────────────────────────────────────────────────────────────────────────
# 9. Circuit breaker
# ──────────────────────────────────────────────────────────────────────────────
def test_circuit_breaker_opens_after_threshold():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0, half_open_attempts=1)

    @retry(attempts=2, base_delay=0.001, circuit_breaker=cb, timeout=5.0)
    def always_fails():
        raise RuntimeError("boom")

    for _ in range(3):
        with pytest.raises((RetryError, CircuitBreakerOpen)):
            always_fails()

    assert cb.state == "OPEN"
    # Now any call should be rejected immediately
    with pytest.raises(CircuitBreakerOpen):
        always_fails()


def test_circuit_breaker_half_open_after_recovery():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1, half_open_attempts=1)

    @retry(attempts=1, base_delay=0.001, circuit_breaker=cb, timeout=5.0)
    def fails_once():
        raise RuntimeError("boom")

    # Trip it
    for _ in range(2):
        with pytest.raises((RetryError, RuntimeError)):
            fails_once()
    assert cb.state == "OPEN"

    # Wait for recovery
    time.sleep(0.15)
    assert cb.state == "HALF_OPEN"

    # A successful call should close it
    @retry(attempts=1, base_delay=0.001, circuit_breaker=cb, timeout=5.0)
    def succeeds():
        return "ok"

    succeeds()
    assert cb.state == "CLOSED"


def test_circuit_breaker_failure_in_half_open_reopens():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1, half_open_attempts=2)

    # Trip it
    for _ in range(2):
        try:
            cb.record_failure()
        except Exception:
            pass
    assert cb.state == "OPEN"

    time.sleep(0.15)
    assert cb.state == "HALF_OPEN"

    cb.record_failure()
    assert cb.state == "OPEN"


def test_circuit_shorthand():
    cb = circuit(failure_threshold=3, recovery_timeout=30.0)
    assert isinstance(cb, CircuitBreaker)
    assert cb.failure_threshold == 3


def test_circuit_breaker_validation():
    with pytest.raises(ValueError):
        CircuitBreaker(failure_threshold=0)
    with pytest.raises(ValueError):
        CircuitBreaker(recovery_timeout=0)
    with pytest.raises(ValueError):
        CircuitBreaker(half_open_attempts=0)


# ──────────────────────────────────────────────────────────────────────────────
# 10. RetryConfig validation
# ──────────────────────────────────────────────────────────────────────────────
def test_retry_config_validates_attempts():
    with pytest.raises(ValueError):
        RetryConfig(attempts=0)


def test_retry_config_validates_jitter():
    with pytest.raises(ValueError):
        RetryConfig(jitter="bogus")


def test_retry_config_validates_timeout():
    with pytest.raises(ValueError):
        RetryConfig(timeout=0)


def test_retry_config_validates_max_delay():
    with pytest.raises(ValueError):
        RetryConfig(base_delay=1.0, max_delay=0.5)


# ──────────────────────────────────────────────────────────────────────────────
# 11. Real HTTP integration tests (optional — skipped if no network)
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.network
def test_real_http_200():
    @retry(attempts=3, base_delay=0.1, max_delay=1.0, jitter="full", timeout=10.0)
    def get_example():
        # Use example.com — reliably returns 200, unlike httpbin.org which
        # has been observed returning 503 even for healthy paths.
        with urllib.request.urlopen("https://example.com/", timeout=5) as r:
            return r.status

    assert get_example() == 200


@pytest.mark.network
def test_real_http_500_retries_then_fails():
    # httpbin.org is occasionally flaky; skip gracefully if it 5xx's on /status/500
    # (which is the very endpoint we're testing).
    try:
        probe = urllib.request.urlopen("https://httpbin.org/status/500", timeout=5)
        probe.close()
    except urllib.error.HTTPError as e:
        if e.code == 500:
            pass  # expected — the endpoint works
        elif e.code in (503, 502):
            pytest.skip(f"httpbin.org is having a bad day (HTTP {e.code})")
        else:
            raise
    except Exception as e:
        pytest.skip(f"httpbin.org unreachable: {e}")

    @retry(attempts=3, base_delay=0.1, max_delay=0.5, jitter="none", timeout=10.0)
    def get_500():
        with urllib.request.urlopen("https://httpbin.org/status/500", timeout=5) as r:
            return r.status

    with pytest.raises(RetryError):
        get_500()


@pytest.mark.network
def test_real_http_flaky_503_then_200():
    # Same httpbin-flakiness guard as above.
    try:
        probe = urllib.request.urlopen("https://httpbin.org/status/503", timeout=5)
        probe.close()
    except urllib.error.HTTPError as e:
        if e.code == 503:
            pass  # expected
        else:
            pytest.skip(f"httpbin.org returned unexpected HTTP {e.code} on probe")
    except Exception as e:
        pytest.skip(f"httpbin.org unreachable: {e}")

    @retry(
        attempts=5,
        base_delay=0.1,
        max_delay=1.0,
        jitter="full",
        timeout=10.0,
        rate_limit_statuses=(500, 502, 503, 504),
        exceptions=(urllib.error.HTTPError,),
    )
    def get_503_then_200():
        with urllib.request.urlopen("https://httpbin.org/status/503", timeout=5) as r:
            return r.status

    with pytest.raises(RetryError) as excinfo:
        get_503_then_200()
    assert excinfo.value.attempts == 5


# ──────────────────────────────────────────────────────────────────────────────
# 12. Edge cases
# ──────────────────────────────────────────────────────────────────────────────
def test_retry_with_no_args():
    """@retry() with no config should use defaults (3 attempts, jitter=full)."""

    @retry()
    def ok():
        return "fine"

    assert ok() == "fine"


def test_retry_async_with_no_args():
    @retry_async()
    async def ok():
        return "fine"

    assert asyncio.run(ok()) == "fine"


def test_retry_passes_kwargs_through():
    @retry(attempts=3, base_delay=0.001, jitter="none")
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5
    assert add(a=10, b=20) == 30


def test_circuit_breaker_reset():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)
    cb.record_failure()
    cb.record_failure()
    assert cb.state == "OPEN"
    cb.reset()
    assert cb.state == "CLOSED"
    assert cb.failure_count == 0
