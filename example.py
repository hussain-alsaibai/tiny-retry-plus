"""
Example usage of tiny-retry-plus.

Run with:
    python example.py

These examples intentionally don't perform real network IO so they're
fully reproducible offline. See `tests/test_tiny_retry_plus.py` for
live httpbin.org integration tests.
"""
from __future__ import annotations

import asyncio
import time
import urllib.error

from tiny_retry_plus import (
    CircuitBreaker,
    CircuitBreakerOpen,
    RetryConfig,
    RetryError,
    attempt,
    retry,
    retry_async,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Basic retry
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("1. Basic retry — 3 attempts, fixed 100ms delay")
print("=" * 70)

flaky_counter = {"n": 0}


@retry(attempts=3, base_delay=0.1, max_delay=0.5, jitter="none", timeout=2.0)
def flaky():
    flaky_counter["n"] += 1
    if flaky_counter["n"] < 2:
        raise RuntimeError("nope")
    return f"succeeded after {flaky_counter['n']} attempts"


print(flaky())


# ──────────────────────────────────────────────────────────────────────────────
# 2. Exponential backoff with full jitter
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("2. Exponential backoff + full jitter (AWS-style)")
print("=" * 70)


@retry(attempts=5, base_delay=0.05, max_delay=1.0, jitter="full")
def eventually():
    eventually.calls += 1
    if eventually.calls < 4:
        raise ConnectionError("transient")
    return "ok"


eventually.calls = 0
t0 = time.monotonic()
result = eventually()
print(f"{result}  (took {time.monotonic() - t0:.3f}s)")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Async retry — httpx-shaped usage
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("3. Async retry")
print("=" * 70)


class FakeResponse:
    def __init__(self, status: int, body: str = ""):
        self.status = status
        self.body = body

    async def json(self):
        return {"status": self.status, "body": self.body}


async def fake_http_get(url: str) -> FakeResponse:
    fake_http_get.calls += 1
    if fake_http_get.calls < 3:
        raise ConnectionError(f"simulated failure for {url}")
    return FakeResponse(200, "hello")


fake_http_get.calls = 0


@retry_async(attempts=4, base_delay=0.05, jitter="decaying")
async def fetch_json(url: str) -> dict:
    resp = await fake_http_get(url)
    return await resp.json()


async def main_async():
    return await fetch_json("https://api.example.com/data")


print(asyncio.run(main_async()))


# ──────────────────────────────────────────────────────────────────────────────
# 4. Circuit breaker integration
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("4. Circuit breaker integration")
print("=" * 70)

breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.5, half_open_attempts=1)


@retry(attempts=3, base_delay=0.01, circuit_breaker=breaker, timeout=1.0)
def always_fails():
    raise RuntimeError("downstream is down")


# First call: 3 attempts, then RetryError → trips breaker (failure_threshold=2)
# Subsequent calls: rejected fast by CircuitBreakerOpen.
for i in range(3):
    t0 = time.monotonic()
    try:
        always_fails()
    except (RetryError, CircuitBreakerOpen) as e:
        print(f"  call {i+1}: {type(e).__name__} after {time.monotonic() - t0:.4f}s — {e}")
    print(f"  breaker state = {breaker.state!r}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Rate limit awareness (Retry-After header)
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("5. Rate-limit awareness (Retry-After header)")
print("=" * 70)


class FakeHTTP429(urllib.error.HTTPError):
    def __init__(self, retry_after: float):
        # Build a minimal urllib HTTPError with our headers
        super().__init__(
            url="https://example.com",
            code=429,
            msg="Too Many Requests",
            hdrs={"Retry-After": str(retry_after)},
            fp=None,
        )


counter_429 = {"n": 0}


@retry(attempts=4, base_delay=0.05, jitter="none", rate_limit_header="Retry-After")
def rate_limited():
    counter_429["n"] += 1
    if counter_429["n"] == 1:
        raise FakeHTTP429(retry_after=0.3)  # Server says: wait 300ms
    return f"recovered on try {counter_429['n']}"


t0 = time.monotonic()
print(rate_limited(), f"(elapsed {time.monotonic() - t0:.3f}s — should be ~0.3s, not 0.05s)")


# ──────────────────────────────────────────────────────────────────────────────
# 6. AttemptContext for manual loops
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("6. AttemptContext — manual retry control")
print("=" * 70)

manual_counter = {"n": 0}
ctx = attempt(attempts=4, base_delay=0.05, jitter="full")

while not ctx.exhausted and not ctx.timed_out:
    with ctx:
        manual_counter["n"] += 1
        if manual_counter["n"] < 3:
            raise ValueError("not yet")

print(f"succeeded after {manual_counter['n']} attempts (last_error={ctx.last_error!r})")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Comparison: naive retry vs. tiny-retry-plus
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("7. Comparison — naive vs. tiny-retry-plus")
print("=" * 70)


def naive_retry(fn, attempts=3, delay=0.1):
    """No jitter, no timeout awareness, no circuit breaker."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last = e
            if i < attempts - 1:
                time.sleep(delay)
    raise last


def make_maybe_fail():
    state = {"n": 0}
    def fn(ok_on):
        state["n"] += 1
        if state["n"] < ok_on:
            raise ConnectionError("transient")
        return "ok"
    return fn, state


# Reset the counter
naive_fn, _ = make_maybe_fail()
t0 = time.monotonic()
naive_retry(lambda: naive_fn(2), attempts=3, delay=0.1)
naive_time = time.monotonic() - t0

trp_fn, _ = make_maybe_fail()


@retry(attempts=3, base_delay=0.1, jitter="full")
def trp():
    return trp_fn(2)


t0 = time.monotonic()
trp()
trp_time = time.monotonic() - t0

print(f"  naive retry:     {naive_time*1000:6.1f}ms (deterministic 100ms — 1 sleep)")
print(f"  tiny-retry-plus: {trp_time*1000:6.1f}ms (jittered — spreads load across retries)")


# ──────────────────────────────────────────────────────────────────────────────
# 8. Config validation
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("8. Config validation")
print("=" * 70)

try:
    RetryConfig(attempts=0)
except ValueError as e:
    print(f"  attempts=0 rejected: {e}")

try:
    RetryConfig(jitter="magic")
except ValueError as e:
    print(f"  jitter='magic' rejected: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 9. on_retry callback
# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("9. on_retry callback")
print("=" * 70)

events = []


@retry(
    attempts=4,
    base_delay=0.02,
    jitter="none",
    on_retry=lambda err, n, delay: events.append((n, delay, type(err).__name__)),
)
def monitored():
    monitored.n = getattr(monitored, "n", 0) + 1
    if monitored.n < 3:
        raise OSError("down")
    return "ok"


monitored.n = 0
monitored()
for n, delay, kind in events:
    print(f"  retry #{n}: slept {delay:.3f}s after {kind}")


print()
print("All examples completed successfully ✅")
