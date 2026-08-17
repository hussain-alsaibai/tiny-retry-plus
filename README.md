# tiny-retry-plus

[![PyPI](https://img.shields.io/pypi/v/tiny-retry-plus?color=blue)](https://pypi.org/project/tiny-retry-plus/)
[![Python](https://img.shields.io/pypi/pyversions/tiny-retry-plus)](https://pypi.org/project/tiny-retry-plus/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Zero deps](https://img.shields.io/badge/dependencies-0-brightgreen)](https://pypi.org/project/tiny-retry-plus/)

> **Zero-dependency retry with jitter, adaptive backoff, and circuit-breaker integration. Sync + async. MIT.**

A stdlib-only Python retry library that takes [`tiny-retry`](https://pypi.org/project/tiny-retry/) and adds the knobs you actually need in production: full / decorrelated / decaying jitter, total-timeout awareness, HTTP `Retry-After` honoring, and an opt-in circuit breaker.

---

## Install

```bash
pip install tiny-retry-plus
```

Or grab the single file — there are no dependencies:

```bash
curl -O https://raw.githubusercontent.com/hussain-alsaibai/tiny-retry-plus/main/tiny_retry_plus.py
```

---

## Quickstart

```python
import urllib.request, urllib.error
from tiny_retry_plus import retry

@retry(
    attempts=5,            # max total tries (including the first)
    base_delay=0.5,        # seconds before retry #2
    max_delay=10.0,        # cap on any single delay
    exponential_base=2.0,  # 0.5 → 1.0 → 2.0 → 4.0 → 8.0 (then capped)
    jitter="full",         # "none" | "full" | "decorrelated" | "decaying"
    timeout=15.0,          # overall budget across all attempts
    exceptions=(urllib.error.URLError, ConnectionError),
)
def fetch(url: str) -> bytes:
    return urllib.request.urlopen(url, timeout=3).read()

print(fetch("https://example.com/"))
```

That's it. No boilerplate, no thread pools, no dependencies.

---

## Features

| Feature | Notes |
|---|---|
| **Exponential backoff** | `base_delay × exponential_base^attempt`, capped at `max_delay` |
| **Jitter** | `"none"`, `"full"` (AWS), `"decorrelated"` (AWS), `"decaying"` (configurable range) |
| **Timeout** | Both total-budget and per-attempt aware — won't sleep past the deadline |
| **Circuit breaker** | Inject a `CircuitBreaker` into any retry; failures trip it; successes in half-open close it |
| **Rate-limit awareness** | Honors `Retry-After` (HTTP 429/503) by sleeping exactly what the server asked |
| **Sync + async** | `@retry` and `@retry_async` with identical semantics |
| **Zero deps** | Pure stdlib; works on Python 3.9+ |
| **Stdlib only** | No `requests`, no `httpx`, no `tenacity`, no `backoff` |
| **MIT** | Permissive license, do what you want |

---

## Benchmark

Measured on Python 3.11, single thread, no failures (success on first try — measures overhead only).

| Operation | Mean latency | Overhead vs baseline |
|---|---|---|
| Plain function call | ~0.22 µs | 1.0× |
| `@retry` (success, jitter=none) | ~0.50 µs | ~2.3× |
| `@retry` (success, jitter=full) | ~0.43 µs | ~2.0× |
| `@retry` (success, jitter=decorrelated) | ~0.47 µs | ~2.1× |
| `@retry` (success, jitter=decaying) | ~0.42 µs | ~1.9× |
| `@retry` with CircuitBreaker | ~0.72 µs | ~3.3× |
| `CircuitBreaker.can_execute()` standalone | ~0.24 µs | ~1.1× |
| `@retry_async` (success) | ~0.44 µs | ~2.0× |

**Retry with real failures** (5 attempts, base_delay=0.1, exponential_base=2.0) — the delay sums are 0.1 + 0.2 + 0.4 = 0.7s of pure backoff:

| Strategy | P50 delay | P99 delay | Spread |
|---|---|---|---|
| `jitter="none"` | 700 ms | 700 ms | 0 ms (thundering herd risk) |
| `jitter="full"` | 350 ms | 700 ms | ~350 ms (reduces synchronization) |
| `jitter="decorrelated"` | ~400 ms | ~900 ms | ~500 ms (best at high concurrency) |
| `jitter="decaying"` | 525 ms | 1050 ms | ~525 ms (configurable range) |

Run `python benchmarks/benchmark.py` to reproduce on your machine.

---

## API reference

### `@retry(...)` — sync decorator

| Parameter | Type | Default | Description |
|---|---|---|---|
| `attempts` | `int` | `3` | Max total attempts (including first). Must be ≥ 1. |
| `timeout` | `float` | `30.0` | Total budget across all attempts (seconds). |
| `base_delay` | `float` | `0.1` | Delay before retry #2 (seconds). |
| `max_delay` | `float` | `30.0` | Cap on any single delay. |
| `exponential_base` | `float` | `2.0` | Multiplier per attempt. |
| `jitter` | `str` | `"full"` | `"none"`, `"full"`, `"decorrelated"`, `"decaying"`. |
| `jitter_range` | `tuple[float, float]` | `(0.5, 1.5)` | Range for decaying jitter. |
| `exceptions` | `tuple[type[Exception], ...]` | `(Exception,)` | Which exceptions to retry. |
| `retriable` | `Callable[[Exception], bool] \| None` | `None` | Custom filter — return `False` to stop. |
| `on_retry` | `Callable[[Exception, int, float], None] \| None` | `None` | Called after each failed attempt: `(error, attempt_number, delay)`. |
| `on_timeout` | `Callable[[Exception \| None], None] \| None` | `None` | Called when overall timeout is reached. |
| `rate_limit_header` | `str \| None` | `"Retry-After"` | Header to consult for rate-limit hints. |
| `rate_limit_statuses` | `tuple[int, ...]` | `(429, 503)` | HTTP status codes treated as "rate limited". |
| `circuit_breaker` | `CircuitBreaker \| None` | `None` | Optional injected breaker. |

You can also pass a pre-built `RetryConfig(...)` object as the first positional argument:

```python
cfg = RetryConfig(attempts=5, base_delay=0.2, jitter="decorrelated")
@retry(cfg)
def fetch(): ...
```

### `@retry_async(...)` — async decorator

Identical signature and semantics as `@retry`, but for `async def` functions. Uses `await asyncio.sleep(...)` between attempts.

### `RetryError`

Raised when all attempts are exhausted.

```python
try:
    fetch(url)
except RetryError as e:
    print(f"gave up after {e.attempts} tries; last error: {e.last_error!r}")
```

### `AttemptContext` (alias `attempt`)

Context manager for manual retry loops when a decorator doesn't fit (e.g. branching logic between attempts).

```python
from tiny_retry_plus import attempt

ctx = attempt(attempts=5, base_delay=0.2, jitter="full")
while not ctx.exhausted and not ctx.timed_out:
    with ctx:
        do_the_thing()
```

### `CircuitBreaker` (and `circuit(...)` shorthand)

Minimal three-state breaker (`CLOSED` → `OPEN` → `HALF_OPEN`).

```python
from tiny_retry_plus import CircuitBreaker, retry

breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0, half_open_attempts=3)

@retry(attempts=10, circuit_breaker=breaker, timeout=60)
def call_downstream(): ...

print(breaker.state)  # "CLOSED" | "OPEN" | "HALF_OPEN"
```

Or use the shorthand: `circuit(5, 30.0, 3)` is equivalent to `CircuitBreaker(5, 30.0, 3)`.

### `CircuitBreakerOpen`

Raised when the breaker is `OPEN` and the call is rejected before any attempt runs.

---

## Jitter strategies — when to use what

| Strategy | Formula | Best for |
|---|---|---|
| `none` | `delay_n = base × exp_base^n` | Single-process tests, deterministic timing |
| `full` | `delay_n = exp_delay × U(0, 1)` | General use; spreads a herd over `[0, delay]` |
| `decorrelated` | `delay_n = U(base, delay_{n-1} × 3)` | High-concurrency clients; reduces correlation between retries |
| `decaying` | `delay_n = exp_delay × U(low, high)` | When you want bounded randomization with a custom range |

For most HTTP APIs, **full** is the right default. For batch workers hammering the same upstream, **decorrelated** reduces synchronized spikes best.

---

## Comparison

| Feature | tiny-retry-plus | tenacity | backoff |
|---|---|---|---|
| **Dependencies** | 0 | 0 (with extras for async) | 0 |
| **Lines of code (lib)** | ~700 | ~3,000+ | ~400 |
| **Sync + async** | ✅ | ✅ | ⚠️ sync only |
| **Exponential backoff** | ✅ | ✅ | ✅ |
| **Jitter (full/decorrelated/decaying)** | ✅ | ⚠️ partial | ✅ |
| **Total timeout** | ✅ | ✅ | ❌ |
| **Per-attempt timeout** | ✅ (via outer try) | ✅ (built-in) | ❌ |
| **Circuit breaker** | ✅ (lightweight) | � | ❌ |
| **Retry-After header** | ✅ | ⚠️ manual | ❌ |
| **`on_retry` / hooks** | ✅ | ✅ (rich) | ✅ |
| **Stop-after-N-seconds** | ✅ (`timeout=`) | ✅ | ❌ |
| **Custom retriable predicate** | ✅ | ✅ | ✅ |
| **Context-manager manual loop** | ✅ (`AttemptContext`) | ❌ | ❌ |
| **Single-file install** | ✅ (`curl -O`) | � | ❌ |
| **License** | MIT | Apache-2.0 | MIT |

`tiny-retry-plus` is for people who want retry that **just works** without learning a new mini-language or pulling in a transitive dependency.

---

## License

MIT — see [LICENSE](LICENSE).
