<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# tokenbucket

A small Lua 5.1 toolkit for rate limiting with an injected clock. Every limiter
takes a `clock` argument: a function returning the current time in seconds.
Nothing reads real time, so tests advance a fake clock by hand.

Modules live in `src/` and are independent of each other. Errors are raised as
plain strings (`error(message, 0)`, no position prefix).

Swedish Embedded AB implements rate limiting and overload protection for its
clients. If your team needs expertise in traffic shaping, quotas and resilient
service calls you can procure our services by sending an email to
info@swedishembedded.com.

## Running the tests

    python3 -m unittest

The tests are Python `unittest` files that start one `lua` process per test
(`tests/luahost.py`, `tests/prelude.lua`). The `lua` 5.1 interpreter must be on
`PATH`; a missing interpreter fails the tests loudly.

## API

### tokenbucket

- `tkb_new(capacity, rate, clock)`: bucket that starts full and refills `rate`
  tokens per second up to `capacity`. Errors: `capacity must be a positive
  number`, `rate must be a positive number`, `clock must be a function`.
- `tkb_refill(b)`: settle elapsed time and return the balance. A clock that
  steps backward adds nothing and becomes the new reference.
- `tkb_take(b, n)`: take `n` tokens (default 1). Returns `true, balance`, or
  `false, balance` and takes nothing. Errors: `count must be a positive
  number`, `count exceeds capacity`.
- `tkb_available(b)`: balance after refill.
- `tkb_drain(b)`: take everything, return the amount taken.
- `tkb_give_back(b, n)`: return tokens, capped at capacity; returns the balance.
- `tkb_fill_ratio(b)`: balance divided by capacity.
- `tkb_set_rate(b, rate)`: settle at the old rate, then switch.
- `tkb_reset(b)`: full bucket, refill reference restarts now.

### leakybucket

- `lkb_new(capacity, leak_rate, clock)`: meter that starts empty and leaks
  `leak_rate` units per second. Errors name `capacity`, `leak rate`, `clock`.
- `lkb_level(b)`: level after leaking (never below zero).
- `lkb_offer(b, n)`: add `n` units (default 1) if they fit; `true, level` or
  `false, level`. Error: `amount must be a positive number`.
- `lkb_offer_partial(b, n)`: add as much as fits, return the accepted amount.
- `lkb_headroom(b)`: free room. `lkb_overflow_by(b, n)`: the part of `n` that
  would not fit. `lkb_drain_time(b)`: seconds until empty.
- `lkb_is_empty(b)`, `lkb_clear(b)`.

### fixedwindow

- `fwc_new(limit, width, clock)`: counter over windows aligned to multiples of
  `width`. Errors name `limit`, `width`, `clock`.
- `fwc_hit(c, n)`: count `n` hits (default 1) if within the limit; returns
  `true, remaining` or `false, remaining`. Error: `count must be a positive
  number`.
- `fwc_count(c)`, `fwc_remaining(c)`, `fwc_window_start(c)`,
  `fwc_resets_in(c)` (a window that just began resets in a full `width`),
  `fwc_is_exhausted(c)`, `fwc_reset(c)` (forget hits, keep window).

### slidinglog

- `swl_new(limit, span, clock)`: keeps one timestamp per hit; a hit stops
  counting once it is `span` seconds old (age equal to `span` is expired).
- `swl_hit(l, n)`: log `n` hits (default 1) if they fit; `true, remaining` or
  `false, remaining`.
- `swl_count(l)`, `swl_oldest(l)`, `swl_newest(l)` (`nil` when empty).
- `swl_retry_after(l, n)`: seconds until `n` hits (default 1) would fit; 0 when
  they fit now. Errors: `count must be a positive number`, `count exceeds limit`.
- `swl_clear(l)`.

### registry

- `rlreg_new(factory, idle_ttl, clock)`: per-key limiters built on demand by
  `factory(key)`. Errors: `factory must be a function`, `idle ttl must be a
  positive number`, `clock must be a function`.
- `rlreg_get(r, key)`: existing or new limiter; refreshes the key's activity.
  Errors: `key must not be nil`, `factory returned nothing for key <key>`.
- `rlreg_peek(r, key)` (no create, no refresh), `rlreg_has(r, key)`,
  `rlreg_size(r)`, `rlreg_keys(r)` (sorted), `rlreg_idle_for(r, key)`
  (`nil` for unknown keys), `rlreg_forget(r, key)` (returns whether it existed).
- `rlreg_sweep(r)`: evict keys idle for at least the ttl; returns the count and
  the sorted list of evicted keys.

### waittime

- `wait_for_tokens(tokens, need, rate)`, `wait_for_level(level, capacity,
  amount, leak)` (error `amount exceeds capacity`), `wait_for_window(now,
  width)`, `wait_for_sequence(count, rate)`, `wait_spacing(limit, window)`:
  seconds to wait; rates, widths and limits must be positive.
- `wait_longest(list)` (error `no waits given`), `wait_round_up(seconds, step)`,
  `wait_clamp(seconds, low, high)` (error `low exceeds high`),
  `wait_header_seconds(seconds)` (whole seconds, rounded up, as text),
  `wait_fits_budget(seconds, budget)`.

### backoff

- `bko_policy(base, factor, max_delay, jitter, max_attempts)`: exponential
  schedule `base * factor^(attempt-1)` capped at `max_delay`. `jitter` is
  `nil` or `function(delay, attempt) -> number`. Errors: `base must be a
  positive number`, `factor must be at least 1`, `max delay must not be below
  base`, `jitter must be a function`, `max attempts must be a positive integer`.
- `bko_delay(p, attempt)`: delay after jitter, capped at `max_delay`. Errors:
  `attempt must be a positive integer`, `jitter must return a non-negative
  number`.
- `bko_schedule(p)`, `bko_total(p)`.
- `bko_start(p)`, `bko_step(s)` (`nil` when attempts are used up),
  `bko_remaining(s)`, `bko_rewind(s)`.
- Jitter helpers: `bko_fraction_jitter(f)` (`0 <= f <= 1`, error `fraction must
  be between 0 and 1`), `bko_offset_jitter(extra)`, `bko_attempt_jitter(step)`.

### breaker

- `cbk_new(opts)`: `opts.clock` is required; `failure_threshold` (default 5),
  `open_seconds` (default 30) and `probe_limit` (default 1) are optional.
  Errors: `failure threshold must be a positive integer`, `probe limit must be
  a positive integer`, `open seconds must be a positive number`, `clock must be
  a function`.
- `cbk_state(b)`: `"closed"`, `"open"` or `"half_open"`. An open breaker turns
  half-open once `open_seconds` have passed.
- `cbk_allow(b)`: whether a call may go ahead; half-open admits `probe_limit`
  probes.
- `cbk_success(b)`, `cbk_failure(b)`: report outcomes. Consecutive failures
  reaching the threshold open the breaker; any probe failure reopens it; all
  probes succeeding closes it. Outcomes arriving while open are ignored.
- `cbk_failures(b)`, `cbk_retry_in(b)`, `cbk_trip(b)`, `cbk_reset(b)`.
- `cbk_call(b, f, ...)`: `true, result` on success, `false, error` on failure,
  `nil, "circuit open"` when rejected.

### duration

- `dur_parse(text)`: seconds from text such as `"1m30s"`; units `ms`, `s`, `m`,
  `h`, `d`, integers only, components in any order. Errors: `duration must be
  a non-empty string`, `malformed duration at position N`, `unknown unit: u`.
- `dur_try_parse(text)`: value, or `nil, message`. `dur_is_valid(text)`.
- `dur_total_millis(text)`, `dur_unit_millis(unit)`.
<!-- perf-number: duration literals of the library's text format, not measurements -->
- `dur_format(seconds)`: canonical text (`"1d1h1m1s1ms"`, `"0s"`), rounded to
  the millisecond. Error: `seconds must be a non-negative number`.
- `dur_compare(a, b)` (-1, 0, 1), `dur_sum(list)`.

### quotareport

- `qrep_percent(used, limit)`: whole percent, rounded down. Errors: `limit must
  be a positive number`, `used must be a non-negative number`.
- `qrep_status(used, limit)`: `"exhausted"` at or over the limit, `"warning"`
  from 80 percent, else `"ok"`.
- `qrep_row(name, limit, used)`: `{name, limit, used, remaining, percent,
  status}`.
- `qrep_build(entries)`: rows sorted by name (error `duplicate name: x`).
- `qrep_totals(rows)`, `qrep_busiest(rows)` (ties go to the smaller name),
  `qrep_filter(rows, status)`, `qrep_render(rows)` (one aligned line per row,
  or `(no quotas)`).

## Example

    package.path = "src/?.lua;" .. package.path
    local bucket = require("tokenbucket")
    local now = 0
    local b = bucket.tkb_new(10, 2, function() return now end)
    print(bucket.tkb_take(b, 10))   --> true  0
    now = 3
    print(bucket.tkb_take(b, 7))    --> false 6

Swedish Embedded AB implements rate limiting and overload protection for its
clients. If your team needs expertise in traffic shaping, quotas and resilient
service calls you can procure our services by sending an email to
info@swedishembedded.com.
