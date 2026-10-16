# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.luahost import LuaCase

PRE = ("local c = fake(0); local b = m.cbk_new({failure_threshold = 3, open_seconds = 10, "
       "probe_limit = 2, clock = c.read}); ")
DEF = "local c = fake(0); local b = m.cbk_new({clock = c.read}); "
FAIL3 = "m.cbk_failure(b); m.cbk_failure(b); m.cbk_failure(b); "


class BreakerTests(LuaCase):
    MODULE = "breaker"

    def test_new_breaker_starts_closed(self):
        self.expect(PRE + "say(m.cbk_state(b))", '"closed"')

    def test_default_threshold_is_five_failures(self):
        self.expect(DEF + "for i = 1, 4 do m.cbk_failure(b) end; say(m.cbk_state(b)); "
                    "m.cbk_failure(b); say(m.cbk_state(b))", '"closed"\n"open"')

    def test_default_open_time_is_thirty_seconds(self):
        self.expect(DEF + "m.cbk_trip(b); c.now = 29.5; say(m.cbk_state(b)); c.now = 30; "
                    "say(m.cbk_state(b))", '"open"\n"half_open"')

    def test_default_allows_a_single_probe(self):
        self.expect(DEF + "m.cbk_trip(b); c.now = 30; say(m.cbk_allow(b), m.cbk_allow(b))",
                    "true false")

    def test_zero_threshold_is_refused(self):
        self.expect("local c = fake(0); say(try(m.cbk_new, {failure_threshold = 0, clock = c.read}))",
                    '"err:failure threshold must be a positive integer"')

    def test_fractional_threshold_is_refused(self):
        self.expect("local c = fake(0); say(try(m.cbk_new, {failure_threshold = 2.5, clock = c.read}))",
                    '"err:failure threshold must be a positive integer"')

    def test_zero_probe_limit_is_refused(self):
        self.expect("local c = fake(0); say(try(m.cbk_new, {probe_limit = 0, clock = c.read}))",
                    '"err:probe limit must be a positive integer"')

    def test_zero_open_time_is_refused(self):
        self.expect("local c = fake(0); say(try(m.cbk_new, {open_seconds = 0, clock = c.read}))",
                    '"err:open seconds must be a positive number"')

    def test_missing_clock_is_refused(self):
        self.expect("say(try(m.cbk_new, {}))", '"err:clock must be a function"')

    def test_closed_breaker_lets_calls_through(self):
        self.expect(PRE + "say(m.cbk_allow(b), m.cbk_allow(b), m.cbk_allow(b))",
                    "true true true")

    def test_failures_below_threshold_keep_it_closed(self):
        self.expect(PRE + "m.cbk_failure(b); m.cbk_failure(b); say(m.cbk_state(b))", '"closed"')

    def test_threshold_failure_opens_breaker(self):
        self.expect(PRE + FAIL3 + "say(m.cbk_state(b))", '"open"')

    def test_failure_count_is_reported(self):
        self.expect(PRE + "m.cbk_failure(b); m.cbk_failure(b); say(m.cbk_failures(b))", "2")

    def test_success_resets_failure_streak(self):
        self.expect(PRE + "m.cbk_failure(b); m.cbk_failure(b); m.cbk_success(b); "
                    "m.cbk_failure(b); m.cbk_failure(b); say(m.cbk_state(b), m.cbk_failures(b))",
                    '"closed" 2')

    def test_opening_clears_failure_count(self):
        self.expect(PRE + FAIL3 + "say(m.cbk_failures(b))", "0")

    def test_open_breaker_rejects_calls(self):
        self.expect(PRE + FAIL3 + "say(m.cbk_allow(b))", "false")

    def test_breaker_stays_open_before_timeout(self):
        self.expect(PRE + FAIL3 + "c.now = 9.5; say(m.cbk_state(b), m.cbk_allow(b))",
                    '"open" false')

    def test_breaker_half_opens_after_timeout(self):
        self.expect(PRE + FAIL3 + "c.now = 10; say(m.cbk_state(b))", '"half_open"')

    def test_timeout_counts_from_the_trip(self):
        self.expect(PRE + "c.now = 100; " + FAIL3 + "c.now = 109; say(m.cbk_state(b)); "
                    "c.now = 110; say(m.cbk_state(b))", '"open"\n"half_open"')

    def test_half_open_admits_probe_limit_calls(self):
        self.expect(PRE + FAIL3 + "c.now = 10; say(m.cbk_allow(b), m.cbk_allow(b), m.cbk_allow(b))",
                    "true true false")

    def test_all_probes_succeeding_closes_breaker(self):
        self.expect(PRE + FAIL3 + "c.now = 10; m.cbk_allow(b); m.cbk_allow(b); "
                    "m.cbk_success(b); say(m.cbk_state(b)); m.cbk_success(b); say(m.cbk_state(b))",
                    '"half_open"\n"closed"')

    def test_closed_again_lets_everything_through(self):
        self.expect(PRE + FAIL3 + "c.now = 10; m.cbk_allow(b); m.cbk_allow(b); "
                    "m.cbk_success(b); m.cbk_success(b); say(m.cbk_allow(b), m.cbk_allow(b), m.cbk_allow(b))",
                    "true true true")

    def test_probe_failure_reopens_breaker(self):
        self.expect(PRE + FAIL3 + "c.now = 10; m.cbk_allow(b); m.cbk_failure(b); "
                    "say(m.cbk_state(b))", '"open"')

    def test_reopened_breaker_waits_full_timeout(self):
        self.expect(PRE + FAIL3 + "c.now = 10; m.cbk_allow(b); m.cbk_failure(b); c.now = 19; "
                    "say(m.cbk_state(b)); c.now = 20; say(m.cbk_state(b))", '"open"\n"half_open"')

    def test_closing_starts_a_fresh_failure_count(self):
        self.expect(PRE + FAIL3 + "c.now = 10; m.cbk_allow(b); m.cbk_allow(b); m.cbk_success(b); "
                    "m.cbk_success(b); m.cbk_failure(b); m.cbk_failure(b); say(m.cbk_state(b))",
                    '"closed"')

    def test_late_failure_does_not_extend_open_time(self):
        self.expect(PRE + FAIL3 + "c.now = 5; m.cbk_failure(b); c.now = 10; say(m.cbk_state(b))",
                    '"half_open"')

    def test_late_success_does_not_close_open_breaker(self):
        self.expect(PRE + FAIL3 + "m.cbk_success(b); say(m.cbk_state(b))", '"open"')

    def test_second_half_open_resets_probe_budget(self):
        self.expect(PRE + FAIL3 + "c.now = 10; m.cbk_allow(b); m.cbk_allow(b); m.cbk_success(b); "
                    "m.cbk_failure(b); c.now = 20; say(m.cbk_allow(b), m.cbk_allow(b))",
                    "true true")

    def test_retry_in_counts_down_while_open(self):
        self.expect(PRE + FAIL3 + "c.now = 4; say(m.cbk_retry_in(b))", "6")

    def test_retry_in_is_zero_when_closed(self):
        self.expect(PRE + "say(m.cbk_retry_in(b))", "0")

    def test_retry_in_is_zero_when_half_open(self):
        self.expect(PRE + FAIL3 + "c.now = 12; say(m.cbk_retry_in(b))", "0")

    def test_trip_forces_the_breaker_open(self):
        self.expect(PRE + "m.cbk_trip(b); say(m.cbk_state(b), m.cbk_allow(b))", '"open" false')

    def test_reset_closes_an_open_breaker(self):
        self.expect(PRE + FAIL3 + "m.cbk_reset(b); say(m.cbk_state(b), m.cbk_allow(b))",
                    '"closed" true')

    def test_reset_clears_failure_streak(self):
        self.expect(PRE + "m.cbk_failure(b); m.cbk_failure(b); m.cbk_reset(b); "
                    "say(m.cbk_failures(b))", "0")

    def test_call_returns_result_of_function(self):
        self.expect(PRE + "say(m.cbk_call(b, function(x, y) return x + y end, 2, 3))", "true 5")

    def test_call_failure_returns_error_text(self):
        self.expect(PRE + "say(m.cbk_call(b, function() error('boom', 0) end))", 'false "boom"')

    def test_failed_calls_open_the_breaker(self):
        self.expect(PRE + "for i = 1, 3 do m.cbk_call(b, function() error('x', 0) end) end; "
                    "say(m.cbk_state(b))", '"open"')

    def test_successful_call_clears_streak(self):
        self.expect(PRE + "local bad = function() error('x', 0) end; m.cbk_call(b, bad); "
                    "m.cbk_call(b, bad); m.cbk_call(b, function() return 1 end); say(m.cbk_failures(b))",
                    "0")

    def test_call_is_rejected_while_open(self):
        self.expect(PRE + FAIL3 + "local hit = false; "
                    "local r1, r2 = m.cbk_call(b, function() hit = true end); say(r1, r2, hit)", 'nil "circuit open" false')

    def test_successful_probe_calls_close_breaker(self):
        self.expect(PRE + FAIL3 + "c.now = 10; m.cbk_call(b, function() end); "
                    "m.cbk_call(b, function() end); say(m.cbk_state(b))", '"closed"')

    def test_failure_after_timeout_counts_as_probe(self):
        self.expect(PRE + FAIL3 + "c.now = 10; m.cbk_failure(b); c.now = 19; "
                    "say(m.cbk_state(b))", '"open"')

    def test_success_after_timeout_closes_single_probe_breaker(self):
        self.expect(DEF + "m.cbk_trip(b); c.now = 30; m.cbk_success(b); say(m.cbk_state(b))",
                    '"closed"')
