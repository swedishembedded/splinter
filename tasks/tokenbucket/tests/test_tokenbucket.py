# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.luahost import LuaCase

PRE = "local c = fake(0); local b = m.tkb_new(10, 2, c.read); "


class TokenBucketTests(LuaCase):
    MODULE = "tokenbucket"

    def test_new_bucket_starts_full(self):
        self.expect(PRE + "say(m.tkb_available(b))", "10")

    def test_zero_capacity_is_rejected(self):
        self.expect("local c = fake(0); say(try(m.tkb_new, 0, 2, c.read))",
                    '"err:capacity must be a positive number"')

    def test_text_capacity_is_rejected(self):
        self.expect('local c = fake(0); say(try(m.tkb_new, "9", 2, c.read))',
                    '"err:capacity must be a positive number"')

    def test_zero_rate_is_rejected(self):
        self.expect("local c = fake(0); say(try(m.tkb_new, 5, 0, c.read))",
                    '"err:rate must be a positive number"')

    def test_missing_clock_is_rejected(self):
        self.expect("say(try(m.tkb_new, 5, 1, 7))", '"err:clock must be a function"')

    def test_single_token_taken_by_default(self):
        self.expect(PRE + "say(m.tkb_take(b))", "true 9")

    def test_several_tokens_taken_at_once(self):
        self.expect(PRE + "say(m.tkb_take(b, 4))", "true 6")

    def test_whole_capacity_can_be_taken(self):
        self.expect(PRE + "say(m.tkb_take(b, 10))", "true 0")

    def test_take_beyond_balance_is_refused(self):
        self.expect(PRE + "m.tkb_take(b, 8); say(m.tkb_take(b, 3))", "false 2")

    def test_refused_take_keeps_balance(self):
        self.expect(PRE + "m.tkb_take(b, 8); m.tkb_take(b, 3); say(m.tkb_available(b))", "2")

    def test_count_above_capacity_raises(self):
        self.expect(PRE + "say(try(m.tkb_take, b, 11))", '"err:count exceeds capacity"')

    def test_zero_count_raises_an_error(self):
        self.expect(PRE + "say(try(m.tkb_take, b, 0))",
                    '"err:count must be a positive number"')

    def test_take_includes_elapsed_refill(self):
        self.expect(PRE + "m.tkb_take(b, 10); c.now = 1; say(m.tkb_take(b, 2))", "true 0")

    def test_tokens_return_with_elapsed_time(self):
        self.expect(PRE + "m.tkb_take(b, 10); c.now = 2; say(m.tkb_available(b))", "4")

    def test_refill_stops_at_capacity(self):
        self.expect(PRE + "m.tkb_take(b, 3); c.now = 100; say(m.tkb_available(b))", "10")

    def test_partial_seconds_refill_fractionally(self):
        self.expect(PRE + "m.tkb_take(b, 10); c.now = 0.25; say(m.tkb_available(b))", "0.5")

    def test_elapsed_time_is_not_counted_twice(self):
        self.expect(PRE + "m.tkb_take(b, 10); c.now = 1; m.tkb_available(b); "
                    "c.now = 2; say(m.tkb_available(b))", "4")

    def test_clock_stepping_back_adds_nothing(self):
        self.expect(PRE + "m.tkb_take(b, 10); c.now = 3; m.tkb_available(b); "
                    "c.now = 1; say(m.tkb_available(b))", "6")

    def test_refill_resumes_after_backward_step(self):
        self.expect(PRE + "m.tkb_take(b, 10); c.now = 3; m.tkb_available(b); "
                    "c.now = 1; m.tkb_available(b); c.now = 2; say(m.tkb_available(b))", "8")

    def test_creation_time_starts_the_refill(self):
        self.expect("local c = fake(100); local b = m.tkb_new(10, 2, c.read); "
                    "m.tkb_take(b, 10); c.now = 101; say(m.tkb_available(b))", "2")

    def test_drain_returns_whole_balance(self):
        self.expect(PRE + "m.tkb_take(b, 4); say(m.tkb_drain(b))", "6")

    def test_drain_leaves_bucket_empty(self):
        self.expect(PRE + "m.tkb_drain(b); say(m.tkb_available(b))", "0")

    def test_drain_counts_pending_refill(self):
        self.expect(PRE + "m.tkb_take(b, 10); c.now = 3; say(m.tkb_drain(b))", "6")

    def test_give_back_restores_tokens(self):
        self.expect(PRE + "m.tkb_take(b, 6); say(m.tkb_give_back(b, 2))", "6")

    def test_give_back_cannot_exceed_capacity(self):
        self.expect(PRE + "say(m.tkb_give_back(b, 5))", "10")

    def test_give_back_rejects_zero(self):
        self.expect(PRE + "say(try(m.tkb_give_back, b, 0))",
                    '"err:count must be a positive number"')

    def test_give_back_adds_to_refilled_balance(self):
        self.expect(PRE + "m.tkb_take(b, 10); c.now = 1; say(m.tkb_give_back(b, 1))", "3")

    def test_fill_ratio_of_half_empty_bucket(self):
        self.expect(PRE + "m.tkb_take(b, 5); say(m.tkb_fill_ratio(b))", "0.5")

    def test_fill_ratio_spans_zero_to_one(self):
        self.expect(PRE + "say(m.tkb_fill_ratio(b)); m.tkb_drain(b); say(m.tkb_fill_ratio(b))",
                    "1\n0")

    def test_new_rate_applies_from_now_on(self):
        self.expect(PRE + "m.tkb_take(b, 10); m.tkb_set_rate(b, 4); c.now = 1; "
                    "say(m.tkb_available(b))", "4")

    def test_new_rate_settles_old_rate_first(self):
        self.expect(PRE + "m.tkb_take(b, 10); c.now = 1; m.tkb_set_rate(b, 4); "
                    "say(m.tkb_available(b))", "2")

    def test_zero_new_rate_is_rejected(self):
        self.expect(PRE + "say(try(m.tkb_set_rate, b, 0))",
                    '"err:rate must be a positive number"')

    def test_reset_refills_the_bucket(self):
        self.expect(PRE + "m.tkb_take(b, 10); m.tkb_reset(b); say(m.tkb_available(b))", "10")

    def test_reset_restarts_the_refill_clock(self):
        self.expect(PRE + "c.now = 5; m.tkb_reset(b); m.tkb_take(b, 10); c.now = 6; "
                    "say(m.tkb_available(b))", "2")
