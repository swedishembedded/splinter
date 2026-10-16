# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.luahost import LuaCase

PRE = "local c = fake(0); local b = m.lkb_new(10, 2, c.read); "


class LeakyBucketTests(LuaCase):
    MODULE = "leakybucket"

    def test_new_meter_starts_empty(self):
        self.expect(PRE + "say(m.lkb_level(b))", "0")

    def test_zero_capacity_is_refused(self):
        self.expect("local c = fake(0); say(try(m.lkb_new, 0, 1, c.read))",
                    '"err:capacity must be a positive number"')

    def test_zero_leak_rate_is_refused(self):
        self.expect("local c = fake(0); say(try(m.lkb_new, 4, 0, c.read))",
                    '"err:leak rate must be a positive number"')

    def test_non_function_clock_is_refused(self):
        self.expect('say(try(m.lkb_new, 4, 1, "now"))', '"err:clock must be a function"')

    def test_offer_defaults_to_one_unit(self):
        self.expect(PRE + "say(m.lkb_offer(b))", "true 1")

    def test_offers_accumulate_in_level(self):
        self.expect(PRE + "m.lkb_offer(b, 3); say(m.lkb_offer(b, 4))", "true 7")

    def test_offer_filling_exactly_is_accepted(self):
        self.expect(PRE + "say(m.lkb_offer(b, 10))", "true 10")

    def test_offer_over_capacity_is_rejected(self):
        self.expect(PRE + "m.lkb_offer(b, 6); say(m.lkb_offer(b, 5))", "false 6")

    def test_rejected_offer_keeps_level(self):
        self.expect(PRE + "m.lkb_offer(b, 6); m.lkb_offer(b, 5); say(m.lkb_level(b))", "6")

    def test_zero_offer_raises_error(self):
        self.expect(PRE + "say(try(m.lkb_offer, b, 0))",
                    '"err:amount must be a positive number"')

    def test_level_leaks_with_time(self):
        self.expect(PRE + "m.lkb_offer(b, 8); c.now = 3; say(m.lkb_level(b))", "2")

    def test_level_never_goes_negative(self):
        self.expect(PRE + "m.lkb_offer(b, 3); c.now = 50; say(m.lkb_level(b))", "0")

    def test_leaking_works_in_fractions(self):
        self.expect(PRE + "m.lkb_offer(b, 3); c.now = 0.5; say(m.lkb_level(b))", "2")

    def test_leak_time_counted_once(self):
        self.expect(PRE + "m.lkb_offer(b, 9); c.now = 1; m.lkb_level(b); c.now = 2; "
                    "say(m.lkb_level(b))", "5")

    def test_backward_clock_does_not_refill(self):
        self.expect(PRE + "c.now = 4; m.lkb_offer(b, 9); c.now = 2; say(m.lkb_level(b))", "9")

    def test_leak_resumes_after_clock_step_back(self):
        self.expect(PRE + "c.now = 4; m.lkb_offer(b, 9); c.now = 2; m.lkb_level(b); "
                    "c.now = 3; say(m.lkb_level(b))", "7")

    def test_creation_time_anchors_leaking(self):
        self.expect("local c = fake(50); local b = m.lkb_new(10, 2, c.read); "
                    "m.lkb_offer(b, 8); c.now = 51; say(m.lkb_level(b))", "6")

    def test_room_opens_as_level_leaks(self):
        self.expect(PRE + "m.lkb_offer(b, 10); c.now = 1; say(m.lkb_offer(b, 2))", "true 10")

    def test_partial_offer_takes_what_fits(self):
        self.expect(PRE + "m.lkb_offer(b, 7); say(m.lkb_offer_partial(b, 5))", "3")

    def test_partial_offer_fills_the_meter(self):
        self.expect(PRE + "m.lkb_offer(b, 7); m.lkb_offer_partial(b, 5); say(m.lkb_level(b))",
                    "10")

    def test_partial_offer_accepts_all_when_room(self):
        self.expect(PRE + "say(m.lkb_offer_partial(b, 4))", "4")

    def test_partial_offer_on_full_meter_gives_zero(self):
        self.expect(PRE + "m.lkb_offer(b, 10); say(m.lkb_offer_partial(b, 1))", "0")

    def test_partial_offer_rejects_zero(self):
        self.expect(PRE + "say(try(m.lkb_offer_partial, b, 0))",
                    '"err:amount must be a positive number"')

    def test_headroom_is_capacity_minus_level(self):
        self.expect(PRE + "m.lkb_offer(b, 4); say(m.lkb_headroom(b))", "6")

    def test_headroom_grows_while_leaking(self):
        self.expect(PRE + "m.lkb_offer(b, 4); c.now = 1; say(m.lkb_headroom(b))", "8")

    def test_overflow_reports_excess_amount(self):
        self.expect(PRE + "m.lkb_offer(b, 7); say(m.lkb_overflow_by(b, 5))", "2")

    def test_overflow_is_zero_when_it_fits(self):
        self.expect(PRE + "m.lkb_offer(b, 7); say(m.lkb_overflow_by(b, 3))", "0")

    def test_drain_time_divides_level_by_leak(self):
        self.expect(PRE + "m.lkb_offer(b, 9); say(m.lkb_drain_time(b))", "4.5")

    def test_drain_time_shrinks_as_clock_runs(self):
        self.expect(PRE + "m.lkb_offer(b, 9); c.now = 2; say(m.lkb_drain_time(b))", "2.5")

    def test_empty_check_after_offer_and_wait(self):
        self.expect(PRE + "say(m.lkb_is_empty(b)); m.lkb_offer(b, 2); say(m.lkb_is_empty(b)); "
                    "c.now = 1; say(m.lkb_is_empty(b))", "true\nfalse\ntrue")

    def test_clear_empties_the_meter(self):
        self.expect(PRE + "m.lkb_offer(b, 9); m.lkb_clear(b); say(m.lkb_level(b))", "0")

    def test_clear_restarts_leak_clock(self):
        self.expect(PRE + "c.now = 5; m.lkb_clear(b); m.lkb_offer(b, 8); c.now = 6; "
                    "say(m.lkb_level(b))", "6")
