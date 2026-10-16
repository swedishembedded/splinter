# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.luahost import LuaCase

PRE = "local c = fake(0); local w = m.fwc_new(5, 10, c.read); "


class FixedWindowTests(LuaCase):
    MODULE = "fixedwindow"

    def test_fresh_counter_has_no_hits(self):
        self.expect(PRE + "say(m.fwc_count(w))", "0")

    def test_zero_limit_is_invalid(self):
        self.expect("local c = fake(0); say(try(m.fwc_new, 0, 10, c.read))",
                    '"err:limit must be a positive number"')

    def test_zero_width_is_invalid(self):
        self.expect("local c = fake(0); say(try(m.fwc_new, 5, 0, c.read))",
                    '"err:width must be a positive number"')

    def test_clock_must_be_callable(self):
        self.expect("say(try(m.fwc_new, 5, 10, {}))", '"err:clock must be a function"')

    def test_one_hit_is_the_default(self):
        self.expect(PRE + "say(m.fwc_hit(w))", "true 4")

    def test_hits_reduce_remaining_quota(self):
        self.expect(PRE + "m.fwc_hit(w, 2); say(m.fwc_hit(w, 2))", "true 1")

    def test_last_slot_can_be_used(self):
        self.expect(PRE + "say(m.fwc_hit(w, 5))", "true 0")

    def test_hit_past_limit_is_denied(self):
        self.expect(PRE + "m.fwc_hit(w, 4); say(m.fwc_hit(w, 2))", "false 1")

    def test_denied_hit_does_not_count(self):
        self.expect(PRE + "m.fwc_hit(w, 4); m.fwc_hit(w, 2); say(m.fwc_count(w))", "4")

    def test_hit_amount_must_be_positive(self):
        self.expect(PRE + "say(try(m.fwc_hit, w, 0))", '"err:count must be a positive number"')

    def test_count_tracks_accepted_hits(self):
        self.expect(PRE + "m.fwc_hit(w, 3); say(m.fwc_count(w))", "3")

    def test_remaining_quota_after_hits(self):
        self.expect(PRE + "m.fwc_hit(w, 3); say(m.fwc_remaining(w))", "2")

    def test_window_start_for_new_counter(self):
        self.expect(PRE + "say(m.fwc_window_start(w))", "0")

    def test_window_start_aligns_to_width(self):
        self.expect("local c = fake(27); local w = m.fwc_new(5, 10, c.read); "
                    "say(m.fwc_window_start(w))", "20")

    def test_boundary_instant_opens_new_window(self):
        self.expect("local c = fake(30); local w = m.fwc_new(5, 10, c.read); "
                    "say(m.fwc_window_start(w))", "30")

    def test_count_survives_inside_window(self):
        self.expect(PRE + "m.fwc_hit(w, 3); c.now = 9.5; say(m.fwc_count(w))", "3")

    def test_count_clears_at_window_boundary(self):
        self.expect(PRE + "m.fwc_hit(w, 3); c.now = 10; say(m.fwc_count(w))", "0")

    def test_quota_returns_in_next_window(self):
        self.expect(PRE + "m.fwc_hit(w, 5); c.now = 10; say(m.fwc_hit(w, 5))", "true 0")

    def test_burst_across_boundary_is_allowed(self):
        self.expect(PRE + "c.now = 9; m.fwc_hit(w, 5); c.now = 10; say(m.fwc_hit(w, 5))",
                    "true 0")

    def test_skipped_windows_leave_clean_count(self):
        self.expect(PRE + "m.fwc_hit(w, 5); c.now = 95; say(m.fwc_count(w))", "0")

    def test_window_start_follows_clock(self):
        self.expect(PRE + "c.now = 25; say(m.fwc_window_start(w))", "20")

    def test_resets_in_full_width_at_start(self):
        self.expect(PRE + "say(m.fwc_resets_in(w))", "10")

    def test_resets_in_counts_down(self):
        self.expect(PRE + "c.now = 7.5; say(m.fwc_resets_in(w))", "2.5")

    def test_resets_in_after_later_window_begins(self):
        self.expect(PRE + "c.now = 34; say(m.fwc_resets_in(w))", "6")

    def test_fractional_width_is_supported(self):
        self.expect("local c = fake(0); local w = m.fwc_new(2, 0.5, c.read); "
                    "m.fwc_hit(w, 2); c.now = 0.5; say(m.fwc_hit(w, 2))", "true 0")

    def test_reset_forgets_the_hits(self):
        self.expect(PRE + "m.fwc_hit(w, 5); m.fwc_reset(w); say(m.fwc_count(w))", "0")

    def test_reset_keeps_current_window(self):
        self.expect(PRE + "c.now = 12; m.fwc_hit(w, 1); m.fwc_reset(w); "
                    "say(m.fwc_window_start(w))", "10")

    def test_exhaustion_flag_when_limit_reached(self):
        self.expect(PRE + "m.fwc_hit(w, 4); say(m.fwc_is_exhausted(w)); m.fwc_hit(w, 1); "
                    "say(m.fwc_is_exhausted(w))", "false\ntrue")

    def test_exhaustion_ends_with_the_window(self):
        self.expect(PRE + "m.fwc_hit(w, 5); c.now = 10; say(m.fwc_is_exhausted(w))", "false")

    def test_remaining_restored_in_new_window(self):
        self.expect(PRE + "m.fwc_hit(w, 5); c.now = 10; say(m.fwc_remaining(w))", "5")
