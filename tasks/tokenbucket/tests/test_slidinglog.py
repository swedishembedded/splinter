# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.luahost import LuaCase

PRE = "local c = fake(0); local l = m.swl_new(3, 10, c.read); "


class T(LuaCase):
    MODULE = "slidinglog"

    def test_new_log_is_empty(self):
        self.expect(PRE + "say(m.swl_count(l))", "0")

    def test_zero_limit_is_not_allowed(self):
        self.expect("local c = fake(0); say(try(m.swl_new, 0, 10, c.read))",
                    '"err:limit must be a positive number"')

    def test_zero_span_is_not_allowed(self):
        self.expect("local c = fake(0); say(try(m.swl_new, 3, 0, c.read))",
                    '"err:span must be a positive number"')

    def test_single_hit_is_the_default(self):
        self.expect(PRE + "say(m.swl_hit(l))", "true 2")

    def test_hits_use_up_the_limit(self):
        self.expect(PRE + "m.swl_hit(l); m.swl_hit(l); say(m.swl_hit(l))", "true 0")

    def test_hit_beyond_limit_is_denied(self):
        self.expect(PRE + "m.swl_hit(l, 3); say(m.swl_hit(l))", "false 0")

    def test_denial_reports_remaining_room(self):
        self.expect(PRE + "m.swl_hit(l, 2); say(m.swl_hit(l, 2))", "false 1")

    def test_denied_hit_is_not_logged(self):
        self.expect(PRE + "m.swl_hit(l, 2); m.swl_hit(l, 2); say(m.swl_count(l))", "2")

    def test_zero_weight_hit_is_invalid(self):
        self.expect(PRE + "say(try(m.swl_hit, l, 0))", '"err:count must be a positive number"')

    def test_old_hits_expire_after_span(self):
        self.expect(PRE + "m.swl_hit(l, 3); c.now = 10.5; say(m.swl_count(l))", "0")

    def test_hit_exactly_span_old_expires(self):
        self.expect(PRE + "m.swl_hit(l); c.now = 10; say(m.swl_count(l))", "0")

    def test_hits_expire_one_by_one(self):
        self.expect(PRE + "m.swl_hit(l); c.now = 4; m.swl_hit(l); c.now = 11; "
                    "say(m.swl_count(l))", "1")

    def test_no_burst_at_window_edge(self):
        self.expect(PRE + "c.now = 9; m.swl_hit(l, 3); c.now = 10; say(m.swl_hit(l))",
                    "false 0")

    def test_room_returns_when_oldest_expires(self):
        self.expect(PRE + "m.swl_hit(l); c.now = 5; m.swl_hit(l, 2); c.now = 10; "
                    "say(m.swl_hit(l))", "true 0")

    def test_oldest_entry_has_earliest_timestamp(self):
        self.expect(PRE + "c.now = 2; m.swl_hit(l); c.now = 4; m.swl_hit(l); "
                    "say(m.swl_oldest(l))", "2")

    def test_oldest_entry_moves_after_expiry(self):
        self.expect(PRE + "c.now = 2; m.swl_hit(l); c.now = 4; m.swl_hit(l); c.now = 12; "
                    "say(m.swl_oldest(l))", "4")

    def test_oldest_of_empty_log_is_nil(self):
        self.expect(PRE + "say(m.swl_oldest(l))", "nil")

    def test_newest_entry_has_latest_timestamp(self):
        self.expect(PRE + "c.now = 2; m.swl_hit(l); c.now = 4; m.swl_hit(l); "
                    "say(m.swl_newest(l))", "4")

    def test_retry_after_zero_when_room_exists(self):
        self.expect(PRE + "m.swl_hit(l, 2); say(m.swl_retry_after(l))", "0")

    def test_retry_after_waits_for_oldest(self):
        self.expect(PRE + "c.now = 1; m.swl_hit(l, 3); c.now = 4; say(m.swl_retry_after(l))",
                    "7")

    def test_retry_after_spans_several_entries(self):
        self.expect(PRE + "m.swl_hit(l); c.now = 3; m.swl_hit(l); c.now = 6; m.swl_hit(l); "
                    "c.now = 7; say(m.swl_retry_after(l, 2))", "6")

    def test_retry_after_for_full_limit_request(self):
        self.expect(PRE + "m.swl_hit(l); c.now = 2; m.swl_hit(l, 2); c.now = 3; "
                    "say(m.swl_retry_after(l, 3))", "9")

    def test_retry_after_rejects_oversized_request(self):
        self.expect(PRE + "say(try(m.swl_retry_after, l, 4))", '"err:count exceeds limit"')

    def test_retry_after_exact_fit_is_zero(self):
        self.expect(PRE + "m.swl_hit(l, 1); say(m.swl_retry_after(l, 2))", "0")

    def test_clear_forgets_all_hits(self):
        self.expect(PRE + "m.swl_hit(l, 3); m.swl_clear(l); say(m.swl_count(l), m.swl_hit(l))",
                    "0 true 2")

    def test_long_run_keeps_limit_steady(self):
        self.expect(PRE + "local ok = 0; for i = 1, 30 do c.now = i; "
                    "if m.swl_hit(l) then ok = ok + 1 end end; say(ok)", "9")

    def test_clock_argument_must_be_function(self):
        self.expect("say(try(m.swl_new, 3, 10, nil))", '"err:clock must be a function"')

    def test_newest_entry_expires_with_span(self):
        self.expect(PRE + "m.swl_hit(l); c.now = 10; say(m.swl_newest(l))", "nil")
