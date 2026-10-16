# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.luahost import LuaCase

PRE = ("local c = fake(0); local made = 0; "
       "local r = m.rlreg_new(function(k) made = made + 1; return {name = k, n = made} end, 60, c.read); ")


class T(LuaCase):
    MODULE = "registry"

    def test_new_registry_holds_nothing(self):
        self.expect(PRE + "say(m.rlreg_size(r))", "0")

    def test_factory_must_be_a_function(self):
        self.expect("local c = fake(0); say(try(m.rlreg_new, 5, 60, c.read))",
                    '"err:factory must be a function"')

    def test_idle_ttl_must_be_positive(self):
        self.expect("local c = fake(0); say(try(m.rlreg_new, function() end, 0, c.read))",
                    '"err:idle ttl must be a positive number"')

    def test_first_get_creates_limiter(self):
        self.expect(PRE + "say(m.rlreg_get(r, 'alice'))", '{n=1,name="alice"}')

    def test_second_get_returns_same_limiter(self):
        self.expect(PRE + "local a = m.rlreg_get(r, 'alice'); say(m.rlreg_get(r, 'alice') == a, made)",
                    "true 1")

    def test_each_key_gets_own_limiter(self):
        self.expect(PRE + "m.rlreg_get(r, 'alice'); m.rlreg_get(r, 'bob'); say(made, m.rlreg_size(r))",
                    "2 2")

    def test_nil_key_is_refused(self):
        self.expect(PRE + "say(try(m.rlreg_get, r, nil))", '"err:key must not be nil"')

    def test_factory_result_is_required(self):
        self.expect("local c = fake(0); local r = m.rlreg_new(function() end, 60, c.read); "
                    "say(try(m.rlreg_get, r, 'zed'))",
                    '"err:factory returned nothing for key zed"')

    def test_failed_creation_stores_nothing(self):
        self.expect("local c = fake(0); local r = m.rlreg_new(function() end, 60, c.read); "
                    "try(m.rlreg_get, r, 'zed'); say(m.rlreg_size(r), m.rlreg_has(r, 'zed'))",
                    "0 false")

    def test_peek_returns_existing_limiter(self):
        self.expect(PRE + "m.rlreg_get(r, 'a'); say(m.rlreg_peek(r, 'a'))", '{n=1,name="a"}')

    def test_peek_never_creates_a_limiter(self):
        self.expect(PRE + "say(m.rlreg_peek(r, 'a'), made, m.rlreg_size(r))", "nil 0 0")

    def test_peek_does_not_refresh_activity(self):
        self.expect(PRE + "m.rlreg_get(r, 'a'); c.now = 30; m.rlreg_peek(r, 'a'); "
                    "say(m.rlreg_idle_for(r, 'a'))", "30")

    def test_size_counts_distinct_keys(self):
        self.expect(PRE + "for _, k in ipairs({'a', 'b', 'a', 'c', 'b'}) do m.rlreg_get(r, k) end; "
                    "say(m.rlreg_size(r))", "3")

    def test_keys_come_back_sorted(self):
        self.expect(PRE + "for _, k in ipairs({'carol', 'alice', 'bob'}) do m.rlreg_get(r, k) end; "
                    "say(m.rlreg_keys(r))", '{"alice","bob","carol"}')

    def test_idle_time_since_last_get(self):
        self.expect(PRE + "m.rlreg_get(r, 'a'); c.now = 25; say(m.rlreg_idle_for(r, 'a'))", "25")

    def test_get_refreshes_idle_time(self):
        self.expect(PRE + "m.rlreg_get(r, 'a'); c.now = 25; m.rlreg_get(r, 'a'); c.now = 30; "
                    "say(m.rlreg_idle_for(r, 'a'))", "5")

    def test_idle_time_of_unknown_key_is_nil(self):
        self.expect(PRE + "say(m.rlreg_idle_for(r, 'a'))", "nil")

    def test_sweep_keeps_recent_entries(self):
        self.expect(PRE + "m.rlreg_get(r, 'a'); c.now = 59; say(m.rlreg_sweep(r))", "0 {}")

    def test_sweep_removes_entry_at_exact_ttl(self):
        self.expect(PRE + "m.rlreg_get(r, 'a'); c.now = 60; say(m.rlreg_sweep(r))", '1 {"a"}')

    def test_sweep_removes_only_idle_entries(self):
        self.expect(PRE + "m.rlreg_get(r, 'old'); c.now = 50; m.rlreg_get(r, 'new'); c.now = 70; "
                    "m.rlreg_sweep(r); say(m.rlreg_keys(r))", '{"new"}')

    def test_sweep_lists_evicted_keys_sorted(self):
        self.expect(PRE + "for _, k in ipairs({'z', 'm', 'a'}) do m.rlreg_get(r, k) end; "
                    "c.now = 100; say(m.rlreg_sweep(r))", '3 {"a","m","z"}')

    def test_sweep_clears_activity_records(self):
        self.expect(PRE + "m.rlreg_get(r, 'a'); c.now = 100; m.rlreg_sweep(r); "
                    "say(m.rlreg_idle_for(r, 'a'))", "nil")

    def test_evicted_key_gets_fresh_limiter(self):
        self.expect(PRE + "m.rlreg_get(r, 'a'); c.now = 100; m.rlreg_sweep(r); "
                    "say(m.rlreg_get(r, 'a'))", '{n=2,name="a"}')

    def test_forget_removes_a_key(self):
        self.expect(PRE + "m.rlreg_get(r, 'a'); say(m.rlreg_forget(r, 'a'), m.rlreg_size(r))",
                    "true 0")

    def test_forget_of_unknown_key_is_false(self):
        self.expect(PRE + "say(m.rlreg_forget(r, 'a'))", "false")

    def test_one_second_idle_ttl_is_accepted(self):
        self.expect("local c = fake(0); local r = m.rlreg_new(function() return 1 end, 1, c.read); "
                    "m.rlreg_get(r, 'a'); c.now = 1; say(m.rlreg_sweep(r))", '1 {"a"}')

    def test_forget_clears_idle_record(self):
        self.expect(PRE + "m.rlreg_get(r, 'a'); m.rlreg_forget(r, 'a'); "
                    "say(m.rlreg_idle_for(r, 'a'))", "nil")

    def test_clock_must_be_a_function(self):
        self.expect("say(try(m.rlreg_new, function() end, 60, false))",
                    '"err:clock must be a function"')
