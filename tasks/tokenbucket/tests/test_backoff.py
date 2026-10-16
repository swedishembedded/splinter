# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.luahost import LuaCase

PRE = "local p = m.bko_policy(1, 2, 20, nil, 6); "


class BackoffTests(LuaCase):
    MODULE = "backoff"

    def test_policy_keeps_its_settings(self):
        self.expect("local p = m.bko_policy(1, 2, 20, nil, 6); "
                    "say(p.base, p.factor, p.max_delay, p.max_attempts)", "1 2 20 6")

    def test_base_must_be_positive(self):
        self.expect("say(try(m.bko_policy, 0, 2, 20, nil, 3))",
                    '"err:base must be a positive number"')

    def test_factor_below_one_is_refused(self):
        self.expect("say(try(m.bko_policy, 1, 0.5, 20, nil, 3))",
                    '"err:factor must be at least 1"')

    def test_factor_of_exactly_one_is_fine(self):
        self.expect("local p = m.bko_policy(3, 1, 20, nil, 3); say(m.bko_schedule(p))",
                    "{3,3,3}")

    def test_max_delay_below_base_is_refused(self):
        self.expect("say(try(m.bko_policy, 5, 2, 4, nil, 3))",
                    '"err:max delay must not be below base"')

    def test_max_delay_equal_to_base_is_fine(self):
        self.expect("local p = m.bko_policy(5, 2, 5, nil, 2); say(m.bko_schedule(p))", "{5,5}")

    def test_jitter_must_be_function_or_nil(self):
        self.expect("say(try(m.bko_policy, 1, 2, 20, 'wide', 3))",
                    '"err:jitter must be a function"')

    def test_zero_attempts_are_refused(self):
        self.expect("say(try(m.bko_policy, 1, 2, 20, nil, 0))",
                    '"err:max attempts must be a positive integer"')

    def test_fractional_attempt_count_is_refused(self):
        self.expect("say(try(m.bko_policy, 1, 2, 20, nil, 2.5))",
                    '"err:max attempts must be a positive integer"')

    def test_single_attempt_policy_is_allowed(self):
        self.expect("local p = m.bko_policy(1, 2, 20, nil, 1); say(m.bko_schedule(p))", "{1}")

    def test_first_delay_equals_base(self):
        self.expect(PRE + "say(m.bko_delay(p, 1))", "1")

    def test_delay_multiplies_by_factor(self):
        self.expect(PRE + "say(m.bko_delay(p, 2), m.bko_delay(p, 3), m.bko_delay(p, 4))",
                    "2 4 8")

    def test_delay_stops_at_ceiling(self):
        self.expect(PRE + "say(m.bko_delay(p, 5), m.bko_delay(p, 6))", "16 20")

    def test_huge_attempt_stays_at_ceiling(self):
        self.expect(PRE + "say(m.bko_delay(p, 5000))", "20")

    def test_fractional_factor_grows_slowly(self):
        self.expect("local p = m.bko_policy(2, 1.5, 100, nil, 4); say(m.bko_delay(p, 3))", "4.5")

    def test_attempt_zero_is_refused(self):
        self.expect(PRE + "say(try(m.bko_delay, p, 0))",
                    '"err:attempt must be a positive integer"')

    def test_fractional_attempt_is_refused(self):
        self.expect(PRE + "say(try(m.bko_delay, p, 1.5))",
                    '"err:attempt must be a positive integer"')

    def test_jitter_receives_delay_and_attempt(self):
        self.expect("local seen; local p = m.bko_policy(1, 2, 20, "
                    "function(d, a) seen = d .. '/' .. a; return d end, 6); "
                    "m.bko_delay(p, 3); say(seen)", '"4/3"')

    def test_jitter_result_replaces_delay(self):
        self.expect("local p = m.bko_policy(1, 2, 20, function(d) return d / 2 end, 6); "
                    "say(m.bko_delay(p, 4))", "4")

    def test_jitter_cannot_exceed_ceiling(self):
        self.expect("local p = m.bko_policy(1, 2, 20, function(d) return d + 15 end, 6); "
                    "say(m.bko_delay(p, 4))", "20")

    def test_zero_jitter_result_is_allowed(self):
        self.expect("local p = m.bko_policy(1, 2, 20, function() return 0 end, 6); "
                    "say(m.bko_delay(p, 4))", "0")

    def test_negative_jitter_result_is_refused(self):
        self.expect("local p = m.bko_policy(1, 2, 20, function() return -1 end, 6); "
                    "say(try(m.bko_delay, p, 1))",
                    '"err:jitter must return a non-negative number"')

    def test_non_numeric_jitter_result_is_refused(self):
        self.expect("local p = m.bko_policy(1, 2, 20, function() return 'x' end, 6); "
                    "say(try(m.bko_delay, p, 1))",
                    '"err:jitter must return a non-negative number"')

    def test_schedule_lists_every_attempt(self):
        self.expect(PRE + "say(m.bko_schedule(p))", "{1,2,4,8,16,20}")

    def test_schedule_applies_jitter_to_each_entry(self):
        self.expect("local p = m.bko_policy(1, 2, 20, function(d, a) return d + a end, 4); "
                    "say(m.bko_schedule(p))", "{2,4,7,12}")

    def test_total_adds_up_schedule(self):
        self.expect(PRE + "say(m.bko_total(p))", "51")

    def test_stepper_yields_delays_in_order(self):
        self.expect(PRE + "local s = m.bko_start(p); say(m.bko_step(s), m.bko_step(s), m.bko_step(s))",
                    "1 2 4")

    def test_stepper_ends_after_last_attempt(self):
        self.expect("local p = m.bko_policy(1, 2, 20, nil, 2); local s = m.bko_start(p); "
                    "m.bko_step(s); m.bko_step(s); say(m.bko_step(s))", "nil")

    def test_remaining_attempts_count_down(self):
        self.expect(PRE + "local s = m.bko_start(p); say(m.bko_remaining(s)); m.bko_step(s); "
                    "say(m.bko_remaining(s))", "6\n5")

    def test_rewind_restarts_the_schedule(self):
        self.expect(PRE + "local s = m.bko_start(p); m.bko_step(s); m.bko_step(s); "
                    "m.bko_rewind(s); say(m.bko_step(s), m.bko_remaining(s))", "1 5")

    def test_fraction_jitter_scales_delay(self):
        self.expect("say(m.bko_fraction_jitter(0.25)(8))", "2")

    def test_fraction_jitter_accepts_both_ends(self):
        self.expect("say(m.bko_fraction_jitter(0)(8), m.bko_fraction_jitter(1)(8))", "0 8")

    def test_fraction_above_one_is_refused(self):
        self.expect("say(try(m.bko_fraction_jitter, 1.5))",
                    '"err:fraction must be between 0 and 1"')

    def test_negative_fraction_is_refused(self):
        self.expect("say(try(m.bko_fraction_jitter, -0.1))",
                    '"err:fraction must be between 0 and 1"')

    def test_offset_jitter_adds_constant(self):
        self.expect("say(m.bko_offset_jitter(3)(8))", "11")

    def test_attempt_jitter_adds_step_per_attempt(self):
        self.expect("say(m.bko_attempt_jitter(0.5)(8, 4))", "10")
