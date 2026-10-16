# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.luahost import LuaCase


class T(LuaCase):
    MODULE = "waittime"

    def test_tokens_wait_is_deficit_over_rate(self):
        self.expect("say(m.wait_for_tokens(1, 5, 2))", "2")

    def test_no_wait_when_tokens_suffice(self):
        self.expect("say(m.wait_for_tokens(5, 3, 2))", "0")

    def test_no_wait_when_tokens_match_need(self):
        self.expect("say(m.wait_for_tokens(4, 4, 2))", "0")

    def test_token_wait_needs_positive_rate(self):
        self.expect("say(try(m.wait_for_tokens, 0, 1, 0))", '"err:rate must be a positive number"')

    def test_level_wait_for_overflowing_amount(self):
        self.expect("say(m.wait_for_level(8, 10, 5, 2))", "1.5")

    def test_level_wait_zero_when_amount_fits(self):
        self.expect("say(m.wait_for_level(5, 10, 5, 2))", "0")

    def test_level_wait_allows_amount_equal_capacity(self):
        self.expect("say(m.wait_for_level(0, 10, 10, 2))", "0")

    def test_level_wait_rejects_oversized_amount(self):
        self.expect("say(try(m.wait_for_level, 0, 10, 11, 2))",
                    '"err:amount exceeds capacity"')

    def test_level_wait_needs_positive_leak(self):
        self.expect("say(try(m.wait_for_level, 0, 10, 1, 0))",
                    '"err:leak rate must be a positive number"')

    def test_window_wait_until_next_boundary(self):
        self.expect("say(m.wait_for_window(23, 10))", "7")

    def test_window_wait_at_boundary_is_full_width(self):
        self.expect("say(m.wait_for_window(30, 10))", "10")

    def test_window_wait_needs_positive_width(self):
        self.expect("say(try(m.wait_for_window, 5, 0))", '"err:width must be a positive number"')

    def test_sequence_wait_serves_count_at_rate(self):
        self.expect("say(m.wait_for_sequence(9, 3))", "3")

    def test_sequence_wait_needs_positive_rate(self):
        self.expect("say(try(m.wait_for_sequence, 3, 0))",
                    '"err:rate must be a positive number"')

    def test_spacing_divides_window_by_limit(self):
        self.expect("say(m.wait_spacing(4, 60))", "15")

    def test_spacing_needs_positive_limit(self):
        self.expect("say(try(m.wait_spacing, 0, 60))", '"err:limit must be a positive number"')

    def test_longest_picks_the_maximum(self):
        self.expect("say(m.wait_longest({2, 9, 4}))", "9")

    def test_longest_looks_at_last_entry(self):
        self.expect("say(m.wait_longest({1, 2, 8}))", "8")

    def test_longest_looks_at_first_entry(self):
        self.expect("say(m.wait_longest({8, 2, 1}))", "8")

    def test_longest_of_empty_list_fails(self):
        self.expect("say(try(m.wait_longest, {}))", '"err:no waits given"')

    def test_round_up_to_step_multiple(self):
        self.expect("say(m.wait_round_up(2.3, 0.5))", "2.5")

    def test_round_up_keeps_exact_multiples(self):
        self.expect("say(m.wait_round_up(3, 0.5))", "3")

    def test_round_up_needs_positive_step(self):
        self.expect("say(try(m.wait_round_up, 1, 0))", '"err:step must be a positive number"')

    def test_clamp_raises_small_wait_to_floor(self):
        self.expect("say(m.wait_clamp(1, 5, 30))", "5")

    def test_clamp_lowers_large_wait_to_ceiling(self):
        self.expect("say(m.wait_clamp(99, 5, 30))", "30")

    def test_clamp_leaves_wait_in_range(self):
        self.expect("say(m.wait_clamp(12, 5, 30))", "12")

    def test_clamp_accepts_equal_bounds(self):
        self.expect("say(m.wait_clamp(12, 7, 7))", "7")

    def test_clamp_rejects_inverted_bounds(self):
        self.expect("say(try(m.wait_clamp, 1, 9, 3))", '"err:low exceeds high"')

    def test_header_seconds_round_up(self):
        self.expect("say(m.wait_header_seconds(2.1))", '"3"')

    def test_budget_accepts_wait_at_limit(self):
        self.expect("say(m.wait_fits_budget(5, 5), m.wait_fits_budget(5.5, 5))", "true false")
