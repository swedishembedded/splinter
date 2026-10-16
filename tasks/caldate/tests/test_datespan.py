# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.caldatehost import LuaCase


class Span(LuaCase):
    MODULE = "datespan"

    def test_earlier_date_compares_as_minus_one(self):
        self.check("run(M.cal_compare_dates, D(2023, 12, 31), D(2024, 1, 1))", "-1")

    def test_later_date_compares_as_plus_one(self):
        self.check("run(M.cal_compare_dates, D(2024, 1, 1), D(2023, 12, 31))", "1")

    def test_equal_dates_compare_as_zero(self):
        self.check("run(M.cal_compare_dates, D(2024, 1, 1), D(2024, 1, 1))", "0")

    def test_comparison_looks_at_year_before_month(self):
        self.check("run(M.cal_compare_dates, D(2023, 12, 1), D(2024, 1, 1))", "-1")

    def test_comparison_looks_at_month_before_day(self):
        self.check("run(M.cal_compare_dates, D(2024, 2, 1), D(2024, 1, 31))", "1")

    def test_seventeen_days_make_two_weeks_three_days(self):
        self.check("run(M.cal_weeks_and_days_between, D(2024, 1, 1), D(2024, 1, 18))", "2, 3")

    def test_backwards_weeks_keep_sign_on_both_parts(self):
        self.check("run(M.cal_weeks_and_days_between, D(2024, 1, 18), D(2024, 1, 1))", "-2, -3")

    def test_month_incomplete_one_day_early(self):
        self.check("run(M.cal_whole_months_between, D(2024, 1, 15), D(2024, 2, 14))", "0")

    def test_months_counted_backwards_are_negative(self):
        self.check("run(M.cal_whole_months_between, D(2024, 1, 31), D(2023, 11, 30))", "-2")

    def test_months_between_identical_dates_is_zero(self):
        self.check("run(M.cal_whole_months_between, D(2024, 1, 31), D(2024, 1, 31))", "0")

    def test_age_one_day_before_birthday(self):
        self.check("run(M.cal_age_in_years, D(1990, 5, 17), D(2024, 5, 16))", "33")

    def test_age_on_day_of_birth_is_zero(self):
        self.check("run(M.cal_age_in_years, D(1990, 5, 17), D(1990, 5, 17))", "0")

    def test_leap_day_baby_turns_one_after_february(self):
        self.check("say(M.cal_age_in_years(D(2000, 2, 29), D(2001, 2, 28)), M.cal_age_in_years(D(2000, 2, 29), D(2001, 3, 1)))", "0, 1")

    def test_age_before_birth_is_rejected(self):
        self.check("run(M.cal_age_in_years, D(1990, 5, 17), D(1990, 5, 16))", "error: date is before birth date")

    def test_first_day_of_range_is_within(self):
        self.check("run(M.cal_is_date_within, D(2024, 1, 1), D(2024, 1, 1), D(2024, 1, 31))", "true")

    def test_last_day_of_range_is_within(self):
        self.check("run(M.cal_is_date_within, D(2024, 1, 31), D(2024, 1, 1), D(2024, 1, 31))", "true")

    def test_day_before_range_is_outside(self):
        self.check("run(M.cal_is_date_within, D(2023, 12, 31), D(2024, 1, 1), D(2024, 1, 31))", "false")

    def test_day_after_range_is_outside(self):
        self.check("run(M.cal_is_date_within, D(2024, 2, 1), D(2024, 1, 1), D(2024, 1, 31))", "false")

    def test_single_day_range_contains_that_day(self):
        self.check("run(M.cal_is_date_within, D(2024, 1, 9), D(2024, 1, 9), D(2024, 1, 9))", "true")

    def test_within_rejects_reversed_range(self):
        self.check("run(M.cal_is_date_within, D(2024, 1, 5), D(2024, 1, 9), D(2024, 1, 1))", "error: range start after end")

    def test_overlap_of_partly_shared_ranges(self):
        self.check("run(M.cal_overlap_days, D(2024, 1, 1), D(2024, 1, 10), D(2024, 1, 5), D(2024, 1, 20))", "6")

    def test_adjacent_ranges_share_nothing(self):
        self.check("run(M.cal_overlap_days, D(2024, 1, 1), D(2024, 1, 10), D(2024, 1, 11), D(2024, 1, 12))", "0")

    def test_far_apart_ranges_share_nothing(self):
        self.check("run(M.cal_overlap_days, D(2024, 1, 1), D(2024, 1, 10), D(2024, 6, 1), D(2024, 6, 2))", "0")

    def test_overlap_ignores_which_range_comes_first(self):
        self.check("run(M.cal_overlap_days, D(2024, 1, 5), D(2024, 1, 20), D(2024, 1, 1), D(2024, 1, 10))", "6")

    def test_overlap_rejects_reversed_second_range(self):
        self.check("run(M.cal_overlap_days, D(2024, 1, 1), D(2024, 1, 2), D(2024, 1, 10), D(2024, 1, 1))", "error: range start after end")

    def test_earliest_date_found_in_the_middle(self):
        self.check("run(M.cal_earliest_date, {D(2024, 5, 1), D(2023, 5, 1), D(2025, 5, 1)})", "<2023-05-01>")

    def test_earliest_tie_keeps_the_first_entry(self):
        self.check("local a, b = D(2024, 1, 1), D(2024, 1, 1) say(M.cal_earliest_date({a, b}) == a)", "true")

    def test_earliest_rejects_empty_list(self):
        self.check("run(M.cal_earliest_date, {})", "error: no dates given")

    def test_latest_date_found_in_the_middle(self):
        self.check("run(M.cal_latest_date, {D(2024, 5, 1), D(2026, 5, 1), D(2025, 5, 1)})", "<2026-05-01>")

    def test_latest_tie_keeps_the_first_entry(self):
        self.check("local a, b = D(2024, 1, 1), D(2024, 1, 1) say(M.cal_latest_date({a, b}) == a)", "true")

    def test_latest_rejects_empty_list(self):
        self.check("run(M.cal_latest_date, {})", "error: no dates given")

    def test_earliest_picks_date_one_day_sooner(self):
        self.check("run(M.cal_earliest_date, {D(2024, 5, 2), D(2024, 5, 1)})", "<2024-05-01>")

    def test_latest_picks_date_one_day_later(self):
        self.check("run(M.cal_latest_date, {D(2024, 5, 1), D(2024, 5, 2)})", "<2024-05-02>")
