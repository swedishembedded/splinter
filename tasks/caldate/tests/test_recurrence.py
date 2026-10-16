# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.caldatehost import LuaCase


class Rec(LuaCase):
    MODULE = "recurrence"

    def test_daily_series_runs_through_leap_day(self):
        self.check(
            "run(M.cal_every_n_days, D(2024, 2, 27), 1, 4)",
            "[<2024-02-27>, <2024-02-28>, <2024-02-29>, <2024-03-01>]",
        )

    def test_weekly_series_crosses_new_year(self):
        self.check(
            "run(M.cal_every_n_days, D(2023, 12, 25), 7, 3)",
            "[<2023-12-25>, <2024-01-01>, <2024-01-08>]",
        )

    def test_zero_count_gives_empty_series(self):
        self.check("run(M.cal_every_n_days, D(2024, 1, 1), 3, 0)", "[]")

    def test_zero_step_is_rejected(self):
        self.check("run(M.cal_every_n_days, D(2024, 1, 1), 0, 3)", "error: step must be a positive integer")

    def test_negative_step_is_rejected(self):
        self.check("run(M.cal_every_n_days, D(2024, 1, 1), -1, 3)", "error: step must be a positive integer")

    def test_negative_count_is_rejected(self):
        self.check("run(M.cal_every_n_days, D(2024, 1, 1), 1, -1)", "error: count must be a non-negative integer")

    def test_fractional_count_is_rejected(self):
        self.check("run(M.cal_every_n_days, D(2024, 1, 1), 1, 2.5)", "error: count must be a non-negative integer")

    def test_bounded_series_stops_short_off_step_end(self):
        self.check(
            "run(M.cal_every_n_days_until, D(2024, 1, 1), 10, D(2024, 1, 30))",
            "[<2024-01-01>, <2024-01-11>, <2024-01-21>]",
        )

    def test_bounded_series_with_equal_ends_yields_start(self):
        self.check("run(M.cal_every_n_days_until, D(2024, 1, 1), 5, D(2024, 1, 1))", "[<2024-01-01>]")

    def test_bounded_series_rejects_zero_step(self):
        self.check("run(M.cal_every_n_days_until, D(2024, 1, 1), 0, D(2024, 2, 1))", "error: step must be a positive integer")

    def test_occurrence_zero_is_rejected(self):
        self.check("run(M.cal_nth_weekday_of_month, 2024, 3, 1, 0)", "error: occurrence must be between 1 and 5")

    def test_fractional_occurrence_is_rejected(self):
        self.check("run(M.cal_nth_weekday_of_month, 2024, 3, 1, 1.5)", "error: occurrence must be between 1 and 5")

    def test_weekday_zero_is_rejected_for_month_lookup(self):
        self.check("run(M.cal_nth_weekday_of_month, 2024, 3, 0, 1)", "error: weekday out of range")

    def test_weekday_eight_is_rejected_for_month_lookup(self):
        self.check("run(M.cal_nth_weekday_of_month, 2024, 3, 8, 1)", "error: weekday out of range")

    def test_nth_weekday_in_february_of_leap_year(self):
        self.check("run(M.cal_nth_weekday_of_month, 2024, 2, 4, 5)", "<2024-02-29>")

    def test_last_monday_falls_on_fourth_occurrence(self):
        self.check("run(M.cal_last_weekday_of_month, 2024, 3, 1)", "<2024-03-25>")

    def test_last_sunday_falls_on_fifth_occurrence(self):
        self.check("run(M.cal_last_weekday_of_month, 2024, 3, 7)", "<2024-03-31>")

    def test_last_thursday_of_leap_february(self):
        self.check("run(M.cal_last_weekday_of_month, 2024, 2, 4)", "<2024-02-29>")

    def test_last_tuesday_of_common_february(self):
        self.check("run(M.cal_last_weekday_of_month, 2023, 2, 2)", "<2023-02-28>")

    def test_monthly_series_skips_occurrence_before_start(self):
        self.check(
            "run(M.cal_monthly_nth_weekday, D(2024, 1, 15), 2, 2, 3)",
            "[<2024-02-13>, <2024-03-12>, <2024-04-09>]",
        )

    def test_monthly_series_includes_occurrence_on_start_day(self):
        self.check(
            "run(M.cal_monthly_nth_weekday, D(2024, 1, 9), 2, 2, 3)",
            "[<2024-01-09>, <2024-02-13>, <2024-03-12>]",
        )

    def test_monthly_series_skips_months_without_fifth_friday(self):
        self.check(
            "run(M.cal_monthly_nth_weekday, D(2024, 1, 1), 5, 5, 3)",
            "[<2024-03-29>, <2024-05-31>, <2024-08-30>]",
        )

    def test_monthly_series_wraps_into_next_year(self):
        self.check(
            "run(M.cal_monthly_nth_weekday, D(2024, 11, 20), 1, 1, 3)",
            "[<2024-12-02>, <2025-01-06>, <2025-02-03>]",
        )

    def test_monthly_series_rejects_negative_count(self):
        self.check("run(M.cal_monthly_nth_weekday, D(2024, 1, 1), 1, 1, -1)", "error: count must be a non-negative integer")

    def test_monthly_series_checks_weekday_without_count(self):
        self.check("run(M.cal_monthly_nth_weekday, D(2024, 1, 1), 9, 1, 0)", "error: weekday out of range")

    def test_monthly_series_checks_occurrence_without_count(self):
        self.check("run(M.cal_monthly_nth_weekday, D(2024, 1, 1), 1, 7, 0)", "error: occurrence must be between 1 and 5")

    def test_weekly_pattern_with_two_weekdays(self):
        self.check(
            "run(M.cal_weekly_on_days, D(2024, 1, 3), {1, 3}, 4)",
            "[<2024-01-03>, <2024-01-08>, <2024-01-10>, <2024-01-15>]",
        )

    def test_weekly_pattern_ignores_list_order(self):
        self.check(
            "run(M.cal_weekly_on_days, D(2024, 1, 3), {3, 1}, 3)",
            "[<2024-01-03>, <2024-01-08>, <2024-01-10>]",
        )

    def test_weekly_pattern_includes_start_when_it_matches(self):
        self.check(
            "run(M.cal_weekly_on_days, D(2024, 1, 5), {5}, 2)",
            "[<2024-01-05>, <2024-01-12>]",
        )

    def test_weekly_pattern_with_zero_count_is_empty(self):
        self.check("run(M.cal_weekly_on_days, D(2024, 1, 5), {5}, 0)", "[]")

    def test_weekly_pattern_rejects_empty_list(self):
        self.check("run(M.cal_weekly_on_days, D(2024, 1, 5), {}, 2)", "error: weekday list is empty")

    def test_weekly_pattern_rejects_bad_weekday(self):
        self.check("run(M.cal_weekly_on_days, D(2024, 1, 5), {1, 8}, 2)", "error: weekday out of range")

    def test_weekly_pattern_rejects_fractional_count(self):
        self.check("run(M.cal_weekly_on_days, D(2024, 1, 5), {1}, 1.5)", "error: count must be a non-negative integer")

    def test_count_fridays_in_march(self):
        self.check("run(M.cal_count_weekday_in_range, D(2024, 3, 1), D(2024, 3, 31), 5)", "5")

    def test_count_mondays_in_march(self):
        self.check("run(M.cal_count_weekday_in_range, D(2024, 3, 1), D(2024, 3, 31), 1)", "4")

    def test_count_sundays_in_march(self):
        self.check("run(M.cal_count_weekday_in_range, D(2024, 3, 1), D(2024, 3, 31), 7)", "5")

    def test_count_thursdays_in_march(self):
        self.check("run(M.cal_count_weekday_in_range, D(2024, 3, 1), D(2024, 3, 31), 4)", "4")

    def test_count_in_single_day_range_that_matches(self):
        self.check("run(M.cal_count_weekday_in_range, D(2024, 3, 1), D(2024, 3, 1), 5)", "1")

    def test_count_in_single_day_range_that_misses(self):
        self.check("run(M.cal_count_weekday_in_range, D(2024, 3, 1), D(2024, 3, 1), 6)", "0")

    def test_count_when_range_ends_on_first_hit(self):
        self.check("run(M.cal_count_weekday_in_range, D(2024, 3, 1), D(2024, 3, 4), 1)", "1")

    def test_count_misses_when_range_ends_early(self):
        self.check("run(M.cal_count_weekday_in_range, D(2024, 3, 1), D(2024, 3, 3), 1)", "0")

    def test_count_rejects_reversed_range(self):
        self.check("run(M.cal_count_weekday_in_range, D(2024, 3, 2), D(2024, 3, 1), 1)", "error: range start after end")

    def test_count_rejects_bad_weekday(self):
        self.check("run(M.cal_count_weekday_in_range, D(2024, 3, 1), D(2024, 3, 2), 0)", "error: weekday out of range")
