# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.caldatehost import LuaCase


class Shift(LuaCase):
    MODULE = "dateshift"

    def test_fractional_day_count_is_rejected(self):
        self.check("run(M.cal_add_days, D(2023, 1, 1), 0.5)", "error: count must be an integer")

    def test_text_day_count_is_rejected(self):
        self.check("run(M.cal_add_days, D(2023, 1, 1), '3')", "error: count must be an integer")

    def test_adding_one_week_moves_seven_days(self):
        self.check("run(M.cal_add_weeks, D(2023, 12, 28), 1)", "<2024-01-04>")

    def test_subtracting_two_weeks_moves_fourteen_days(self):
        self.check("run(M.cal_add_weeks, D(2024, 3, 10), -2)", "<2024-02-25>")

    def test_one_seventh_week_count_is_rejected(self):
        self.check("run(M.cal_add_weeks, D(2024, 3, 10), 1 / 7)", "error: count must be an integer")

    def test_fractional_month_count_is_rejected(self):
        self.check("run(M.cal_add_months, D(2023, 1, 1), 1.5)", "error: count must be an integer")

    def test_month_addition_rejects_impossible_date(self):
        self.check("run(M.cal_add_months, D(2023, 2, 31), 1)", "error: day out of range")

    def test_month_addition_rejects_non_table(self):
        self.check("run(M.cal_add_months, 7, 1)", "error: date must be a table")

    def test_leap_day_lands_on_february_twenty_eighth(self):
        self.check("run(M.cal_add_years, D(2024, 2, 29), 1)", "<2025-02-28>")

    def test_leap_day_minus_one_year_clamps(self):
        self.check("run(M.cal_add_years, D(2024, 2, 29), -1)", "<2023-02-28>")

    def test_fractional_year_count_is_rejected(self):
        self.check("run(M.cal_add_years, D(2023, 6, 15), 0.5)", "error: count must be an integer")

    def test_month_start_resets_day_to_first(self):
        self.check("run(M.cal_month_start, D(2023, 9, 17))", "<2023-09-01>")

    def test_month_start_rejects_impossible_date(self):
        self.check("run(M.cal_month_start, D(2023, 9, 31))", "error: day out of range")

    def test_month_end_of_leap_february(self):
        self.check("run(M.cal_month_end, D(2024, 2, 3))", "<2024-02-29>")

    def test_month_end_of_thirty_day_month(self):
        self.check("run(M.cal_month_end, D(2023, 11, 3))", "<2023-11-30>")

    def test_month_end_rejects_impossible_date(self):
        self.check("run(M.cal_month_end, D(2023, 2, 29))", "error: day out of range")

    def test_wednesday_to_next_monday(self):
        self.check("run(M.cal_next_weekday_on_or_after, D(2024, 1, 3), 1)", "<2024-01-08>")

    def test_next_weekday_one_day_ahead(self):
        self.check("run(M.cal_next_weekday_on_or_after, D(2024, 1, 3), 4)", "<2024-01-04>")

    def test_next_weekday_sunday_from_wednesday(self):
        self.check("run(M.cal_next_weekday_on_or_after, D(2024, 1, 3), 7)", "<2024-01-07>")

    def test_next_weekday_tuesday_wraps_the_week(self):
        self.check("run(M.cal_next_weekday_on_or_after, D(2024, 1, 3), 2)", "<2024-01-09>")

    def test_next_weekday_rejects_zero(self):
        self.check("run(M.cal_next_weekday_on_or_after, D(2024, 1, 3), 0)", "error: weekday out of range")

    def test_next_weekday_rejects_fractional(self):
        self.check("run(M.cal_next_weekday_on_or_after, D(2024, 1, 3), 2.5)", "error: weekday out of range")

    def test_wednesday_back_to_monday(self):
        self.check("run(M.cal_previous_weekday_on_or_before, D(2024, 1, 3), 1)", "<2024-01-01>")

    def test_previous_weekday_friday_crosses_year_end(self):
        self.check("run(M.cal_previous_weekday_on_or_before, D(2024, 1, 3), 5)", "<2023-12-29>")

    def test_previous_weekday_thursday_is_six_days_back(self):
        self.check("run(M.cal_previous_weekday_on_or_before, D(2024, 1, 3), 4)", "<2023-12-28>")

    def test_previous_weekday_rejects_zero(self):
        self.check("run(M.cal_previous_weekday_on_or_before, D(2024, 1, 3), 0)", "error: weekday out of range")

    def test_previous_weekday_rejects_eight(self):
        self.check("run(M.cal_previous_weekday_on_or_before, D(2024, 1, 3), 8)", "error: weekday out of range")
