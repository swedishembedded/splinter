# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.caldatehost import LuaCase


class Leap(LuaCase):
    MODULE = "leapyear"

    def test_fractional_year_is_rejected_in_leap_query(self):
        self.check("run(M.cal_is_leap_year, 2024.5)", "error: year must be an integer")

    def test_text_year_is_rejected_in_leap_query(self):
        self.check("run(M.cal_is_leap_year, '2024')", "error: year must be an integer")

    def test_only_september_april_june_november_are_short(self):
        self.check(
            "local r = {} for m = 1, 12 do r[m] = M.cal_days_in_month(2023, m) end say(r)",
            "[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]",
        )

    def test_month_zero_is_out_of_range(self):
        self.check("run(M.cal_days_in_month, 2023, 0)", "error: month out of range")

    def test_month_length_rejects_fractional_year(self):
        self.check("run(M.cal_days_in_month, 1.5, 1)", "error: year must be an integer")

    def test_leap_year_length_in_days(self):
        self.check("run(M.cal_days_in_year, 2024)", "366")

    def test_common_year_length_in_days(self):
        self.check("run(M.cal_days_in_year, 2023)", "365")

    def test_month_lengths_list_for_leap_year(self):
        self.check(
            "run(M.cal_month_lengths, 2000)",
            "[31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]",
        )

    def test_month_lengths_reject_bad_year(self):
        self.check("run(M.cal_month_lengths, 'x')", "error: year must be an integer")

    def test_day_zero_is_invalid_date(self):
        self.check("run(M.cal_is_valid_date, D(2023, 4, 0))", "false")

    def test_non_table_is_invalid_date(self):
        self.check("run(M.cal_is_valid_date, '2023-01-01')", "false")

    def test_assert_date_rejects_non_table(self):
        self.check("run(M.cal_assert_date, 5)", "error: date must be a table")

    def test_assert_date_rejects_missing_year(self):
        self.check("run(M.cal_assert_date, {month = 1, day = 1})", "error: date fields must be integers")

    def test_assert_date_rejects_missing_month(self):
        self.check("run(M.cal_assert_date, {year = 2020, day = 1})", "error: date fields must be integers")

    def test_leap_years_between_counts_inclusive_ends(self):
        self.check("run(M.cal_leap_years_between, 2000, 2004)", "2")

    def test_leap_years_between_single_common_year(self):
        self.check("run(M.cal_leap_years_between, 2023, 2023)", "0")

    def test_leap_years_between_single_leap_year(self):
        self.check("run(M.cal_leap_years_between, 2024, 2024)", "1")

    def test_leap_years_between_long_span(self):
        self.check("run(M.cal_leap_years_between, 1, 2000)", "485")

    def test_leap_years_between_rejects_reversed_range(self):
        self.check("run(M.cal_leap_years_between, 2005, 2004)", "error: range start after end")

    def test_leap_years_between_rejects_fractional_bounds(self):
        self.check("say(pcall(M.cal_leap_years_between, 1.5, 4))", "false, 'year must be an integer'")

    def test_leap_years_between_rejects_fractional_end(self):
        self.check("say(pcall(M.cal_leap_years_between, 1, 4.5))", "false, 'year must be an integer'")

    def test_next_leap_year_is_strictly_later(self):
        self.check("run(M.cal_next_leap_year, 2024)", "2028")

    def test_next_leap_year_from_year_before_leap(self):
        self.check("run(M.cal_next_leap_year, 2023)", "2024")

    def test_next_leap_year_skips_century_gap(self):
        self.check("run(M.cal_next_leap_year, 1896)", "1904")

    def test_next_leap_year_rejects_text(self):
        self.check("run(M.cal_next_leap_year, 'a')", "error: year must be an integer")
