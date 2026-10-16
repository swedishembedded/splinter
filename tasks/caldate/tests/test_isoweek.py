# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.caldatehost import LuaCase


class Week(LuaCase):
    MODULE = "isoweek"

    def test_saturday_january_first_follows_previous_year(self):
        self.check("run(M.cal_iso_week_year, D(2005, 1, 1))", "2004")

    def test_week_year_rejects_impossible_date(self):
        self.check("run(M.cal_iso_week_year, D(2023, 2, 29))", "error: day out of range")

    def test_early_january_sunday_is_week_fifty_three(self):
        self.check("run(M.cal_iso_week_number, D(2021, 1, 3))", "53")

    def test_new_year_thursday_is_week_one(self):
        self.check("run(M.cal_iso_week_number, D(2026, 1, 1))", "1")

    def test_sunday_new_year_is_week_fifty_two(self):
        self.check("run(M.cal_iso_week_number, D(2023, 1, 1))", "52")

    def test_leap_day_falls_in_week_nine(self):
        self.check("run(M.cal_iso_week_number, D(2024, 2, 29))", "9")

    def test_final_sunday_of_ordinary_year_closes_week(self):
        self.check("run(M.cal_iso_week_number, D(2024, 12, 29))", "52")

    def test_january_seventh_stays_in_week_one(self):
        self.check("run(M.cal_iso_week_number, D(2024, 1, 7))", "1")

    def test_week_number_rejects_non_table(self):
        self.check("run(M.cal_iso_week_number, nil)", "error: date must be a table")

    def test_thursday_start_year_is_long(self):
        self.check("run(M.cal_iso_weeks_in_year, 2026)", "53")

    def test_wednesday_start_leap_year_is_long(self):
        self.check("run(M.cal_iso_weeks_in_year, 2020)", "53")

    def test_monday_start_year_is_short(self):
        self.check("run(M.cal_iso_weeks_in_year, 2024)", "52")

    def test_friday_start_year_is_short(self):
        self.check("run(M.cal_iso_weeks_in_year, 2021)", "52")

    def test_saturday_start_leap_year_is_short(self):
        self.check("run(M.cal_iso_weeks_in_year, 2000)", "52")

    def test_thursday_start_leap_year_is_long(self):
        self.check("run(M.cal_iso_weeks_in_year, 2004)", "53")

    def test_tuesday_start_year_is_short(self):
        self.check("run(M.cal_iso_weeks_in_year, 2019)", "52")

    def test_weeks_in_year_rejects_text_year(self):
        self.check("run(M.cal_iso_weeks_in_year, 'y')", "error: date fields must be integers")

    def test_week_one_may_begin_in_december(self):
        self.check("run(M.cal_iso_week_monday, 2025, 1)", "<2024-12-30>")

    def test_week_one_monday_on_january_first(self):
        self.check("run(M.cal_iso_week_monday, 2024, 1)", "<2024-01-01>")

    def test_week_one_monday_after_january_third(self):
        self.check("run(M.cal_iso_week_monday, 2021, 1)", "<2021-01-04>")

    def test_week_fifty_three_monday_of_long_year(self):
        self.check("run(M.cal_iso_week_monday, 2020, 53)", "<2020-12-28>")

    def test_week_nine_monday_in_leap_year(self):
        self.check("run(M.cal_iso_week_monday, 2024, 9)", "<2024-02-26>")

    def test_week_zero_is_rejected(self):
        self.check("run(M.cal_iso_week_monday, 2024, 0)", "error: week out of range")

    def test_week_fifty_three_missing_from_short_year(self):
        self.check("run(M.cal_iso_week_monday, 2021, 53)", "error: week out of range")

    def test_fractional_week_is_rejected(self):
        self.check("run(M.cal_iso_week_monday, 2020, 2.5)", "error: week out of range")

    def test_text_week_is_rejected(self):
        self.check("run(M.cal_iso_week_monday, 2020, 'w')", "error: week out of range")

    def test_parts_for_sunday_of_week_fifty_three(self):
        self.check("run(M.cal_iso_date_from_parts, 2020, 53, 7)", "<2021-01-03>")

    def test_parts_for_monday_of_week_one(self):
        self.check("run(M.cal_iso_date_from_parts, 2024, 1, 1)", "<2024-01-01>")

    def test_parts_for_thursday_of_week_nine(self):
        self.check("run(M.cal_iso_date_from_parts, 2024, 9, 4)", "<2024-02-29>")

    def test_parts_reject_weekday_zero(self):
        self.check("run(M.cal_iso_date_from_parts, 2024, 9, 0)", "error: weekday out of range")

    def test_parts_reject_weekday_eight(self):
        self.check("run(M.cal_iso_date_from_parts, 2024, 9, 8)", "error: weekday out of range")

    def test_parts_reject_fractional_weekday(self):
        self.check("run(M.cal_iso_date_from_parts, 2024, 9, 1.5)", "error: weekday out of range")

    def test_parts_reject_text_weekday(self):
        self.check("run(M.cal_iso_date_from_parts, 2024, 9, 'm')", "error: weekday out of range")

    def test_parts_report_week_problem_for_bad_week(self):
        self.check("run(M.cal_iso_date_from_parts, 2021, 53, 1)", "error: week out of range")

    def test_week_dates_list_a_full_week(self):
        self.check(
            "run(M.cal_iso_week_dates, 2024, 9)",
            "[<2024-02-26>, <2024-02-27>, <2024-02-28>, <2024-02-29>, <2024-03-01>, <2024-03-02>, <2024-03-03>]",
        )

    def test_week_dates_cross_year_boundary(self):
        self.check(
            "run(M.cal_iso_week_dates, 2020, 53)",
            "[<2020-12-28>, <2020-12-29>, <2020-12-30>, <2020-12-31>, <2021-01-01>, <2021-01-02>, <2021-01-03>]",
        )

    def test_week_dates_reject_missing_week(self):
        self.check("run(M.cal_iso_week_dates, 2021, 53)", "error: week out of range")

    def test_label_for_leap_day_thursday(self):
        self.check("run(M.cal_iso_week_label, D(2024, 2, 29))", "'2024-W09-4'")

    def test_label_for_week_fifty_three_sunday(self):
        self.check("run(M.cal_iso_week_label, D(2021, 1, 3))", "'2020-W53-7'")

    def test_label_pads_week_with_leading_zero(self):
        self.check("run(M.cal_iso_week_label, D(2024, 1, 1))", "'2024-W01-1'")
