# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.caldatehost import LuaCase


class Day(LuaCase):
    MODULE = "daynumber"

    def test_century_leap_day_missing_in_nineteen_hundred(self):
        self.check("run(M.cal_days_from_date, D(1900, 2, 29))", "error: day out of range")

    def test_month_zero_is_rejected_in_number_lookup(self):
        self.check("run(M.cal_days_from_date, D(2023, 0, 1))", "error: month out of range")

    def test_month_thirteen_is_rejected_in_number_lookup(self):
        self.check("run(M.cal_days_from_date, D(2023, 13, 1))", "error: month out of range")

    def test_number_lookup_rejects_non_table_date(self):
        self.check("run(M.cal_days_from_date, 'today')", "error: date must be a table")

    def test_fractional_day_number_is_rejected(self):
        self.check("run(M.cal_date_from_days, 1.5)", "error: day number must be an integer")

    def test_text_day_number_is_rejected(self):
        self.check("run(M.cal_date_from_days, '5')", "error: day number must be an integer")

    def test_full_four_century_cycle_round_trips(self):
        self.check(
            "local bad = 0 for n = 0, 146096 do "
            "if M.cal_days_from_date(M.cal_date_from_days(n)) ~= n then bad = bad + 1 end end say(bad)",
            "0",
        )

    def test_monday_maps_to_one(self):
        self.check("run(M.cal_weekday, D(2100, 3, 1))", "1")

    def test_week_of_names_runs_monday_to_sunday(self):
        self.check(
            "local r = {} for d = 1, 7 do r[d] = M.cal_weekday_name(D(2024, 1, d)) end say(r)",
            "['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']",
        )

    def test_weekday_name_for_leap_day(self):
        self.check("run(M.cal_weekday_name, D(2024, 2, 29))", "'Thursday'")

    def test_year_day_one_is_january_first(self):
        self.check("run(M.cal_date_from_year_day, 2023, 1)", "<2023-01-01>")

    def test_year_day_last_of_leap_year(self):
        self.check("run(M.cal_date_from_year_day, 2024, 366)", "<2024-12-31>")

    def test_year_day_past_leap_year_is_rejected(self):
        self.check("run(M.cal_date_from_year_day, 2024, 367)", "error: day of year out of range")

    def test_year_day_zero_is_rejected(self):
        self.check("run(M.cal_date_from_year_day, 2024, 0)", "error: day of year out of range")

    def test_year_day_fractional_is_rejected(self):
        self.check("run(M.cal_date_from_year_day, 2024, 2.5)", "error: day of year out of range")

    def test_year_day_rejects_text_year(self):
        self.check("run(M.cal_date_from_year_day, 'x', 5)", "error: date fields must be integers")
