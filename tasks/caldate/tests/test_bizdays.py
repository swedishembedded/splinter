# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.caldatehost import LuaCase

HOLIDAYS = "local H = M.cal_holiday_set({D(2024, 12, 25), D(2024, 12, 26), D(2025, 1, 1)})\n"


class Biz(LuaCase):
    MODULE = "bizdays"

    def check_h(self, body, expected):
        self.check(HOLIDAYS + body, expected)

    def test_weekend_is_not_business_day(self):
        self.check_h("run(M.cal_is_business_day, H, D(2024, 12, 28))", "false")

    def test_next_business_day_after_friday_is_monday(self):
        self.check_h("run(M.cal_next_business_day, H, D(2024, 12, 20))", "<2024-12-23>")

    def test_next_business_day_skips_two_holidays(self):
        self.check_h("run(M.cal_next_business_day, H, D(2024, 12, 24))", "<2024-12-27>")

    def test_next_business_day_from_friday_skips_weekend(self):
        self.check_h("run(M.cal_next_business_day, H, D(2024, 12, 27))", "<2024-12-30>")

    def test_next_business_day_skips_new_year_holiday(self):
        self.check_h("run(M.cal_next_business_day, H, D(2024, 12, 31))", "<2025-01-02>")

    def test_next_business_day_is_strictly_later(self):
        self.check_h("run(M.cal_next_business_day, H, D(2024, 12, 23))", "<2024-12-24>")

    def test_previous_business_day_skips_two_holidays(self):
        self.check_h("run(M.cal_previous_business_day, H, D(2024, 12, 27))", "<2024-12-24>")

    def test_previous_business_day_from_monday_is_friday(self):
        self.check_h("run(M.cal_previous_business_day, H, D(2024, 12, 23))", "<2024-12-20>")

    def test_previous_business_day_across_new_year(self):
        self.check_h("run(M.cal_previous_business_day, H, D(2025, 1, 2))", "<2024-12-31>")

    def test_previous_business_day_from_monday_after_holidays(self):
        self.check_h("run(M.cal_previous_business_day, H, D(2024, 12, 30))", "<2024-12-27>")

    def test_roll_forward_keeps_open_day(self):
        self.check_h("run(M.cal_roll_forward_to_business_day, H, D(2024, 12, 20))", "<2024-12-20>")

    def test_roll_forward_moves_saturday_to_monday(self):
        self.check_h("run(M.cal_roll_forward_to_business_day, H, D(2024, 12, 21))", "<2024-12-23>")

    def test_roll_forward_leaves_holiday_for_open_day(self):
        self.check_h("run(M.cal_roll_forward_to_business_day, H, D(2024, 12, 25))", "<2024-12-27>")

    def test_one_business_day_after_friday(self):
        self.check_h("run(M.cal_add_business_days, H, D(2024, 12, 20), 1)", "<2024-12-23>")

    def test_three_business_days_hop_over_holidays(self):
        self.check_h("run(M.cal_add_business_days, H, D(2024, 12, 20), 3)", "<2024-12-27>")

    def test_five_business_days_cross_holidays_and_weekend(self):
        self.check_h("run(M.cal_add_business_days, H, D(2024, 12, 20), 5)", "<2024-12-31>")

    def test_one_business_day_back_skips_holidays(self):
        self.check_h("run(M.cal_add_business_days, H, D(2024, 12, 27), -1)", "<2024-12-24>")

    def test_three_business_days_back(self):
        self.check_h("run(M.cal_add_business_days, H, D(2024, 12, 27), -3)", "<2024-12-20>")

    def test_one_business_day_from_saturday(self):
        self.check_h("run(M.cal_add_business_days, H, D(2024, 12, 21), 1)", "<2024-12-23>")

    def test_adding_business_days_without_holidays(self):
        self.check("run(M.cal_add_business_days, M.cal_holiday_set({}), D(2024, 12, 20), 10)", "<2025-01-03>")

    def test_fractional_business_day_count_is_rejected(self):
        self.check_h("run(M.cal_add_business_days, H, D(2024, 12, 20), 1.5)", "error: count must be an integer")

    def test_text_business_day_count_is_rejected(self):
        self.check_h("run(M.cal_add_business_days, H, D(2024, 12, 20), 'x')", "error: count must be an integer")

    def test_business_days_in_week_around_holidays(self):
        self.check_h("run(M.cal_business_days_between, H, D(2024, 12, 20), D(2024, 12, 31))", "5")

    def test_business_days_count_is_negative_when_reversed(self):
        self.check_h("run(M.cal_business_days_between, H, D(2024, 12, 31), D(2024, 12, 20))", "-5")

    def test_business_count_excludes_the_end_date(self):
        self.check_h("run(M.cal_business_days_between, H, D(2024, 12, 23), D(2024, 12, 24))", "1")

    def test_business_count_includes_the_start_date(self):
        self.check_h("run(M.cal_business_days_between, H, D(2024, 12, 20), D(2024, 12, 21))", "1")

    def test_weekend_span_has_no_business_days(self):
        self.check_h("run(M.cal_business_days_between, H, D(2024, 12, 21), D(2024, 12, 23))", "0")

    def test_business_days_across_new_year_holiday(self):
        self.check_h("run(M.cal_business_days_between, H, D(2024, 12, 20), D(2025, 1, 3))", "7")
