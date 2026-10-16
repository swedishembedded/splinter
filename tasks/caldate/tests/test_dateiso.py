# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.caldatehost import LuaCase


class Text(LuaCase):
    MODULE = "dateiso"

    def test_month_name_rejects_zero(self):
        self.check("run(M.cal_month_name, 0)", "error: month out of range")

    def test_month_name_rejects_fraction(self):
        self.check("run(M.cal_month_name, 2.5)", "error: month out of range")

    def test_month_name_rejects_text(self):
        self.check("run(M.cal_month_name, 'May')", "error: month out of range")

    def test_abbreviated_name_finds_month_number(self):
        self.check("run(M.cal_month_number_from_name, 'Sep')", "9")

    def test_abbreviation_lookup_ignores_case(self):
        self.check("run(M.cal_month_number_from_name, 'JAN')", "1")

    def test_four_letter_prefix_is_not_a_month(self):
        self.check("run(M.cal_month_number_from_name, 'Marc')", "nil")

    def test_unknown_name_gives_nil(self):
        self.check("run(M.cal_month_number_from_name, 'Smarch')", "nil")

    def test_name_lookup_rejects_number(self):
        self.check("run(M.cal_month_number_from_name, 3)", "error: month name must be a string")

    def test_iso_format_accepts_year_zero(self):
        self.check("run(M.cal_format_iso_date, D(0, 1, 1))", "'0000-01-01'")

    def test_iso_format_accepts_largest_four_digit_year(self):
        self.check("run(M.cal_format_iso_date, D(9999, 12, 31))", "'9999-12-31'")

    def test_iso_format_rejects_five_digit_year(self):
        self.check("run(M.cal_format_iso_date, D(10000, 1, 1))", "error: year out of range for formatting")

    def test_iso_format_rejects_negative_year(self):
        self.check("run(M.cal_format_iso_date, D(-1, 1, 1))", "error: year out of range for formatting")

    def test_iso_format_rejects_impossible_date(self):
        self.check("run(M.cal_format_iso_date, D(2023, 2, 29))", "error: day out of range")

    def test_iso_parse_rejects_day_zero(self):
        self.check("run(M.cal_parse_iso_date, '2024-01-00')", "error: day out of range")

    def test_iso_parse_rejects_month_zero(self):
        self.check("run(M.cal_parse_iso_date, '2024-00-10')", "error: month out of range")

    def test_iso_parse_rejects_two_digit_year(self):
        self.check("run(M.cal_parse_iso_date, '24-01-01')", "error: malformed date text")

    def test_iso_parse_rejects_number_input(self):
        self.check("run(M.cal_parse_iso_date, 20240101)", "error: date text must be a string")

    def test_try_parse_returns_date_on_success(self):
        self.check("run(M.cal_try_parse_iso_date, '2024-03-09')", "<2024-03-09>")

    def test_try_parse_reports_message_on_malformed_text(self):
        self.check("run(M.cal_try_parse_iso_date, '2024-3-9')", "nil, 'malformed date text'")

    def test_try_parse_reports_message_on_bad_day(self):
        self.check("run(M.cal_try_parse_iso_date, '2023-02-29')", "nil, 'day out of range'")

    def test_try_parse_reports_message_on_non_string(self):
        self.check("run(M.cal_try_parse_iso_date, nil)", "nil, 'date text must be a string'")

    def test_compact_parse_reads_eight_digits(self):
        self.check("run(M.cal_parse_compact_date, '20240229')", "<2024-02-29>")

    def test_compact_parse_rejects_seven_digits(self):
        self.check("run(M.cal_parse_compact_date, '2024022')", "error: malformed date text")

    def test_compact_parse_rejects_nine_digits(self):
        self.check("run(M.cal_parse_compact_date, '202402291')", "error: malformed date text")

    def test_compact_parse_rejects_bad_month(self):
        self.check("run(M.cal_parse_compact_date, '20241301')", "error: month out of range")

    def test_compact_parse_rejects_non_string(self):
        self.check("run(M.cal_parse_compact_date, {})", "error: date text must be a string")

    def test_first_ordinal_ends_in_st(self):
        self.check("run(M.cal_format_day_ordinal, 1)", "'1st'")

    def test_teens_all_end_in_th(self):
        self.check(
            "say(M.cal_format_day_ordinal(11), M.cal_format_day_ordinal(12), M.cal_format_day_ordinal(13))",
            "'11th', '12th', '13th'",
        )

    def test_twenties_use_special_endings_again(self):
        self.check(
            "say(M.cal_format_day_ordinal(21), M.cal_format_day_ordinal(22), M.cal_format_day_ordinal(23))",
            "'21st', '22nd', '23rd'",
        )

    def test_thirty_first_ordinal_ends_in_st(self):
        self.check("say(M.cal_format_day_ordinal(30), M.cal_format_day_ordinal(31))", "'30th', '31st'")

    def test_ordinal_suffix_rejects_zero_input(self):
        self.check("run(M.cal_format_day_ordinal, 0)", "error: day out of range")

    def test_ordinal_suffix_rejects_fraction_input(self):
        self.check("run(M.cal_format_day_ordinal, 1.5)", "error: day out of range")

    def test_long_format_has_unpadded_day(self):
        self.check("run(M.cal_format_long_date, D(2025, 3, 3))", "'3 March 2025'")

    def test_long_format_with_two_digit_day(self):
        self.check("run(M.cal_format_long_date, D(2024, 12, 25))", "'25 December 2024'")

    def test_long_format_rejects_impossible_date(self):
        self.check("run(M.cal_format_long_date, D(2024, 6, 31))", "error: day out of range")

    def test_ordinal_date_puts_month_first(self):
        self.check("run(M.cal_format_ordinal_date, D(2025, 3, 3))", "'March 3rd, 2025'")

    def test_ordinal_date_for_teen_day(self):
        self.check("run(M.cal_format_ordinal_date, D(2025, 7, 12))", "'July 12th, 2025'")

    def test_ordinal_date_rejects_impossible_date(self):
        self.check("run(M.cal_format_ordinal_date, D(2025, 2, 29))", "error: day out of range")
