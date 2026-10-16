# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.luahost import LuaCase


class T(LuaCase):
    MODULE = "duration"

    def test_second_unit_is_thousand(self):
        self.expect("say(m.dur_unit_millis('s'))", "1000")

    def test_minute_unit_is_sixty_seconds(self):
        self.expect("say(m.dur_unit_millis('m'))", "60000")

    def test_hour_unit_is_sixty_minutes(self):
        self.expect("say(m.dur_unit_millis('h'))", "3600000")

    def test_day_unit_is_twenty_four_hours(self):
        self.expect("say(m.dur_unit_millis('d'))", "86400000")

    def test_unknown_unit_is_named_in_error(self):
        self.expect("say(try(m.dur_unit_millis, 'y'))", '"err:unknown unit: y"')

    def test_total_millis_of_seconds(self):
        self.expect("say(m.dur_total_millis('45s'))", "45000")

    def test_parse_minutes_and_seconds(self):
        self.expect("say(m.dur_parse('1m30s'))", "90")

    def test_parse_days_with_hours(self):
        self.expect("say(m.dur_parse('1d2h'))", "93600")

    def test_parse_every_unit_together(self):
        self.expect("say(m.dur_parse('1d1h1m1s1ms'))", "90061.001")

    def test_parse_milliseconds_as_fraction(self):
        self.expect("say(m.dur_parse('1500ms'))", "1.5")

    def test_parse_zero_seconds_text(self):
        self.expect("say(m.dur_parse('0s'))", "0")

    def test_parse_ms_not_confused_with_minutes(self):
        self.expect("say(m.dur_parse('2m'), m.dur_parse('2ms'))", "120 0.002")

    def test_parse_empty_text_fails(self):
        self.expect("say(try(m.dur_parse, ''))", '"err:duration must be a non-empty string"')

    def test_parse_number_without_unit_fails(self):
        self.expect("say(try(m.dur_parse, '30'))", '"err:malformed duration at position 1"')

    def test_parse_unit_without_number_fails(self):
        self.expect("say(try(m.dur_parse, 's'))", '"err:malformed duration at position 1"')

    def test_parse_reports_position_of_garbage(self):
        self.expect("say(try(m.dur_parse, '1m 30s'))", '"err:malformed duration at position 3"')

    def test_parse_rejects_negative_text(self):
        self.expect("say(try(m.dur_parse, '-5s'))", '"err:malformed duration at position 1"')

    def test_parse_rejects_unknown_unit(self):
        self.expect("say(try(m.dur_parse, '5x'))", '"err:unknown unit: x"')

    def test_try_parse_returns_value(self):
        self.expect("say(m.dur_try_parse('2m'))", "120")

    def test_try_parse_returns_nil_and_message(self):
        self.expect("say(m.dur_try_parse('oops'))", 'nil "malformed duration at position 1"')

    def test_valid_text_is_recognised(self):
        self.expect("say(m.dur_is_valid('3h'), m.dur_is_valid('3'))", "true false")

    def test_format_small_value_in_seconds(self):
        self.expect("say(m.dur_format(45))", '"45s"')

    def test_format_minutes_and_seconds(self):
        self.expect("say(m.dur_format(90))", '"1m30s"')

    def test_format_exactly_one_hour(self):
        self.expect("say(m.dur_format(3600))", '"1h"')

    def test_format_uses_every_unit_once(self):
        self.expect("say(m.dur_format(90061.001))", '"1d1h1m1s1ms"')

    def test_format_zero_is_zero_seconds(self):
        self.expect("say(m.dur_format(0))", '"0s"')

    def test_format_rounds_to_nearest_millisecond(self):
        self.expect("say(m.dur_format(0.0014), m.dur_format(0.0016))", '"1ms" "2ms"')

    def test_negative_seconds_cannot_be_formatted(self):
        self.expect("say(try(m.dur_format, -1))", '"err:seconds must be a non-negative number"')

    def test_formatting_a_string_fails(self):
        self.expect("say(try(m.dur_format, '5'))", '"err:seconds must be a non-negative number"')

    def test_shorter_left_duration_sorts_before(self):
        self.expect("say(m.dur_compare('59s', '1m'))", "-1")

    def test_longer_left_duration_sorts_after(self):
        self.expect("say(m.dur_compare('2h', '119m'))", "1")

    def test_compare_equal_spellings_gives_zero(self):
        self.expect("say(m.dur_compare('90s', '1m30s'))", "0")

    def test_sum_adds_parsed_values(self):
        self.expect("say(m.dur_sum({'1m', '30s', '500ms'}))", "90.5")

    def test_sum_stops_on_bad_entry(self):
        self.expect("say(try(m.dur_sum, {'1m', 'bad'}))", '"err:malformed duration at position 1"')
