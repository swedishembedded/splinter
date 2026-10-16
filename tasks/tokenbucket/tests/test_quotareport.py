# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
from tests.luahost import LuaCase

ROWS = ("local rows = m.qrep_build({{name='web', limit=100, used=50}, "
        "{name='api', limit=200, used=190}, {name='batch', limit=10, used=10}}); ")


class QuotaReportTests(LuaCase):
    MODULE = "quotareport"

    def test_percent_of_half_use(self):
        self.expect("say(m.qrep_percent(50, 100))", "50")

    def test_percent_is_rounded_down(self):
        self.expect("say(m.qrep_percent(2, 3))", "66")

    def test_percent_of_untouched_quota(self):
        self.expect("say(m.qrep_percent(0, 10))", "0")

    def test_percent_may_exceed_hundred(self):
        self.expect("say(m.qrep_percent(15, 10))", "150")

    def test_percent_needs_positive_limit(self):
        self.expect("say(try(m.qrep_percent, 1, 0))", '"err:limit must be a positive number"')

    def test_percent_needs_non_negative_usage(self):
        self.expect("say(try(m.qrep_percent, -1, 10))",
                    '"err:used must be a non-negative number"')

    def test_percent_rejects_text_usage(self):
        self.expect("say(try(m.qrep_percent, 'a', 10))",
                    '"err:used must be a non-negative number"')

    def test_status_ok_below_warning_level(self):
        self.expect("say(m.qrep_status(79, 100))", '"ok"')

    def test_status_warning_at_eighty_percent(self):
        self.expect("say(m.qrep_status(80, 100))", '"warning"')

    def test_status_warning_just_below_limit(self):
        self.expect("say(m.qrep_status(99, 100))", '"warning"')

    def test_status_exhausted_at_limit(self):
        self.expect("say(m.qrep_status(100, 100))", '"exhausted"')

    def test_status_exhausted_beyond_limit(self):
        self.expect("say(m.qrep_status(130, 100))", '"exhausted"')

    def test_status_for_empty_usage(self):
        self.expect("say(m.qrep_status(0, 100))", '"ok"')

    def test_row_carries_its_figures(self):
        self.expect("say(m.qrep_row('web', 100, 85))",
                    '{limit=100,name="web",percent=85,remaining=15,status="warning",used=85}')

    def test_row_remaining_never_negative(self):
        self.expect("say(m.qrep_row('web', 100, 120).remaining)", "0")

    def test_row_of_unused_quota(self):
        self.expect("say(m.qrep_row('web', 40, 0))",
                    '{limit=40,name="web",percent=0,remaining=40,status="ok",used=0}')

    def test_build_sorts_rows_by_name(self):
        self.expect(ROWS + "say(rows[1].name, rows[2].name, rows[3].name)",
                    '"api" "batch" "web"')

    def test_build_computes_every_row(self):
        self.expect(ROWS + "say(rows[1].percent, rows[2].status, rows[3].remaining)",
                    '95 "exhausted" 50')

    def test_build_of_nothing_is_empty(self):
        self.expect("say(m.qrep_build({}))", "{}")

    def test_build_refuses_duplicate_names(self):
        self.expect("say(try(m.qrep_build, {{name='a', limit=1, used=0}, "
                    "{name='a', limit=2, used=1}}))", '"err:duplicate name: a"')

    def test_totals_sum_limits_and_usage(self):
        self.expect(ROWS + "local t = m.qrep_totals(rows); say(t.limit, t.used, t.remaining)",
                    "310 250 60")

    def test_totals_percent_of_combined_quota(self):
        self.expect(ROWS + "say(m.qrep_totals(rows).percent)", "80")

    def test_totals_remaining_ignores_overuse(self):
        self.expect("local t = m.qrep_totals({m.qrep_row('a', 10, 15)}); say(t.remaining)", "0")

    def test_totals_of_no_rows_are_zero(self):
        self.expect("say(m.qrep_totals({}))", "{limit=0,percent=0,remaining=0,used=0}")

    def test_busiest_picks_highest_percent(self):
        self.expect(ROWS + "say(m.qrep_busiest(rows).name)", '"batch"')

    def test_busiest_prefers_earlier_name_on_tie(self):
        self.expect("local rows = {m.qrep_row('b', 10, 5), m.qrep_row('a', 20, 10)}; "
                    "say(m.qrep_busiest(rows).name)", '"a"')

    def test_busiest_tie_independent_of_order(self):
        self.expect("local rows = {m.qrep_row('a', 20, 10), m.qrep_row('b', 10, 5)}; "
                    "say(m.qrep_busiest(rows).name)", '"a"')

    def test_busiest_ignores_lower_usage_later(self):
        self.expect("local rows = {m.qrep_row('a', 10, 9), m.qrep_row('b', 10, 2)}; "
                    "say(m.qrep_busiest(rows).name)", '"a"')

    def test_busiest_of_no_rows_is_nil(self):
        self.expect("say(m.qrep_busiest({}))", "nil")

    def test_filter_keeps_matching_status(self):
        self.expect(ROWS + "local kept = m.qrep_filter(rows, 'exhausted'); "
                    "say(#kept, kept[1].name)", '1 "batch"')

    def test_filter_keeps_order_of_rows(self):
        self.expect("local rows = {m.qrep_row('a', 10, 1), m.qrep_row('b', 10, 2), "
                    "m.qrep_row('c', 10, 10)}; local kept = m.qrep_filter(rows, 'ok'); "
                    "say(kept[1].name, kept[2].name, #kept)", '"a" "b" 2')

    def test_filter_without_match_is_empty(self):
        self.expect(ROWS + "say(m.qrep_filter(rows, 'unknown'))", "{}")

    def test_render_formats_one_line_per_row(self):
        self.expect(ROWS + "print(m.qrep_render(rows))",
                    "api   190/200  95% warning\n"
                    "batch 10/10 100% exhausted\n"
                    "web   50/100  50% ok")

    def test_render_pads_percent_to_three_columns(self):
        self.expect("print(m.qrep_render({m.qrep_row('x', 1000, 5)}))", "x 5/1000   0% ok")

    def test_render_of_nothing_says_so(self):
        self.expect("print(m.qrep_render({}))", "(no quotas)")
