# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

import unittest

from linediff.csvio import (
    LinediffCsvError,
    linediff_csv_pad_rows,
    linediff_csv_quote_field,
    linediff_csv_ragged_rows,
    linediff_csv_read,
    linediff_csv_records,
    linediff_csv_write,
)

DELIMITER_MESSAGE = "delimiter must be one character other than a quote or line break"


def csv_error(call, *args):
    with unittest.TestCase().assertRaises(LinediffCsvError) as caught:
        call(*args)
    return str(caught.exception)


class CsvReadTests(unittest.TestCase):
    def test_empty_text_has_no_rows(self):
        self.assertEqual(linediff_csv_read(""), [])

    def test_single_field_is_one_row(self):
        self.assertEqual(linediff_csv_read("a"), [["a"]])

    def test_lines_become_rows_of_fields(self):
        self.assertEqual(linediff_csv_read("a,b\nc,d\n"), [["a", "b"], ["c", "d"]])

    def test_missing_final_newline_makes_no_difference(self):
        self.assertEqual(linediff_csv_read("a,b\nc,d"), linediff_csv_read("a,b\nc,d\n"))

    def test_carriage_return_pairs_end_rows(self):
        self.assertEqual(linediff_csv_read("a\r\nb\r\n"), [["a"], ["b"]])

    def test_bare_carriage_returns_end_rows(self):
        self.assertEqual(linediff_csv_read("a\rb\r"), [["a"], ["b"]])

    def test_quoted_field_may_hold_delimiter(self):
        self.assertEqual(linediff_csv_read('"a,b",c'), [["a,b", "c"]])

    def test_quoted_field_may_hold_newline(self):
        self.assertEqual(linediff_csv_read('"a\nb",c\nd,e\n'), [["a\nb", "c"], ["d", "e"]])

    def test_quoted_carriage_return_pair_is_kept(self):
        self.assertEqual(linediff_csv_read('"a\r\nb"'), [["a\r\nb"]])

    def test_doubled_quotes_become_one_quote(self):
        self.assertEqual(linediff_csv_read('"say ""hi"""'), [['say "hi"']])

    def test_empty_quoted_field_is_empty_string(self):
        self.assertEqual(linediff_csv_read('"",x'), [["", "x"]])

    def test_adjacent_delimiters_make_empty_fields(self):
        self.assertEqual(linediff_csv_read("a,,c"), [["a", "", "c"]])

    def test_trailing_delimiter_adds_empty_field(self):
        self.assertEqual(linediff_csv_read("a,b,"), [["a", "b", ""]])

    def test_trailing_delimiter_before_newline_adds_field(self):
        self.assertEqual(linediff_csv_read("a,\nb\n"), [["a", ""], ["b"]])

    def test_leading_delimiter_starts_with_empty_field(self):
        self.assertEqual(linediff_csv_read(",a"), [["", "a"]])

    def test_blank_line_reads_as_one_empty_field(self):
        self.assertEqual(linediff_csv_read("a\n\nb\n"), [["a"], [""], ["b"]])

    def test_ragged_rows_keep_their_own_width(self):
        self.assertEqual(linediff_csv_read("a,b,c\nd\n"), [["a", "b", "c"], ["d"]])

    def test_custom_delimiter_splits_fields(self):
        self.assertEqual(linediff_csv_read("a;b,c\td", ";"), [["a", "b,c\td"]])

    def test_tab_delimiter_with_quoted_tab(self):
        self.assertEqual(linediff_csv_read('"a\tb"\tc', "\t"), [["a\tb", "c"]])

    def test_quote_inside_plain_field_is_literal(self):
        self.assertEqual(linediff_csv_read('ab"c,d'), [['ab"c', "d"]])

    def test_spaces_around_fields_are_kept(self):
        self.assertEqual(linediff_csv_read(" a , b "), [[" a ", " b "]])

    def test_quoted_fields_before_crlf_end_rows(self):
        self.assertEqual(linediff_csv_read('"a"\r\n"b"\r\n'), [["a"], ["b"]])

    def test_unterminated_quote_reports_start_line(self):
        self.assertEqual(
            csv_error(linediff_csv_read, 'a\n"b'), "line 2: unterminated quoted field"
        )

    def test_text_after_closing_quote_is_rejected(self):
        self.assertEqual(
            csv_error(linediff_csv_read, '"a"b'),
            "line 1: unexpected character after closing quote",
        )

    def test_error_line_counts_newlines_inside_quotes(self):
        self.assertEqual(
            csv_error(linediff_csv_read, '"x\ny"z'),
            "line 2: unexpected character after closing quote",
        )

    def test_error_line_counts_earlier_rows(self):
        self.assertEqual(
            csv_error(linediff_csv_read, 'a\n"b"c'),
            "line 2: unexpected character after closing quote",
        )

    def test_unsuitable_delimiters_are_refused(self):
        for bad in ("", ",,", '"', "\n", "\r"):
            with self.assertRaises(ValueError) as caught:
                linediff_csv_read("a", bad)
            self.assertEqual(str(caught.exception), DELIMITER_MESSAGE)


class CsvQuoteFieldTests(unittest.TestCase):
    def test_plain_text_is_left_alone(self):
        self.assertEqual(linediff_csv_quote_field("a b"), "a b")

    def test_delimiter_inside_value_forces_quotes(self):
        self.assertEqual(linediff_csv_quote_field("a,b"), '"a,b"')

    def test_embedded_quotes_are_doubled_inside(self):
        self.assertEqual(linediff_csv_quote_field('a"b'), '"a""b"')

    def test_line_breaks_force_quotes(self):
        self.assertEqual(linediff_csv_quote_field("a\nb"), '"a\nb"')
        self.assertEqual(linediff_csv_quote_field("a\rb"), '"a\rb"')

    def test_other_delimiter_changes_what_is_quoted(self):
        self.assertEqual(linediff_csv_quote_field("a,b;c", ";"), '"a,b;c"')
        self.assertEqual(linediff_csv_quote_field("a,b", ";"), "a,b")

    def test_unsuitable_delimiter_is_refused_when_quoting(self):
        with self.assertRaises(ValueError) as caught:
            linediff_csv_quote_field("a", "")
        self.assertEqual(str(caught.exception), DELIMITER_MESSAGE)

    def test_empty_value_stays_empty(self):
        self.assertEqual(linediff_csv_quote_field(""), "")


class CsvWriteTests(unittest.TestCase):
    def test_rows_end_with_newlines(self):
        self.assertEqual(linediff_csv_write([["a", "b"], ["c", "d"]]), "a,b\nc,d\n")

    def test_no_rows_give_empty_text(self):
        self.assertEqual(linediff_csv_write([]), "")

    def test_fields_are_quoted_when_needed(self):
        self.assertEqual(linediff_csv_write([["a,b", 'c"d']]), '"a,b","c""d"\n')

    def test_newline_and_delimiter_are_configurable(self):
        self.assertEqual(linediff_csv_write([["a", "b"]], "|", "\r\n"), "a|b\r\n")

    def test_row_without_fields_is_rejected(self):
        self.assertEqual(
            csv_error(linediff_csv_write, [["a"], []]), "row 2 has no fields"
        )

    def test_written_text_reads_back_unchanged(self):
        rows = [["a", 'b"c', "d\ne"], ["", ",", "f"], [""]]
        self.assertEqual(linediff_csv_read(linediff_csv_write(rows)), rows)


class CsvShapeTests(unittest.TestCase):
    def test_uniform_rows_are_not_ragged(self):
        self.assertEqual(linediff_csv_ragged_rows([["a", "b"], ["c", "d"]]), [])

    def test_rows_of_other_width_are_listed(self):
        rows = [["a", "b"], ["c"], ["d", "e"], ["f", "g", "h"]]
        self.assertEqual(linediff_csv_ragged_rows(rows), [1, 3])

    def test_no_rows_have_no_ragged_rows(self):
        self.assertEqual(linediff_csv_ragged_rows([]), [])

    def test_padding_widens_short_rows(self):
        self.assertEqual(
            linediff_csv_pad_rows([["a"], ["b", "c", "d"]]),
            [["a", "", ""], ["b", "c", "d"]],
        )

    def test_padding_uses_given_fill(self):
        self.assertEqual(linediff_csv_pad_rows([["a"], ["b", "c"]], "-"), [["a", "-"], ["b", "c"]])

    def test_padding_leaves_empty_rows_empty(self):
        self.assertEqual(linediff_csv_pad_rows([[]]), [[]])

    def test_padding_nothing_gives_nothing(self):
        self.assertEqual(linediff_csv_pad_rows([]), [])


class CsvRecordsTests(unittest.TestCase):
    def test_rows_become_dictionaries_by_header(self):
        self.assertEqual(
            linediff_csv_records([["id", "name"], ["1", "ann"], ["2", "bob"]]),
            (["id", "name"], [{"id": "1", "name": "ann"}, {"id": "2", "name": "bob"}]),
        )

    def test_short_rows_are_padded_with_empty_values(self):
        self.assertEqual(
            linediff_csv_records([["id", "name"], ["1"]])[1], [{"id": "1", "name": ""}]
        )

    def test_header_only_gives_no_records(self):
        self.assertEqual(linediff_csv_records([["id"]]), (["id"], []))

    def test_row_wider_than_header_is_rejected(self):
        self.assertEqual(
            csv_error(linediff_csv_records, [["a", "b"], ["1", "2"], ["1", "2", "3"]]),
            "row 3 has 3 fields but the header has 2",
        )

    def test_missing_header_is_rejected(self):
        self.assertEqual(csv_error(linediff_csv_records, []), "no header row")

    def test_repeated_column_name_is_rejected(self):
        self.assertEqual(
            csv_error(linediff_csv_records, [["a", "b", "a"]]),
            "duplicate column 'a' in header",
        )
