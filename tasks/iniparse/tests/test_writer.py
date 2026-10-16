# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from iniparse.diagnostics import IniEncodeError
from iniparse.reader import ini_load_document
from iniparse.writer import (
    ini_dump_document, ini_dump_table, ini_encode_pair, ini_encode_scalar, ini_encode_value,
)


def refusal(call, *args):
    try:
        call(*args)
    except IniEncodeError as error:
        return str(error)
    raise AssertionError("no error raised")


class EncodeScalarTests(unittest.TestCase):
    def test_booleans_are_lower_case_words(self):
        self.assertEqual((ini_encode_scalar(True), ini_encode_scalar(False)), ("true", "false"))

    def test_whole_numbers_are_plain_digits(self):
        self.assertEqual((ini_encode_scalar(0), ini_encode_scalar(-12)), ("0", "-12"))

    def test_float_keeps_a_fraction(self):
        self.assertEqual(ini_encode_scalar(2.0), "2.0")

    def test_large_float_uses_exponent_form(self):
        self.assertEqual(ini_encode_scalar(1e22), "1e+22")

    def test_signed_zero_float_keeps_its_sign(self):
        self.assertEqual(ini_encode_scalar(-0.0), "-0.0")

    def test_string_is_double_quoted(self):
        self.assertEqual(ini_encode_scalar("hi"), '"hi"')

    def test_string_escapes_are_written(self):
        self.assertEqual(ini_encode_scalar('a"\n'), '"a\\"\\n"')

    def test_infinite_float_is_refused(self):
        self.assertEqual(refusal(ini_encode_scalar, float("inf")), "cannot encode non-finite float inf")

    def test_not_a_number_is_refused(self):
        self.assertEqual(refusal(ini_encode_scalar, float("nan")), "cannot encode non-finite float nan")

    def test_table_is_not_a_scalar(self):
        self.assertEqual(refusal(ini_encode_scalar, {}), "cannot encode a table as a scalar")

    def test_array_is_not_a_scalar(self):
        self.assertEqual(refusal(ini_encode_scalar, []), "cannot encode a array as a scalar")

    def test_unsupported_type_is_refused(self):
        self.assertEqual(refusal(ini_encode_scalar, None), "unsupported value of type NoneType")


class EncodeValueTests(unittest.TestCase):
    def test_scalar_item_is_encoded_directly(self):
        self.assertEqual(ini_encode_value(5), "5")

    def test_no_items_gives_empty_brackets(self):
        self.assertEqual(ini_encode_value([]), "[]")

    def test_array_items_are_comma_separated(self):
        self.assertEqual(ini_encode_value([1, 2, 3]), "[1, 2, 3]")

    def test_string_items_are_quoted(self):
        self.assertEqual(ini_encode_value(["a", "b"]), '["a", "b"]')

    def test_inner_arrays_stay_in_brackets(self):
        self.assertEqual(ini_encode_value([[1], [2, 3]]), "[[1], [2, 3]]")

    def test_mixed_array_is_refused(self):
        self.assertEqual(refusal(ini_encode_value, [1, "a"]), "cannot encode an array of mixed types")

    def test_integer_and_float_are_mixed(self):
        self.assertEqual(refusal(ini_encode_value, [1, 2.0]), "cannot encode an array of mixed types")

    def test_boolean_and_integer_are_mixed(self):
        self.assertEqual(refusal(ini_encode_value, [True, 1]), "cannot encode an array of mixed types")

    def test_table_in_array_is_refused(self):
        self.assertEqual(refusal(ini_encode_value, [{}]), "cannot encode a table as a scalar")

    def test_unsupported_item_is_refused(self):
        self.assertEqual(refusal(ini_encode_value, [None]), "unsupported value of type NoneType")


class EncodePairTests(unittest.TestCase):
    def test_short_pair_is_inline(self):
        self.assertEqual(ini_encode_pair("a", 1, 72), "a = 1")

    def test_odd_key_is_quoted(self):
        self.assertEqual(ini_encode_pair("a b", 1, 72), '"a b" = 1')

    def test_array_fitting_the_width_is_inline(self):
        self.assertEqual(ini_encode_pair("a", [1, 2], 10), "a = [1, 2]")

    def test_array_over_the_width_is_split(self):
        self.assertEqual(ini_encode_pair("a", [1, 2], 9), "a = [\n    1,\n    2,\n]")

    def test_line_exactly_at_width_stays_inline(self):
        self.assertEqual(len(ini_encode_pair("a", [1, 2], 10)), 10)

    def test_long_string_stays_inline(self):
        self.assertEqual(ini_encode_pair("k", "x" * 20, 5), 'k = "' + "x" * 20 + '"')

    def test_long_scalar_stays_inline(self):
        self.assertEqual(ini_encode_pair("key", 123456789, 3), "key = 123456789")

    def test_empty_array_stays_inline(self):
        self.assertEqual(ini_encode_pair("a_long_key", [], 3), "a_long_key = []")

    def test_split_items_are_encoded(self):
        self.assertEqual(ini_encode_pair("n", ["a", "b"], 3), 'n = [\n    "a",\n    "b",\n]')

    def test_split_nested_items_stay_inline(self):
        self.assertEqual(ini_encode_pair("n", [[1, 2]], 3), "n = [\n    [1, 2],\n]")

    def test_split_array_reads_back(self):
        text = ini_encode_pair("n", list(range(40)), 20)
        self.assertEqual(ini_load_document(text), {"n": list(range(40))})


class DumpDocumentTests(unittest.TestCase):
    def test_empty_document_is_empty_text(self):
        self.assertEqual(ini_dump_document({}), "")

    def test_root_pairs_end_with_a_newline(self):
        self.assertEqual(ini_dump_document({"a": 1, "b": "x"}), 'a = 1\nb = "x"\n')

    def test_table_gets_a_header(self):
        self.assertEqual(ini_dump_document({"s": {"a": 1}}), "[s]\na = 1\n")

    def test_root_pairs_come_before_tables(self):
        self.assertEqual(ini_dump_document({"s": {"a": 1}, "r": 2}), "r = 2\n\n[s]\na = 1\n")

    def test_blocks_are_separated_by_a_blank_line(self):
        self.assertEqual(ini_dump_document({"s": {"a": 1}, "t": {"b": 2}}), "[s]\na = 1\n\n[t]\nb = 2\n")

    def test_nested_table_uses_a_dotted_header(self):
        self.assertEqual(ini_dump_document({"s": {"a": 1, "t": {"b": 2}}}), "[s]\na = 1\n\n[s.t]\nb = 2\n")

    def test_table_holding_only_tables_has_no_header(self):
        self.assertEqual(ini_dump_document({"s": {"t": {"b": 2}}}), "[s.t]\nb = 2\n")

    def test_empty_table_keeps_its_header(self):
        self.assertEqual(ini_dump_document({"s": {}}), "[s]\n")

    def test_empty_nested_table_keeps_its_header(self):
        self.assertEqual(ini_dump_document({"s": {"t": {}}}), "[s.t]\n")

    def test_odd_table_names_are_quoted(self):
        self.assertEqual(ini_dump_document({"a b": {"c.d": 1}}), '["a b"]\n"c.d" = 1\n')

    def test_width_option_splits_long_arrays(self):
        self.assertEqual(ini_dump_document({"a": [1, 2]}, width=5), "a = [\n    1,\n    2,\n]\n")

    def test_default_width_is_seventy_two(self):
        short = {"a": ["x" * 64]}
        long = {"a": ["x" * 65]}
        self.assertEqual(len(ini_dump_document(short).splitlines()), 1)
        self.assertEqual(len(ini_dump_document(long).splitlines()), 3)

    def test_non_string_root_key_is_refused(self):
        self.assertEqual(refusal(ini_dump_document, {1: 2}), "table keys must be strings, not int")

    def test_non_string_table_key_is_refused(self):
        self.assertEqual(refusal(ini_dump_document, {"s": {2: 2}}), "table keys must be strings, not int")

    def test_empty_child_table_rejects_non_string_key(self):
        self.assertEqual(refusal(ini_dump_document, {"s": {3: {}}}), "table keys must be strings, not int")

    def test_dump_table_appends_to_the_given_blocks(self):
        blocks = ["first"]
        ini_dump_table({"a": 1}, ("t",), 72, blocks)
        self.assertEqual(blocks, ["first", "[t]\na = 1"])


class RoundTripTests(unittest.TestCase):
    def assert_round_trip(self, document):
        again = ini_load_document(ini_dump_document(document))
        self.assertEqual(repr(again), repr(document))

    def test_round_trip_of_scalars(self):
        self.assert_round_trip({"a": 1, "b": 2.5, "c": True, "d": "x y", "e": False})

    def test_round_trip_of_awkward_strings(self):
        self.assert_round_trip({"a": 'q"\\\n\t\x01 # not a comment ; ok', "b": "", "c": "caf\u00e9 [x] = y"})

    def test_round_trip_of_tables(self):
        self.assert_round_trip({"a": 1, "s": {"b": 2, "t": {"c": 3}}, "u": {"v": {}}})

    def test_round_trip_of_arrays(self):
        self.assert_round_trip({"a": [1, 2], "b": [], "c": [[1], [2, 3]], "d": ["x,y", "]"], "e": [True, False]})

    def test_round_trip_of_odd_keys(self):
        self.assert_round_trip({"a b": {"c.d": 1, "": 2, "e\n": 3}, "x": {"y z": {"w": 4}}})

    def test_round_trip_of_long_arrays(self):
        self.assert_round_trip({"s": {"a": list(range(50)), "b": ["word"] * 30}})

    def test_round_trip_of_extreme_floats(self):
        self.assert_round_trip({"a": 1e22, "b": 1.5e-07, "c": -0.0, "d": 123456789.125})


if __name__ == "__main__":
    unittest.main()
