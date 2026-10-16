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


class C1(unittest.TestCase):
    def test_booleans_are_lower_case_words(self):
        self.assertEqual((ini_encode_scalar(True), ini_encode_scalar(False)), ("true", "false"))


    def test_float_keeps_a_fraction(self):
        self.assertEqual(ini_encode_scalar(2.0), "2.0")

    def test_large_float_uses_exponent_form(self):
        self.assertEqual(ini_encode_scalar(1e22), "1e+22")


    def test_not_a_number_is_refused(self):
        self.assertEqual(refusal(ini_encode_scalar, float("nan")), "cannot encode non-finite float nan")

    def test_table_is_not_a_scalar(self):
        self.assertEqual(refusal(ini_encode_scalar, {}), "cannot encode a table as a scalar")

    def test_array_is_not_a_scalar(self):
        self.assertEqual(refusal(ini_encode_scalar, []), "cannot encode a array as a scalar")

    def test_unsupported_type_is_refused(self):
        self.assertEqual(refusal(ini_encode_scalar, None), "unsupported value of type NoneType")


class C2(unittest.TestCase):


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


class C3(unittest.TestCase):


    def test_array_fitting_the_width_is_inline(self):
        self.assertEqual(ini_encode_pair("a", [1, 2], 10), "a = [1, 2]")


    def test_line_exactly_at_width_stays_inline(self):
        self.assertEqual(len(ini_encode_pair("a", [1, 2], 10)), 10)


    def test_empty_array_stays_inline(self):
        self.assertEqual(ini_encode_pair("a_long_key", [], 3), "a_long_key = []")

    def test_split_items_are_encoded(self):
        self.assertEqual(ini_encode_pair("n", ["a", "b"], 3), 'n = [\n    "a",\n    "b",\n]')


    def test_split_array_reads_back(self):
        text = ini_encode_pair("n", list(range(40)), 20)
        self.assertEqual(ini_load_document(text), {"n": list(range(40))})


class C4(unittest.TestCase):
    def test_empty_document_is_empty_text(self):
        self.assertEqual(ini_dump_document({}), "")


    def test_table_holding_only_tables_has_no_header(self):
        self.assertEqual(ini_dump_document({"s": {"t": {"b": 2}}}), "[s.t]\nb = 2\n")


    def test_empty_nested_table_keeps_its_header(self):
        self.assertEqual(ini_dump_document({"s": {"t": {}}}), "[s.t]\n")


    def test_default_width_is_seventy_two(self):
        short = {"a": ["x" * 64]}
        long = {"a": ["x" * 65]}
        self.assertEqual(len(ini_dump_document(short).splitlines()), 1)
        self.assertEqual(len(ini_dump_document(long).splitlines()), 3)


class C5(unittest.TestCase):
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
