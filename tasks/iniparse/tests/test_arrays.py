# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from iniparse.arrays import ini_decode_array, ini_decode_value, ini_split_top_level
from iniparse.diagnostics import IniSyntaxError


def failure(call, *args):
    try:
        call(*args)
    except IniSyntaxError as error:
        return str(error)
    raise AssertionError("no error raised")


class C1(unittest.TestCase):

    def test_commas_split_with_offsets(self):
        self.assertEqual(ini_split_top_level("a,bb,c"), [("a", 0), ("bb", 2), ("c", 5)])


    def test_nested_brackets_hide_commas(self):
        self.assertEqual(ini_split_top_level("[1,2],3"), [("[1,2]", 0), ("3", 6)])


    def test_deeper_nesting_is_balanced(self):
        self.assertEqual(ini_split_top_level("[[1],[2]],x"), [("[[1],[2]]", 0), ("x", 10)])


class C2(unittest.TestCase):


    def test_quoted_strings_may_contain_commas(self):
        self.assertEqual(ini_decode_array('["a,b", \'c\']'), ["a,b", "c"])


    def test_trailing_comma_is_allowed(self):
        self.assertEqual(ini_decode_array("[1, 2,]"), [1, 2])

    def test_whitespace_around_elements_is_trimmed(self):
        self.assertEqual(ini_decode_array("[  true ,false  ]"), [True, False])

    def test_arrays_may_hold_other_arrays(self):
        self.assertEqual(ini_decode_array("[[1, 2], [3]]"), [[1, 2], [3]])


    def test_integer_and_float_do_not_mix(self):
        self.assertEqual(failure(ini_decode_array, "[1, 2.5]", 1, 1),
                         "line 1, column 5: mixed types in array: float after integer")

    def test_boolean_and_integer_do_not_mix(self):
        self.assertEqual(failure(ini_decode_array, "[true, 1]"),
                         "mixed types in array: integer after boolean")

    def test_mixed_error_names_the_first_offender(self):
        self.assertEqual(failure(ini_decode_array, '[1, "a", 2.5]', 1, 1),
                         "line 1, column 5: mixed types in array: string after integer")

    def test_array_and_scalar_do_not_mix(self):
        self.assertEqual(failure(ini_decode_array, "[[1], 2]"), "mixed types in array: integer after array")


    def test_extra_close_bracket_is_an_element_error(self):
        self.assertEqual(failure(ini_decode_array, "[1]]"), "unrecognized value '1]'")


class C3(unittest.TestCase):


    def test_array_failure_without_location_arguments(self):
        self.assertEqual(failure(ini_decode_value, "[1, x]"), "unrecognized value 'x'")


if __name__ == "__main__":
    unittest.main()
