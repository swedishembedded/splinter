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


class SplitTopLevelTests(unittest.TestCase):
    def test_single_piece_without_commas(self):
        self.assertEqual(ini_split_top_level("abc"), [("abc", 0)])

    def test_commas_split_with_offsets(self):
        self.assertEqual(ini_split_top_level("a,bb,c"), [("a", 0), ("bb", 2), ("c", 5)])

    def test_commas_in_quotes_do_not_split(self):
        self.assertEqual(ini_split_top_level('"a,b",c'), [('"a,b"', 0), ("c", 6)])

    def test_nested_brackets_hide_commas(self):
        self.assertEqual(ini_split_top_level("[1,2],3"), [("[1,2]", 0), ("3", 6)])

    def test_trailing_comma_leaves_empty_tail(self):
        self.assertEqual(ini_split_top_level("a,"), [("a", 0), ("", 2)])

    def test_empty_text_is_one_empty_piece(self):
        self.assertEqual(ini_split_top_level(""), [("", 0)])

    def test_deeper_nesting_is_balanced(self):
        self.assertEqual(ini_split_top_level("[[1],[2]],x"), [("[[1],[2]]", 0), ("x", 10)])


class DecodeArrayTests(unittest.TestCase):
    def test_empty_brackets_decode_to_an_empty_list(self):
        self.assertEqual(ini_decode_array("[]"), [])

    def test_blank_inside_is_still_empty(self):
        self.assertEqual(ini_decode_array("[   ]"), [])

    def test_integers_keep_their_written_order(self):
        self.assertEqual(ini_decode_array("[1, 2, 3]"), [1, 2, 3])

    def test_quoted_strings_may_contain_commas(self):
        self.assertEqual(ini_decode_array('["a,b", \'c\']'), ["a,b", "c"])

    def test_one_element_array_has_one_item(self):
        self.assertEqual(ini_decode_array("[7]"), [7])

    def test_trailing_comma_is_allowed(self):
        self.assertEqual(ini_decode_array("[1, 2,]"), [1, 2])

    def test_whitespace_around_elements_is_trimmed(self):
        self.assertEqual(ini_decode_array("[  true ,false  ]"), [True, False])

    def test_arrays_may_hold_other_arrays(self):
        self.assertEqual(ini_decode_array("[[1, 2], [3]]"), [[1, 2], [3]])

    def test_empty_nested_arrays_are_homogeneous(self):
        self.assertEqual(ini_decode_array("[[], [1]]"), [[], [1]])

    def test_float_elements_form_a_valid_array(self):
        self.assertEqual(ini_decode_array("[1.5, 2e0]"), [1.5, 2.0])

    def test_close_bracket_is_required_at_the_end(self):
        self.assertEqual(failure(ini_decode_array, "[1, 2", 4, 6), "line 4, column 6: unterminated array")

    def test_lone_open_bracket_is_unterminated(self):
        self.assertEqual(failure(ini_decode_array, "[", 1, 1), "line 1, column 1: unterminated array")

    def test_empty_element_between_commas(self):
        self.assertEqual(failure(ini_decode_array, "[1,,2]", 1, 10), "line 1, column 13: empty array element")

    def test_leading_comma_is_an_empty_element(self):
        self.assertEqual(failure(ini_decode_array, "[,1]", 1, 1), "line 1, column 2: empty array element")

    def test_lone_comma_is_an_empty_element(self):
        self.assertEqual(failure(ini_decode_array, "[,]", 1, 1), "line 1, column 2: empty array element")

    def test_double_trailing_comma_is_refused(self):
        self.assertEqual(failure(ini_decode_array, "[1,,]", 1, 1), "line 1, column 4: empty array element")

    def test_element_error_points_at_the_element(self):
        self.assertEqual(failure(ini_decode_array, "[1,  bad]", 2, 5),
                         "line 2, column 10: unrecognized value 'bad'")

    def test_first_element_error_column_is_exact(self):
        self.assertEqual(failure(ini_decode_array, "[bad]", 1, 3), "line 1, column 4: unrecognized value 'bad'")

    def test_mixed_integer_and_string(self):
        self.assertEqual(failure(ini_decode_array, '[1, "a"]', 1, 1),
                         "line 1, column 5: mixed types in array: string after integer")

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

    def test_nested_error_is_positioned_inside(self):
        self.assertEqual(failure(ini_decode_array, "[[1, x]]", 1, 1),
                         "line 1, column 6: unrecognized value 'x'")

    def test_extra_close_bracket_is_an_element_error(self):
        self.assertEqual(failure(ini_decode_array, "[1]]"), "unrecognized value '1]'")


class DecodeValueTests(unittest.TestCase):
    def test_bracket_text_becomes_an_array(self):
        self.assertEqual(ini_decode_value("[1]"), [1])

    def test_other_text_is_a_scalar(self):
        self.assertEqual(ini_decode_value("5"), 5)

    def test_quoted_bracket_text_is_a_string(self):
        self.assertEqual(ini_decode_value('"[1]"'), "[1]")

    def test_position_reaches_the_scalar_decoder(self):
        self.assertEqual(failure(ini_decode_value, "nope", 3, 8), "line 3, column 8: unrecognized value 'nope'")

    def test_array_failure_without_location_arguments(self):
        self.assertEqual(failure(ini_decode_value, "[1, x]"), "unrecognized value 'x'")

    def test_position_reaches_the_array_decoder(self):
        self.assertEqual(failure(ini_decode_value, "[1", 3, 8), "line 3, column 8: unterminated array")


if __name__ == "__main__":
    unittest.main()
