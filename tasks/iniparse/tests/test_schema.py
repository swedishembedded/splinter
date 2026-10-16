# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import dataclasses
import unittest

from iniparse.diagnostics import IniSchemaError
from iniparse.schema import (
    IniField, ini_apply_schema, ini_check_field, ini_coerce_array, ini_coerce_boolean, ini_coerce_field,
    ini_coerce_float, ini_coerce_integer, ini_coerce_string, ini_coerce_value, ini_required_paths,
    ini_schema_defaults,
)


def problems_of(call, *args, **kwargs):
    try:
        call(*args, **kwargs)
    except IniSchemaError as error:
        return list(error.problems)
    raise AssertionError("no error raised")


class CoerceStringTests(unittest.TestCase):
    def test_integer_turns_into_digit_text(self):
        self.assertEqual(ini_coerce_string(80), "80")

    def test_float_turns_into_plain_text(self):
        self.assertEqual(ini_coerce_string(0.5), "0.5")

    def test_booleans_turn_into_words(self):
        self.assertEqual((ini_coerce_string(True), ini_coerce_string(False)), ("true", "false"))

    def test_array_cannot_become_text(self):
        self.assertIsNone(ini_coerce_string([1]))


class CoerceIntegerTests(unittest.TestCase):
    def test_whole_float_becomes_integer(self):
        self.assertEqual(ini_coerce_integer(3.0), 3)

    def test_fractional_float_is_refused(self):
        self.assertIsNone(ini_coerce_integer(3.5))

    def test_digits_with_padding_become_integer(self):
        self.assertEqual(ini_coerce_integer(" 42 "), 42)

    def test_zero_x_text_becomes_integer(self):
        self.assertEqual(ini_coerce_integer("0x10"), 16)

    def test_float_text_is_refused(self):
        self.assertIsNone(ini_coerce_integer("1.5"))

    def test_non_numeric_text_cannot_be_integer(self):
        self.assertIsNone(ini_coerce_integer("many"))

    def test_boolean_cannot_be_integer(self):
        self.assertIsNone(ini_coerce_integer(True))


class CoerceFloatTests(unittest.TestCase):
    def test_integer_widens_to_float(self):
        result = ini_coerce_float(2)
        self.assertEqual((result, type(result)), (2.0, float))

    def test_digits_with_padding_become_float(self):
        self.assertEqual(ini_coerce_float("0.25"), 0.25)

    def test_integer_text_becomes_float(self):
        result = ini_coerce_float("7")
        self.assertEqual((result, type(result)), (7.0, float))

    def test_non_numeric_text_cannot_be_float(self):
        self.assertIsNone(ini_coerce_float("x"))

    def test_boolean_cannot_be_float(self):
        self.assertIsNone(ini_coerce_float(False))


class CoerceBooleanTests(unittest.TestCase):
    def test_affirmative_words_mean_true(self):
        self.assertEqual([ini_coerce_boolean(w) for w in ("true", "yes", "on", "1")], [True] * 4)

    def test_negative_words_mean_false(self):
        self.assertEqual([ini_coerce_boolean(w) for w in ("false", "no", "off", "0")], [False] * 4)

    def test_words_ignore_case_and_padding(self):
        self.assertEqual((ini_coerce_boolean(" YES "), ini_coerce_boolean("Off")), (True, False))

    def test_zero_and_one_integers(self):
        self.assertEqual((ini_coerce_boolean(1), ini_coerce_boolean(0)), (True, False))

    def test_other_integers_are_refused(self):
        self.assertIsNone(ini_coerce_boolean(2))
        self.assertIsNone(ini_coerce_boolean(-1))

    def test_other_words_are_refused(self):
        self.assertIsNone(ini_coerce_boolean("maybe"))


class CoerceArrayTests(unittest.TestCase):
    def test_comma_text_is_split_and_trimmed(self):
        self.assertEqual(ini_coerce_array("a, b ,c"), ["a", "b", "c"])

    def test_one_word_makes_one_element(self):
        self.assertEqual(ini_coerce_array("solo"), ["solo"])

    def test_blank_text_is_empty(self):
        self.assertEqual(ini_coerce_array("   "), [])

    def test_number_cannot_be_array(self):
        self.assertIsNone(ini_coerce_array(5))


class CoerceValueTests(unittest.TestCase):
    def test_matching_kind_is_unchanged(self):
        self.assertEqual(ini_coerce_value(5, "integer"), 5)

    def test_text_is_converted_to_wanted_kind(self):
        self.assertEqual(ini_coerce_value("5", "integer"), 5)

    def test_unreadable_text_names_the_text(self):
        self.assertEqual(problems_of(ini_coerce_value, "five", "integer"), ["cannot read 'five' as integer"])

    def test_wrong_type_names_both_kinds(self):
        self.assertEqual(problems_of(ini_coerce_value, [1], "integer"), ["expected integer, got array"])

    def test_false_result_is_not_a_failure(self):
        self.assertIs(ini_coerce_value("no", "boolean"), False)

    def test_zero_result_is_not_a_failure(self):
        self.assertEqual(ini_coerce_value("0", "integer"), 0)

    def test_table_cannot_become_text(self):
        self.assertEqual(problems_of(ini_coerce_value, {}, "string"), ["expected string, got table"])


class FieldDefinitionTests(unittest.TestCase):
    def test_new_field_has_neutral_settings(self):
        field = IniField("string")
        self.assertEqual((field.required, field.default, field.minimum, field.maximum), (False, None, None, None))
        self.assertEqual((field.choices, field.item_kind, field.max_items), (None, None, None))

    def test_field_cannot_be_modified_afterwards(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            IniField("string").required = True

    def test_unknown_kind_is_refused(self):
        self.assertEqual(problems_of(IniField, "date"), ["unknown kind 'date'"])

    def test_unknown_item_kind_is_refused(self):
        self.assertEqual(problems_of(IniField, "array", item_kind="blob"), ["unknown kind 'blob'"])

    def test_table_is_not_a_field_kind(self):
        self.assertEqual(problems_of(IniField, "table"), ["unknown kind 'table'"])


class CheckFieldTests(unittest.TestCase):
    def test_choice_inside_the_set(self):
        ini_check_field("b", IniField("string", choices=("a", "b")))

    def test_choice_outside_the_set(self):
        self.assertEqual(problems_of(ini_check_field, "c", IniField("string", choices=("a", "b"))),
                         ["must be one of a, b"])

    def test_choices_may_be_numbers(self):
        self.assertEqual(problems_of(ini_check_field, 3, IniField("integer", choices=(1, 2))), ["must be one of 1, 2"])

    def test_value_equal_to_minimum_passes(self):
        ini_check_field(5, IniField("integer", minimum=5))

    def test_value_under_minimum_fails(self):
        self.assertEqual(problems_of(ini_check_field, 4, IniField("integer", minimum=5)), ["must be at least 5"])

    def test_value_equal_to_maximum_passes(self):
        ini_check_field(9, IniField("integer", maximum=9))

    def test_value_over_maximum_fails(self):
        self.assertEqual(problems_of(ini_check_field, 10, IniField("integer", maximum=9)), ["must be at most 9"])

    def test_fractional_bounds_are_compared(self):
        self.assertEqual(problems_of(ini_check_field, 0.5, IniField("float", minimum=0.75)), ["must be at least 0.75"])

    def test_zero_bound_is_honoured(self):
        self.assertEqual(problems_of(ini_check_field, -1, IniField("integer", minimum=0)), ["must be at least 0"])

    def test_bounds_ignore_other_kinds(self):
        ini_check_field("zzz", IniField("string", minimum=100, maximum=1))

    def test_item_count_limit_is_inclusive(self):
        ini_check_field([1, 2], IniField("array", max_items=2))

    def test_oversized_array_fails_the_limit(self):
        self.assertEqual(problems_of(ini_check_field, [1, 2, 3], IniField("array", max_items=2)),
                         ["must have at most 2 items"])

    def test_zero_item_limit_is_honoured(self):
        self.assertEqual(problems_of(ini_check_field, [1], IniField("array", max_items=0)),
                         ["must have at most 0 items"])


class CoerceFieldTests(unittest.TestCase):
    def test_converted_value_meets_its_bounds(self):
        self.assertEqual(ini_coerce_field("8080", IniField("integer", minimum=1, maximum=65535)), 8080)

    def test_unreadable_text_fails_the_field(self):
        self.assertEqual(problems_of(ini_coerce_field, "x", IniField("integer")), ["cannot read 'x' as integer"])

    def test_bounds_are_checked_after_conversion(self):
        self.assertEqual(problems_of(ini_coerce_field, "70000", IniField("integer", maximum=65535)),
                         ["must be at most 65535"])

    def test_array_items_are_converted(self):
        self.assertEqual(ini_coerce_field(["1", "2"], IniField("array", item_kind="integer")), [1, 2])

    def test_text_becomes_converted_array(self):
        self.assertEqual(ini_coerce_field("1, 2, 3", IniField("array", item_kind="integer")), [1, 2, 3])

    def test_bad_item_names_its_position(self):
        self.assertEqual(problems_of(ini_coerce_field, ["1", "x"], IniField("array", item_kind="integer")),
                         ["item 1: cannot read 'x' as integer"])

    def test_first_item_position_is_zero(self):
        self.assertEqual(problems_of(ini_coerce_field, [[1]], IniField("array", item_kind="integer")),
                         ["item 0: expected integer, got array"])

    def test_array_without_item_kind_is_unchanged(self):
        self.assertEqual(ini_coerce_field([1, "a"], IniField("array")), [1, "a"])

    def test_item_limit_applies_after_conversion(self):
        self.assertEqual(problems_of(ini_coerce_field, "a,b,c", IniField("array", max_items=2)),
                         ["must have at most 2 items"])


SCHEMA = {
    "name": IniField("string", required=True),
    "server": {
        "port": IniField("integer", default=8080, minimum=1, maximum=65535),
        "debug": IniField("boolean", default=False),
        "tags": IniField("array", item_kind="string"),
    },
}


class ApplySchemaTests(unittest.TestCase):
    def test_defaults_fill_missing_keys(self):
        self.assertEqual(ini_apply_schema({"name": "x"}, SCHEMA), {"name": "x", "server": {"port": 8080, "debug": False}})

    def test_present_values_are_coerced(self):
        document = {"name": "x", "server": {"port": "81", "debug": "yes", "tags": "a, b"}}
        self.assertEqual(ini_apply_schema(document, SCHEMA),
                         {"name": "x", "server": {"port": 81, "debug": True, "tags": ["a", "b"]}})

    def test_false_default_is_applied(self):
        self.assertIs(ini_apply_schema({"name": "x"}, SCHEMA)["server"]["debug"], False)

    def test_zero_default_is_applied(self):
        self.assertEqual(ini_apply_schema({}, {"n": IniField("integer", default=0)}), {"n": 0})

    def test_missing_required_key_is_reported(self):
        self.assertEqual(problems_of(ini_apply_schema, {}, SCHEMA), ["name: missing required key"])

    def test_optional_key_without_default_is_omitted(self):
        self.assertEqual(ini_apply_schema({}, {"a": IniField("string")}), {})

    def test_all_problems_are_collected(self):
        document = {"server": {"port": "x", "debug": "maybe"}}
        self.assertEqual(problems_of(ini_apply_schema, document, SCHEMA), [
            "name: missing required key",
            "server.port: cannot read 'x' as integer",
            "server.debug: cannot read 'maybe' as boolean",
        ])

    def test_section_that_is_not_a_table(self):
        self.assertEqual(problems_of(ini_apply_schema, {"name": "x", "server": 5}, SCHEMA),
                         ["server: expected table, got integer"])

    def test_missing_section_still_gets_defaults(self):
        result = ini_apply_schema({"name": "x"}, {"name": IniField("string"), "s": {"a": IniField("integer", default=1)}})
        self.assertEqual(result["s"], {"a": 1})

    def test_missing_section_without_defaults_is_empty(self):
        self.assertEqual(ini_apply_schema({}, {"s": {"a": IniField("integer")}}), {"s": {}})

    def test_nested_problem_path_is_dotted(self):
        schema = {"a": {"b": {"c": IniField("integer")}}}
        self.assertEqual(problems_of(ini_apply_schema, {"a": {"b": {"c": "z"}}}, schema),
                         ["a.b.c: cannot read 'z' as integer"])

    def test_unknown_keys_are_ignored_by_default(self):
        self.assertEqual(ini_apply_schema({"extra": 1}, {}), {})

    def test_strict_mode_reports_unknown_keys(self):
        self.assertEqual(problems_of(ini_apply_schema, {"name": "x", "extra": 1}, SCHEMA, strict=True),
                         ["extra: unknown key"])

    def test_strict_mode_reports_unknown_nested_keys(self):
        self.assertEqual(problems_of(ini_apply_schema, {"name": "x", "server": {"zzz": 1}}, SCHEMA, strict=True),
                         ["server.zzz: unknown key"])

    def test_strict_mode_quotes_odd_unknown_names(self):
        self.assertEqual(problems_of(ini_apply_schema, {"a b": 1}, {}, strict=True), ['"a b": unknown key'])

    def test_input_document_is_untouched(self):
        document = {"name": "x"}
        ini_apply_schema(document, SCHEMA)
        self.assertEqual(document, {"name": "x"})

    def test_array_default_is_copied(self):
        schema = {"a": IniField("array", default=[1])}
        first = ini_apply_schema({}, schema)
        first["a"].append(2)
        self.assertEqual(ini_apply_schema({}, schema), {"a": [1]})

    def test_error_message_lists_the_problems(self):
        with self.assertRaises(IniSchemaError) as caught:
            ini_apply_schema({}, SCHEMA)
        self.assertEqual(str(caught.exception), "name: missing required key")


class SchemaIntrospectionTests(unittest.TestCase):
    def test_defaults_are_collected_in_a_nested_dict(self):
        self.assertEqual(ini_schema_defaults(SCHEMA), {"server": {"port": 8080, "debug": False}})

    def test_copied_defaults_do_not_alias(self):
        schema = {"a": IniField("array", default=[1])}
        ini_schema_defaults(schema)["a"].append(2)
        self.assertEqual(ini_schema_defaults(schema), {"a": [1]})

    def test_defaults_of_an_empty_schema(self):
        self.assertEqual(ini_schema_defaults({}), {})

    def test_required_paths_are_dotted(self):
        schema = {"a": IniField("string", required=True), "s": {"b": IniField("integer", required=True),
                                                               "c": IniField("integer")}}
        self.assertEqual(ini_required_paths(schema), ["a", "s.b"])

    def test_schema_without_requirements_lists_none(self):
        self.assertEqual(ini_required_paths({"a": IniField("string")}), [])


if __name__ == "__main__":
    unittest.main()
