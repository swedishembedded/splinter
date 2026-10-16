# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from iniparse.diagnostics import IniInterpolationError
from iniparse.interpolate import (
    IniExpander, ini_expand_document, ini_expand_text, ini_stringify_reference, ini_tokenize_references,
)


def refusal(call, *args, **kwargs):
    try:
        call(*args, **kwargs)
    except IniInterpolationError as error:
        return str(error)
    raise AssertionError("no error raised")


class TokenizeTests(unittest.TestCase):
    def test_plain_text_is_one_literal(self):
        self.assertEqual(ini_tokenize_references("hello"), [("text", "hello", None)])

    def test_empty_text_has_no_tokens(self):
        self.assertEqual(ini_tokenize_references(""), [])

    def test_single_reference_is_one_token(self):
        self.assertEqual(ini_tokenize_references("${a.b}"), [("ref", "a.b", None)])

    def test_text_around_a_reference(self):
        self.assertEqual(ini_tokenize_references("x${a}y"),
                         [("text", "x", None), ("ref", "a", None), ("text", "y", None)])

    def test_two_references_in_a_row(self):
        self.assertEqual(ini_tokenize_references("${a}${b}"), [("ref", "a", None), ("ref", "b", None)])

    def test_fallback_follows_colon_dash(self):
        self.assertEqual(ini_tokenize_references("${a:-none}"), [("ref", "a", "none")])

    def test_empty_fallback_is_kept_distinct_from_none(self):
        self.assertEqual(ini_tokenize_references("${a:-}"), [("ref", "a", "")])

    def test_fallback_may_contain_colon_dash(self):
        self.assertEqual(ini_tokenize_references("${a:-x:-y}"), [("ref", "a", "x:-y")])

    def test_whitespace_in_reference_name_is_trimmed(self):
        self.assertEqual(ini_tokenize_references("${ a }"), [("ref", "a", None)])

    def test_doubled_dollar_escapes_a_reference(self):
        self.assertEqual(ini_tokenize_references("$${a}"), [("text", "${a}", None)])

    def test_escape_merges_with_neighbouring_text(self):
        self.assertEqual(ini_tokenize_references("a$${b}c"), [("text", "a${b}c", None)])

    def test_single_dollar_is_ordinary(self):
        self.assertEqual(ini_tokenize_references("cost $5"), [("text", "cost $5", None)])

    def test_brace_without_dollar_is_ordinary(self):
        self.assertEqual(ini_tokenize_references("{a}"), [("text", "{a}", None)])

    def test_unterminated_reference_is_refused(self):
        self.assertEqual(refusal(ini_tokenize_references, "ab ${cd"), "unterminated reference at offset 3")

    def test_empty_name_is_refused(self):
        self.assertEqual(refusal(ini_tokenize_references, "x${ }"), "empty reference name at offset 1")

    def test_fallback_without_name_is_refused(self):
        self.assertEqual(refusal(ini_tokenize_references, "${:-x}"), "empty reference name at offset 0")


class StringifyTests(unittest.TestCase):
    def test_string_value_is_embedded_unchanged(self):
        self.assertEqual(ini_stringify_reference("abc", "n"), "abc")

    def test_integer_value_is_embedded_as_digits(self):
        self.assertEqual(ini_stringify_reference(42, "n"), "42")

    def test_float_value_is_embedded_as_text(self):
        self.assertEqual(ini_stringify_reference(0.25, "n"), "0.25")

    def test_booleans_are_lower_case(self):
        self.assertEqual(ini_stringify_reference(True, "n") + ini_stringify_reference(False, "n"), "truefalse")

    def test_array_cannot_be_embedded(self):
        self.assertEqual(refusal(ini_stringify_reference, [1], "list"), "cannot embed array 'list' in text")

    def test_table_cannot_be_embedded(self):
        self.assertEqual(refusal(ini_stringify_reference, {}, "tab"), "cannot embed table 'tab' in text")


class ExpandDocumentTests(unittest.TestCase):
    def test_document_without_references_is_equal(self):
        document = {"a": 1, "b": "x", "c": {"d": [1, 2]}}
        self.assertEqual(ini_expand_document(document), document)

    def test_reference_embedded_in_longer_text(self):
        self.assertEqual(ini_expand_document({"a": "x", "b": "<${a}>"}), {"a": "x", "b": "<x>"})

    def test_whole_value_reference_keeps_the_type(self):
        self.assertEqual(ini_expand_document({"port": 80, "p": "${port}"}), {"port": 80, "p": 80})

    def test_whole_value_reference_may_be_an_array(self):
        self.assertEqual(ini_expand_document({"l": [1, 2], "m": "${l}"})["m"], [1, 2])

    def test_dotted_name_reaches_a_nested_table(self):
        document = {"s": {"host": "h"}, "url": "http://${s.host}/"}
        self.assertEqual(ini_expand_document(document)["url"], "http://h/")

    def test_reference_inside_a_nested_table(self):
        document = {"root": "/r", "s": {"dir": "${root}/s"}}
        self.assertEqual(ini_expand_document(document)["s"], {"dir": "/r/s"})

    def test_every_array_item_is_expanded(self):
        self.assertEqual(ini_expand_document({"a": "x", "l": ["${a}", "y"]})["l"], ["x", "y"])

    def test_reference_may_target_another_reference(self):
        document = {"a": "1", "b": "${a}2", "c": "${b}3"}
        self.assertEqual(ini_expand_document(document)["c"], "123")

    def test_forward_reference_to_a_later_key(self):
        self.assertEqual(ini_expand_document({"a": "${b}", "b": "late"})["a"], "late")

    def test_index_segment_reaches_an_array_element(self):
        self.assertEqual(ini_expand_document({"l": ["p", "q"], "x": "${l.1}"})["x"], "q")

    def test_booleans_embed_in_text(self):
        self.assertEqual(ini_expand_document({"f": True, "t": "flag=${f}"})["t"], "flag=true")

    def test_escaped_reference_becomes_literal(self):
        self.assertEqual(ini_expand_document({"a": "$${x}"}), {"a": "${x}"})

    def test_input_document_is_untouched(self):
        document = {"a": "x", "b": "${a}"}
        ini_expand_document(document)
        self.assertEqual(document, {"a": "x", "b": "${a}"})

    def test_key_order_is_preserved(self):
        self.assertEqual(list(ini_expand_document({"z": "${y}", "y": 1})), ["z", "y"])

    def test_same_reference_used_twice(self):
        self.assertEqual(ini_expand_document({"a": "x", "b": "${a}${a}"})["b"], "xx")

    def test_array_cannot_be_embedded_in_text(self):
        self.assertEqual(refusal(ini_expand_document, {"l": [1], "s": "v=${l}"}), "cannot embed array 'l' in text")

    def test_reference_to_a_table_is_refused(self):
        self.assertEqual(refusal(ini_expand_document, {"t": {"a": 1}, "s": "${t}"}), "reference 't' is a table")

    def test_invalid_reference_name_is_refused(self):
        self.assertEqual(refusal(ini_expand_document, {"s": "${a..b}"}), "invalid reference name 'a..b'")


class MissingReferenceTests(unittest.TestCase):
    def test_unknown_reference_is_an_error_by_default(self):
        self.assertEqual(refusal(ini_expand_document, {"a": "${nope}"}), "unknown reference 'nope'")

    def test_fallback_replaces_a_missing_reference(self):
        self.assertEqual(ini_expand_document({"a": "${nope:-dflt}"}), {"a": "dflt"})

    def test_fallback_is_ignored_when_the_target_exists(self):
        self.assertEqual(ini_expand_document({"x": "real", "a": "${x:-dflt}"})["a"], "real")

    def test_empty_fallback_gives_an_empty_string(self):
        self.assertEqual(ini_expand_document({"a": "[${nope:-}]"}), {"a": "[]"})

    def test_keep_mode_leaves_the_reference_text(self):
        self.assertEqual(ini_expand_document({"a": "x${nope}y"}, missing="keep"), {"a": "x${nope}y"})

    def test_keep_mode_for_a_whole_value(self):
        self.assertEqual(ini_expand_document({"a": "${nope}"}, missing="keep"), {"a": "${nope}"})

    def test_empty_mode_drops_the_reference(self):
        self.assertEqual(ini_expand_document({"a": "x${nope}y"}, missing="empty"), {"a": "xy"})

    def test_unknown_mode_is_refused(self):
        self.assertEqual(refusal(ini_expand_document, {}, missing="ignore"), "unknown missing mode 'ignore'")

    def test_fallback_ends_at_first_closing_brace(self):
        self.assertEqual(ini_expand_document({"x": "1", "a": "${nope:-$${x}}"})["a"], "$${x}")

    def test_fallback_text_is_not_expanded_again(self):
        self.assertEqual(ini_expand_document({"x": "1", "a": "${nope:-${x}"})["a"], "${x")


class CycleTests(unittest.TestCase):
    def test_key_pointing_at_itself_is_circular(self):
        self.assertEqual(refusal(ini_expand_document, {"a": "${a}"}), "circular reference: a -> a")

    def test_mutual_references_are_circular(self):
        self.assertEqual(refusal(ini_expand_document, {"a": "${b}", "b": "${a}"}), "circular reference: a -> b -> a")

    def test_cycle_through_a_nested_table(self):
        document = {"a": "${s.b}", "s": {"b": "${a}"}}
        self.assertEqual(refusal(ini_expand_document, document), "circular reference: a -> s.b -> a")

    def test_chain_names_only_the_loop_itself(self):
        document = {"a": "${b}", "b": "${c}", "c": "${b}"}
        self.assertEqual(refusal(ini_expand_document, document), "circular reference: b -> c -> b")

    def test_diamond_shaped_sharing_is_not_a_cycle(self):
        document = {"d": "x", "b": "${d}", "c": "${d}", "a": "${b}${c}"}
        self.assertEqual(ini_expand_document(document)["a"], "xx")

    def test_array_item_referencing_its_array(self):
        document = {"a": ["${a.0}"]}
        self.assertEqual(refusal(ini_expand_document, document), "circular reference: a.0 -> a.0")


class ExpandTextTests(unittest.TestCase):
    def test_text_is_expanded_against_a_document(self):
        self.assertEqual(ini_expand_text("hi ${n}", {"n": "bob"}), "hi bob")

    def test_whole_reference_keeps_its_type(self):
        self.assertEqual(ini_expand_text("${n}", {"n": 7}), 7)

    def test_missing_mode_is_honoured(self):
        self.assertEqual(ini_expand_text("a${n}b", {}, missing="empty"), "ab")

    def test_missing_reference_raises_by_default(self):
        self.assertEqual(refusal(ini_expand_text, "${n}", {}), "unknown reference 'n'")

    def test_expander_can_be_reused_for_many_texts(self):
        expander = IniExpander({"a": "${b}", "b": "x"})
        self.assertEqual((expander.expand_value("${a}"), expander.expand_value("${a}${b}")), ("x", "xx"))


if __name__ == "__main__":
    unittest.main()
