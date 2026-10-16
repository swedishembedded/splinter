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


class C1(unittest.TestCase):


    def test_text_around_a_reference(self):
        self.assertEqual(ini_tokenize_references("x${a}y"),
                         [("text", "x", None), ("ref", "a", None), ("text", "y", None)])


    def test_doubled_dollar_escapes_a_reference(self):
        self.assertEqual(ini_tokenize_references("$${a}"), [("text", "${a}", None)])

    def test_escape_merges_with_neighbouring_text(self):
        self.assertEqual(ini_tokenize_references("a$${b}c"), [("text", "a${b}c", None)])


    def test_unterminated_reference_is_refused(self):
        self.assertEqual(refusal(ini_tokenize_references, "ab ${cd"), "unterminated reference at offset 3")


class C2(unittest.TestCase):


    def test_booleans_are_lower_case(self):
        self.assertEqual(ini_stringify_reference(True, "n") + ini_stringify_reference(False, "n"), "truefalse")

    def test_array_cannot_be_embedded(self):
        self.assertEqual(refusal(ini_stringify_reference, [1], "list"), "cannot embed array 'list' in text")

    def test_table_cannot_be_embedded(self):
        self.assertEqual(refusal(ini_stringify_reference, {}, "tab"), "cannot embed table 'tab' in text")


class C3(unittest.TestCase):
    def test_document_without_references_is_equal(self):
        document = {"a": 1, "b": "x", "c": {"d": [1, 2]}}
        self.assertEqual(ini_expand_document(document), document)


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


    def test_index_segment_reaches_an_array_element(self):
        self.assertEqual(ini_expand_document({"l": ["p", "q"], "x": "${l.1}"})["x"], "q")

    def test_booleans_embed_in_text(self):
        self.assertEqual(ini_expand_document({"f": True, "t": "flag=${f}"})["t"], "flag=true")

    def test_escaped_reference_becomes_literal(self):
        self.assertEqual(ini_expand_document({"a": "$${x}"}), {"a": "${x}"})


    def test_array_cannot_be_embedded_in_text(self):
        self.assertEqual(refusal(ini_expand_document, {"l": [1], "s": "v=${l}"}), "cannot embed array 'l' in text")

    def test_reference_to_a_table_is_refused(self):
        self.assertEqual(refusal(ini_expand_document, {"t": {"a": 1}, "s": "${t}"}), "reference 't' is a table")


class C4(unittest.TestCase):
    def test_unknown_reference_is_an_error_by_default(self):
        self.assertEqual(refusal(ini_expand_document, {"a": "${nope}"}), "unknown reference 'nope'")


    def test_keep_mode_leaves_the_reference_text(self):
        self.assertEqual(ini_expand_document({"a": "x${nope}y"}, missing="keep"), {"a": "x${nope}y"})

    def test_keep_mode_for_a_whole_value(self):
        self.assertEqual(ini_expand_document({"a": "${nope}"}, missing="keep"), {"a": "${nope}"})

    def test_empty_mode_drops_the_reference(self):
        self.assertEqual(ini_expand_document({"a": "x${nope}y"}, missing="empty"), {"a": "xy"})


class C5(unittest.TestCase):

    def test_mutual_references_are_circular(self):
        self.assertEqual(refusal(ini_expand_document, {"a": "${b}", "b": "${a}"}), "circular reference: a -> b -> a")

    def test_cycle_through_a_nested_table(self):
        document = {"a": "${s.b}", "s": {"b": "${a}"}}
        self.assertEqual(refusal(ini_expand_document, document), "circular reference: a -> s.b -> a")

    def test_chain_names_only_the_loop_itself(self):
        document = {"a": "${b}", "b": "${c}", "c": "${b}"}
        self.assertEqual(refusal(ini_expand_document, document), "circular reference: b -> c -> b")


    def test_array_item_referencing_its_array(self):
        document = {"a": ["${a.0}"]}
        self.assertEqual(refusal(ini_expand_document, document), "circular reference: a.0 -> a.0")


class C6(unittest.TestCase):

    def test_whole_reference_keeps_its_type(self):
        self.assertEqual(ini_expand_text("${n}", {"n": 7}), 7)

    def test_missing_mode_is_honoured(self):
        self.assertEqual(ini_expand_text("a${n}b", {}, missing="empty"), "ab")

    def test_missing_reference_raises_by_default(self):
        self.assertEqual(refusal(ini_expand_text, "${n}", {}), "unknown reference 'n'")


if __name__ == "__main__":
    unittest.main()
