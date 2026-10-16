# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from mdrender.escapes import (
    mdr_escape_html,
    mdr_format_html_attributes,
    mdr_is_safe_url,
    mdr_percent_encode_url,
    mdr_strip_control_characters,
    mdr_unescape_html,
)


class C0(unittest.TestCase):
    def test_ampersand_becomes_named_reference(self):
        self.assertEqual(mdr_escape_html("a&b"), "a&amp;b")

    def test_greater_than_becomes_named_reference(self):
        self.assertEqual(mdr_escape_html("2>1"), "2&gt;1")

    def test_double_quote_becomes_named_reference(self):
        self.assertEqual(mdr_escape_html('say "hi"'), "say &quot;hi&quot;")

    def test_single_quote_becomes_numeric_reference(self):
        self.assertEqual(mdr_escape_html("it's"), "it&#39;s")

    def test_existing_reference_is_escaped_again(self):
        self.assertEqual(mdr_escape_html("&lt;"), "&amp;lt;")

    def test_script_tag_is_neutralised(self):
        self.assertEqual(mdr_escape_html("<script>x</script>"), "&lt;script&gt;x&lt;/script&gt;")


class C1(unittest.TestCase):
    def test_hex_reference_is_decoded(self):
        self.assertEqual(mdr_unescape_html("&#x41;&#x6a;&#x6A;"), "Ajj")

    def test_unknown_named_reference_is_kept(self):
        self.assertEqual(mdr_unescape_html("&bogus;"), "&bogus;")

    def test_zero_reference_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            mdr_unescape_html("&#0;")
        self.assertEqual(str(caught.exception), "invalid character reference: &#0;")

    def test_largest_code_point_is_accepted(self):
        self.assertEqual(mdr_unescape_html("&#x10FFFF;"), chr(0x10FFFF))

    def test_code_point_past_the_range_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            mdr_unescape_html("&#1114112;")
        self.assertEqual(str(caught.exception), "invalid character reference: &#1114112;")

    def test_first_surrogate_is_rejected(self):
        with self.assertRaises(ValueError):
            mdr_unescape_html("&#xD800;")

    def test_code_points_beside_surrogates_are_accepted(self):
        self.assertEqual(mdr_unescape_html("&#xD7FF;&#xE000;"), "\ud7ff\ue000")

    def test_text_around_references_is_preserved(self):
        self.assertEqual(mdr_unescape_html("a &lt; b &amp; c"), "a < b & c")


class C2(unittest.TestCase):
    def test_http_scheme_is_safe(self):
        self.assertTrue(mdr_is_safe_url("http://example.org/a"))

    def test_https_scheme_is_safe(self):
        self.assertTrue(mdr_is_safe_url("https://example.org"))

    def test_mailto_scheme_is_safe(self):
        self.assertTrue(mdr_is_safe_url("mailto:someone@example.org"))

    def test_ftp_scheme_is_safe(self):
        self.assertTrue(mdr_is_safe_url("ftp://files.example.org"))

    def test_scheme_check_ignores_letter_case(self):
        self.assertFalse(mdr_is_safe_url("JavaScript:alert(1)"))
        self.assertTrue(mdr_is_safe_url("HTTPS://example.org"))

    def test_tab_inside_scheme_is_ignored(self):
        self.assertFalse(mdr_is_safe_url("java\tscript:alert(1)"))

    def test_leading_space_before_scheme_is_ignored(self):
        self.assertFalse(mdr_is_safe_url(" javascript:alert(1)"))

    def test_digits_are_allowed_inside_a_scheme_name(self):
        self.assertFalse(mdr_is_safe_url("h9:x"))

    def test_bang_is_not_a_scheme_character(self):
        self.assertTrue(mdr_is_safe_url("ja!va:x"))


class C3(unittest.TestCase):
    def test_space_becomes_percent_twenty(self):
        self.assertEqual(mdr_percent_encode_url("a b"), "a%20b")

    def test_reserved_characters_are_kept(self):
        url = "https://h.org:80/p?q=1&r=2#f[x]@!$'()*+,;=~"
        self.assertEqual(mdr_percent_encode_url(url), url)


class C4(unittest.TestCase):
    def test_delete_character_is_removed(self):
        self.assertEqual(mdr_strip_control_characters("a\x7fb"), "ab")

    def test_plain_space_is_kept_here(self):
        self.assertEqual(mdr_strip_control_characters("a b"), "a b")

    def test_unit_separator_is_removed(self):
        self.assertEqual(mdr_strip_control_characters("a\x1fb"), "ab")


class C5(unittest.TestCase):
    def test_pairs_render_with_leading_space(self):
        self.assertEqual(mdr_format_html_attributes([("id", "x"), ("class", "y")]), ' id="x" class="y"')

    def test_attribute_values_are_escaped(self):
        self.assertEqual(mdr_format_html_attributes([("title", 'a"b<')]), ' title="a&quot;b&lt;"')


if __name__ == "__main__":
    unittest.main()
