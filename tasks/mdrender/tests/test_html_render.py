# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
import unittest

from mdrender.html_render import mdr_render_html, mdr_render_inline_html, mdr_render_page


def text(value):
    return {"type": "text", "text": value}


def para(value):
    return {"type": "paragraph", "children": [text(value)]}


def link(href, title, *children):
    return {"type": "link", "href": href, "title": title, "children": list(children)}


def bullets(*items, tight=True):
    return {"type": "list", "ordered": False, "start": None, "tight": tight, "items": list(items)}


def numbers(start, *items, tight=True):
    return {"type": "list", "ordered": True, "start": start, "tight": tight, "items": list(items)}


class InlineTests(unittest.TestCase):
    def test_text_content_is_escaped_fully(self):
        self.assertEqual(mdr_render_inline_html([text("a < b & \"c\"")]), "a &lt; b &amp; &quot;c&quot;")

    def test_strong_uses_the_strong_tag(self):
        self.assertEqual(mdr_render_inline_html([{"type": "strong", "children": [text("x")]}]),
                         "<strong>x</strong>")

    def test_nested_emphasis_inside_strong_is_rendered(self):
        node = {"type": "strong", "children": [text("a"), {"type": "emph", "children": [text("b")]}]}
        self.assertEqual(mdr_render_inline_html([node]), "<strong>a<em>b</em></strong>")

    def test_code_span_text_is_escaped(self):
        self.assertEqual(mdr_render_inline_html([{"type": "code", "text": "<b>"}]), "<code>&lt;b&gt;</code>")

    def test_soft_break_becomes_a_newline(self):
        self.assertEqual(mdr_render_inline_html([text("a"), {"type": "softbreak"}, text("b")]), "a\nb")

    def test_hard_break_becomes_a_br_tag(self):
        self.assertEqual(mdr_render_inline_html([text("a"), {"type": "linebreak"}, text("b")]), "a<br>\nb")

    def test_link_renders_href_and_text(self):
        self.assertEqual(mdr_render_inline_html([link("http://x.org/a", None, text("t"))]),
                         '<a href="http://x.org/a">t</a>')

    def test_link_title_is_added_and_escaped(self):
        self.assertEqual(mdr_render_inline_html([link("u", 'a"b', text("t"))]),
                         '<a href="u" title="a&quot;b">t</a>')

    def test_link_href_ampersand_is_escaped(self):
        self.assertEqual(mdr_render_inline_html([link("u?a=1&b=2", None, text("t"))]),
                         '<a href="u?a=1&amp;b=2">t</a>')

    def test_link_href_spaces_are_percent_encoded(self):
        self.assertEqual(mdr_render_inline_html([link("a b", None, text("t"))]), '<a href="a%20b">t</a>')

    def test_script_link_is_reduced_to_its_text(self):
        self.assertEqual(mdr_render_inline_html([link("javascript:alert(1)", None, text("t"))]), "t")

    def test_unsafe_link_keeps_its_formatted_children(self):
        node = link("data:x", None, {"type": "emph", "children": [text("t")]})
        self.assertEqual(mdr_render_inline_html([node]), "<em>t</em>")

    def test_empty_title_is_still_rendered(self):
        self.assertEqual(mdr_render_inline_html([link("u", "", text("t"))]), '<a href="u" title="">t</a>')

    def test_unknown_inline_node_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            mdr_render_inline_html([{"type": "blink"}])
        self.assertEqual(str(caught.exception), "unknown node type: blink")


class BlockTests(unittest.TestCase):
    def test_paragraph_is_wrapped_in_p(self):
        self.assertEqual(mdr_render_html([para("a & b")]), "<p>a &amp; b</p>")

    def test_blocks_are_joined_with_newlines(self):
        self.assertEqual(mdr_render_html([para("a"), {"type": "rule"}, para("b")]), "<p>a</p>\n<hr>\n<p>b</p>")

    def test_heading_level_selects_the_tag(self):
        block = {"type": "heading", "level": 3, "children": [text("T")]}
        self.assertEqual(mdr_render_html([block]), "<h3>T</h3>")

    def test_heading_with_an_id_gets_the_attribute(self):
        block = {"type": "heading", "level": 1, "id": "intro", "children": [text("Intro")]}
        self.assertEqual(mdr_render_html([block]), '<h1 id="intro">Intro</h1>')

    def test_heading_level_six_is_accepted(self):
        self.assertEqual(mdr_render_html([{"type": "heading", "level": 6, "children": []}]), "<h6></h6>")

    def test_heading_level_seven_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            mdr_render_html([{"type": "heading", "level": 7, "children": []}])
        self.assertEqual(str(caught.exception), "heading level out of range: 7")

    def test_heading_level_zero_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            mdr_render_html([{"type": "heading", "level": 0, "children": []}])
        self.assertEqual(str(caught.exception), "heading level out of range: 0")

    def test_rule_renders_as_hr(self):
        self.assertEqual(mdr_render_html([{"type": "rule"}]), "<hr>")

    def test_code_block_escaped_with_final_newline(self):
        block = {"type": "code", "info": "", "text": "if a < b:\n  x"}
        self.assertEqual(mdr_render_html([block]), "<pre><code>if a &lt; b:\n  x\n</code></pre>")

    def test_code_block_info_becomes_a_language_class(self):
        block = {"type": "code", "info": "python", "text": "x"}
        self.assertEqual(mdr_render_html([block]), '<pre><code class="language-python">x\n</code></pre>')

    def test_first_info_word_names_language(self):
        block = {"type": "code", "info": "c  extra words", "text": "x"}
        self.assertEqual(mdr_render_html([block]), '<pre><code class="language-c">x\n</code></pre>')

    def test_quote_wraps_its_children(self):
        block = {"type": "quote", "children": [para("a"), para("b")]}
        self.assertEqual(mdr_render_html([block]), "<blockquote>\n<p>a</p>\n<p>b</p>\n</blockquote>")

    def test_empty_quote_has_no_children_lines(self):
        self.assertEqual(mdr_render_html([{"type": "quote", "children": []}]), "<blockquote>\n</blockquote>")

    def test_unknown_block_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            mdr_render_html([{"type": "table"}])
        self.assertEqual(str(caught.exception), "unknown block type: table")


class ListTests(unittest.TestCase):
    def test_tight_bullet_list_unwraps_paragraphs(self):
        self.assertEqual(mdr_render_html([bullets([para("a")], [para("b")])]),
                         "<ul>\n<li>a</li>\n<li>b</li>\n</ul>")

    def test_loose_list_keeps_paragraph_tags(self):
        self.assertEqual(mdr_render_html([bullets([para("a")], [para("b")], tight=False)]),
                         "<ul>\n<li>\n<p>a</p>\n</li>\n<li>\n<p>b</p>\n</li>\n</ul>")

    def test_ordered_list_from_one_omits_start(self):
        self.assertEqual(mdr_render_html([numbers(1, [para("a")])]), "<ol>\n<li>a</li>\n</ol>")

    def test_ordered_list_from_five_has_start(self):
        self.assertEqual(mdr_render_html([numbers(5, [para("a")])]), '<ol start="5">\n<li>a</li>\n</ol>')

    def test_ordered_list_from_zero_has_start(self):
        self.assertEqual(mdr_render_html([numbers(0, [para("a")])]), '<ol start="0">\n<li>a</li>\n</ol>')

    def test_nested_list_follows_the_item_text(self):
        inner = bullets([para("b")])
        self.assertEqual(mdr_render_html([bullets([para("a"), inner])]),
                         "<ul>\n<li>a\n<ul>\n<li>b</li>\n</ul></li>\n</ul>")

    def test_empty_item_renders_an_empty_li(self):
        self.assertEqual(mdr_render_html([bullets([])]), "<ul>\n<li></li>\n</ul>")

    def test_empty_loose_item_renders_an_empty_li(self):
        self.assertEqual(mdr_render_html([bullets([], tight=False)]), "<ul>\n<li></li>\n</ul>")

    def test_tight_item_keeps_non_paragraph_blocks_rendered(self):
        item = [para("a"), {"type": "code", "info": "", "text": "x"}]
        self.assertEqual(mdr_render_html([bullets(item)]),
                         "<ul>\n<li>a\n<pre><code>x\n</code></pre></li>\n</ul>")


class PageTests(unittest.TestCase):
    def test_page_wraps_the_body_in_a_document(self):
        self.assertEqual(
            mdr_render_page("<p>x</p>", "T"),
            '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<title>T</title>\n'
            "</head>\n<body>\n<p>x</p>\n</body>\n</html>\n",
        )

    def test_page_title_is_escaped(self):
        self.assertIn("<title>a &lt;b&gt; &amp; c</title>", mdr_render_page("", "a <b> & c"))

    def test_page_language_can_be_chosen(self):
        self.assertIn('<html lang="sv">', mdr_render_page("", "T", lang="sv"))


if __name__ == "__main__":
    unittest.main()
