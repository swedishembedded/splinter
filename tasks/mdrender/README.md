<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# mdrender

Text layout and a lightweight markup renderer, standard library only. It wraps and
aligns text, parses a small markup subset into plain dict/list trees, and renders
those trees as HTML or wrapped plain text, with table of contents and pipe-table
helpers.

Run the tests from this directory with `python3 -m unittest`.

## Data model

Inline nodes are dicts with a `type`:
`{"type": "text", "text": s}`, `{"type": "emph" | "strong", "children": [...]}`,
`{"type": "code", "text": s}`,
`{"type": "link", "href": s, "title": s or None, "children": [...]}`,
`{"type": "softbreak"}`, `{"type": "linebreak"}`.

Blocks are dicts with a `type`:
`{"type": "heading", "level": 1-6, "children": [inline...]}` (an optional `"id"`),
`{"type": "paragraph", "children": [inline...]}`, `{"type": "rule"}`,
`{"type": "code", "info": s, "text": s}`, `{"type": "quote", "children": [blocks]}`,
`{"type": "list", "ordered": bool, "start": int or None, "tight": bool, "items": [[blocks], ...]}`.

A table is `{"header": [str], "align": ["left"|"right"|"center"], "rows": [[str]]}`.

## Modules and public API

All functions are re-exported by the `mdrender` package.

### wrapping

- `mdr_split_words(text)`: words split on whitespace runs.
- `mdr_split_hyphens(word)`: pieces, split after a hyphen that sits between two
  letters or digits (`"well-known"` gives `["well-", "known"]`).
- `mdr_break_long_word(word, width, hyphenate=False, first_width=None)`: chunks of at
  most `width` (the first chunk may use `first_width`); with `hyphenate` each cut gets
  a trailing `-`. `ValueError("width must be positive")`;
  `ValueError("hyphenation needs a width of at least 2")`.
- `mdr_wrap_text(text, width, first_indent="", rest_indent="", break_long=True, hyphenate=False, split_hyphens=True)`:
  greedy wrap to a list of lines of at most `width` characters, indents included.
  Overlong words are broken (or kept whole with `break_long=False`).
  `ValueError("indent leaves no room for text")` when an indent fills the width.
- `mdr_fill_paragraphs(text, width, first_indent="", rest_indent="")`: wrap every
  blank-line separated paragraph, joined by one blank line.
- `mdr_unwrap_lines(lines)`: one line, breaks collapsed to single spaces.
- `mdr_truncate_text(text, width, marker="...")`: cut with a trailing marker;
  `ValueError("width is smaller than the marker")`.
- `mdr_count_words(text)`, `mdr_longest_line_length(lines)`.

### alignment

- `mdr_pad_line(text, width, align="left", fill=" ")`: left, right or center (the
  odd space goes right). `ValueError("unknown alignment: X")`,
  `ValueError("fill must be one character")`.
- `mdr_justify_line(line, width)`: spread words over `width`, extra spaces to the
  leftmost gaps; a line that cannot be justified is collapsed to single spaces.
- `mdr_justify_lines(lines, width, justify_last=False)`: justify all but the last
  line of each paragraph; blank lines pass through.
- `mdr_align_lines(lines, width, align)`: left, right, center or `"justify"`.
- `mdr_center_block(lines, width=None)`: center on the widest line by default.
- `mdr_indent_block(text, prefix)`: prefix non-blank lines.
- `mdr_dedent_block(text)`: remove the common margin of spaces; blank lines become empty.
- `mdr_apply_hanging_indent(lines, first_prefix, rest_prefix)`.
- `mdr_strip_trailing_space(lines)`, `mdr_trim_blank_edges(lines)`.
- `mdr_join_columns(left_lines, right_lines, left_width, gap=2)`: side by side blocks.

### inline

- `mdr_parse_inline(text)`: nodes for backslash escapes, code spans (any backtick
  run length), `*emph*`, `_emph_`, `**strong**`, `__strong__`, `***both***`,
  `[text](href "title")`, `<http://autolinks>` (http, https, mailto), soft breaks and
  hard breaks (two trailing spaces or a trailing backslash). Intraword `_` is literal.
  Unmatched delimiters stay literal text.
- `mdr_collect_links(nodes)`, `mdr_collect_code_spans(nodes)`: hrefs and code texts in order.

### blocks

- `mdr_parse_blocks(text)`: heading, paragraph, rule, code, quote and list blocks.
  ATX headings only; fenced code with at least three backticks or tildes (unclosed
  fences run to the end); lists with bullets `- + *` or numbers `1.` `1)` and any
  nesting by indentation; a list is loose when a blank line separates items or
  sits inside an item; lazy continuation lines join the item paragraph. Leading tabs
  count as four spaces.
- `mdr_match_atx_heading(line)`, `mdr_is_thematic_break(line)`,
  `mdr_match_code_fence(line)`, `mdr_match_list_marker(line)`: the line recognisers
  (`None` when the line does not match).
- `mdr_walk_blocks(blocks)`: generator over all blocks, nested ones included.
- `mdr_count_blocks(blocks)`: counts by block type.

### escapes

- `mdr_escape_html(text)`, `mdr_unescape_html(text)` (named references amp, lt, gt,
  quot, apos and numeric ones; `ValueError("invalid character reference: &#0;")` for
  zero, past U+10FFFF or surrogates).
- `mdr_is_safe_url(url)`: relative URLs and http, https, mailto, ftp only; ignores
  case, whitespace and control characters inside the scheme.
- `mdr_percent_encode_url(url)`, `mdr_strip_control_characters(text)`,
  `mdr_format_html_attributes(pairs)`.

### html_render

- `mdr_render_inline_html(nodes)`: escaped text; links with unsafe URLs keep only their text.
- `mdr_render_html(blocks)`: tight list items unwrap paragraphs; `id` attributes on
  headings that carry one. `ValueError("unknown block type: X")`,
  `ValueError("unknown node type: X")`, `ValueError("heading level out of range: N")`.
- `mdr_render_page(body, title, lang="en")`: full HTML document.

### text_render

- `mdr_render_inline_text(nodes)`: `_emph_`, `*strong*`, `` `code` ``, `label (href)`.
- `mdr_render_text(blocks, width=72)`: wrapped plain text; headings level 1 and 2 are
  underlined with `=` and `-`, deeper ones get `#` marks; quotes use `> `; code is
  indented four spaces; list markers hang and nested bullets alternate `* - +`.

### toc

- `mdr_slugify(text)`, `mdr_unique_slug(slug, used)`, `mdr_node_plain_text(nodes)`.
- `mdr_assign_heading_ids(blocks)`: copies with unique heading ids (`a`, `a-1`, ...).
- `mdr_collect_headings(blocks, levels=range(1, 7))`: entries `{level, text, id}` for
  top-level headings whose level is in `levels`.
- `mdr_build_toc(entries)`: nested tree; `mdr_number_toc(tree)`: `(label, depth, node)`.
- `mdr_render_toc_text(tree, indent=2)`, `mdr_toc_to_list_block(tree, ordered=False)`,
  `mdr_toc_depth(tree)`, `mdr_find_toc_entry(tree, entry_id)`.

### tables

- `mdr_split_table_row(line)`, `mdr_parse_alignment_row(cells)`
  (`ValueError("invalid alignment cell: X")`), `mdr_parse_table(lines)`
  (`ValueError("table needs a header and an alignment row")`,
  `ValueError("header and alignment row differ in column count")`).
- `mdr_column_widths(rows, min_width=1, max_width=None)`, `mdr_grid_width(widths)`,
  `mdr_fit_widths(widths, limit)` (`ValueError("table cannot fit in N columns")`),
  `mdr_fit_cell(text, width)`.
- `mdr_render_table_text(table, max_width=None)`, `mdr_select_columns(table, indexes)`,
  `mdr_sort_rows(table, column, descending=False)` (`IndexError("column index out of range: N")`).

## Example

```python
from mdrender import mdr_parse_blocks, mdr_assign_heading_ids, mdr_render_html, mdr_render_text

blocks = mdr_assign_heading_ids(mdr_parse_blocks("# Hello\n\nSome *text*.\n\n- a\n- b\n"))
print(mdr_render_html(blocks))
print(mdr_render_text(blocks, 30))
```

Swedish Embedded AB implements document conversion and text layout pipelines for its
clients. If your team needs expertise in lightweight markup rendering you can procure
our services by sending an email to info@swedishembedded.com.
