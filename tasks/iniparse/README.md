<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# iniparse

A parser, writer and toolbox for an INI/TOML-subset configuration format. It works on
strings (no files, no clock, standard library only) and covers: line scanning and
comments, typed scalars, arrays, sections and dotted table paths, error reporting
with line and column, merging of documents, path lookup with defaults, `${name}`
interpolation, a serializer that round-trips, and schema coercion and validation.

Run the tests from this directory with `python3 -m unittest`.

## Format

```
# comment            ; also a comment
name = "demo"        # trailing comment
port = 8_080         # integers: 42, -7, 1_000, 0xff, 0o17, 0b101
ratio = 0.5          # floats: 2.5, 1e3, -1.25e+2
debug = true         # booleans: true, false
path = 'C:\raw'      # single quotes: literal text
ports = [80, 443,    # arrays may span lines, trailing comma allowed
         8080]
[server.tls]         # dotted section path
enabled = false
limits.max = 10      # dotted key creates sub-tables
```

Strings in double quotes support `\n \t \r \" \\ \xHH \uHHHH`. Arrays must hold one
kind of value. A document is a plain `dict` (tables), `list`, `str`, `int`, `float`
and `bool` values. All errors derive from `IniError` and carry `message`, `line`
and `column` (1-based, 0 when unknown).

## Public API

### Reading

- `ini_load_document(text) -> dict`: parse a document; raises the first
  `IniSyntaxError` (or `IniDuplicateError` for a repeated key or section header,
  message `duplicate key 'a' (first defined on line N)`).
- `ini_collect_diagnostics(text) -> (dict, list[IniError])`: parse without stopping;
  keys under a bad section header are skipped until the next header.
- `ini_report_problems(text, limit=3) -> str`: all problems with source excerpt and
  caret, at most `limit` of them, `""` when the text is clean.
- `ini_scan_lines(text) -> list[IniLine]`: lines as `IniLine(number, end, kind, text,
  comment, indent)`; kind is `blank`, `comment`, `section` or `pair`; an array left
  open continues over following lines.
- `ini_decode_value(text, line=0, column=0)`, `ini_decode_scalar(...)`,
  `ini_decode_array(...)`: decode one value; `IniSyntaxError` with the position of the
  offending element.
- `ini_split_key_path(text)`, `ini_join_key_path(segments)`: dotted key paths with
  optional quoted segments, and the inverse.

### Merging

- `ini_merge_documents(base, override, arrays="replace", strict=False) -> dict`:
  deep merge into a new document. `arrays` is `replace`, `append` or `union`.
  With `strict=True` replacing a value by another kind raises `IniMergeError`.
- `ini_merge_many(documents, arrays="replace", strict=False)`: left-to-right fold.
- `ini_shared_leaf_paths(base, override) -> list[str]`: dotted paths both define.

### Lookup

- `ini_lookup(document, path, default=INI_NO_DEFAULT)`: `path` is dotted text or a
  tuple; numeric segments index arrays (negative counts from the end). Raises
  `IniLookupError` (also a `LookupError`) when absent and no default is given.
- `ini_lookup_kind(document, path, kind, default=INI_NO_DEFAULT)`: the same, and the
  value must be of the kind `boolean`, `integer`, `float`, `string`, `array` or `table`.
- `ini_lookup_first(document, paths, default=INI_NO_DEFAULT)`: first path that exists.
- `ini_has_path(document, path) -> bool`, `ini_list_paths(document) -> list[str]`.

### Interpolation

- `ini_expand_document(document, missing="error") -> dict`: replace `${a.b}` in
  strings. A value that is exactly one reference keeps the referenced type. `${a:-x}`
  gives a fallback, `$${` is a literal `${`. `missing` is `error`, `keep` or `empty`.
  Cycles raise `IniInterpolationError` (`circular reference: a -> b -> a`).
- `ini_expand_text(text, document, missing="error")`: expand a single string.

### Writing

- `ini_dump_document(document, width=72) -> str`: root pairs first, then one block
  per table; arrays longer than `width` are written one item per line. The output
  reads back to an equal document. Unsupported values raise `IniEncodeError`.
- `ini_encode_value(value) -> str`: text of one scalar or array.

### Schema

- `IniField(kind, required=False, default=None, minimum=None, maximum=None,
  choices=None, item_kind=None, max_items=None)`; kinds are `string`, `integer`,
  `float`, `boolean`, `array`.
- `ini_apply_schema(document, schema, strict=False) -> dict`: `schema` maps keys to
  `IniField` or to nested schema dicts. Values are coerced (`"8080"` to 8080, `"yes"`
  to True, `"a, b"` to a list), defaults filled, bounds and choices checked.
  All problems are collected into one `IniSchemaError` (`.problems`).
- `ini_coerce_value(value, kind)`, `ini_schema_defaults(schema)`,
  `ini_required_paths(schema)`.

### Diagnostics

- `ini_line_column(text, offset) -> (line, column)`, `ini_excerpt_with_caret(text,
  line, column)`, `ini_error_report(error, text)`.

## Example

```python
from iniparse import ini_load_document, ini_expand_document, ini_dump_document

doc = ini_load_document('root = "/srv"\n[app]\ndir = "${root}/app"\n')
doc = ini_expand_document(doc)
assert doc["app"]["dir"] == "/srv/app"
assert ini_load_document(ini_dump_document(doc)) == doc
```

Swedish Embedded AB implements configuration tooling that fails loudly with exact
positions for its clients. If your team needs expertise in configuration parsing,
layering and validation you can procure our services by sending an email to
info@swedishembedded.com.
