<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# linediff

Compare text and tabular data with the standard library only: a longest-common-subsequence
line diff that produces edit scripts, hunk grouping with context, unified-style rendering and
parsing, applying and reversing patches with conflict detection, a three-way merge of line
lists, a CSV reader and writer, a keyed CSV row diff, and similarity scoring of lines and words.

Run the tests from this directory with `python3 -m unittest`.

Lines are plain strings without line terminators; "text" means a list of such lines unless
stated otherwise. Positions are 0-based everywhere except in rendered patch text.

## Edit scripts (`linediff.editscript`)

An edit script is a list of `(tag, a_index, b_index)` tuples. Tag `"="` keeps `a[a_index]`
(equal to `b[b_index]`), `"-"` drops `a[a_index]` (`b_index` is where in `b` the drop happens),
`"+"` adds `b[b_index]` (`a_index` is where in `a` the addition happens). Any sequences work.

- `linediff_trim_common(a, b)` returns `(prefix, suffix)`, the counts of shared leading and
  trailing items; the two never overlap.
- `linediff_lcs_table(a, b)` returns `t` where `t[i][j]` is the common subsequence length of
  `a[i:]` and `b[j:]`.
- `linediff_lcs_length(a, b)` returns the longest common subsequence length.
- `linediff_edit_script(a, b)` returns the edit script; within a replaced region drops come
  before additions.
- `linediff_common_pairs(script)` returns the `(a_index, b_index)` pairs of kept lines.
- `linediff_edit_distance(script)` returns the number of drops plus additions.
- `linediff_change_blocks(script)` returns `(a_start, a_end, b_start, b_end)` per run of
  non-keep operations.

## Hunks (`linediff.hunks`)

- `LinediffHunk(a_start, a_count, b_start, b_count, lines)`: frozen dataclass; `lines` is a
  tuple of `(tag, text)` with tags `"="`, `"-"`, `"+"`. Methods `old_lines()` and `new_lines()`
  return the text before and after the hunk.
- `linediff_group_hunks(script, a, b, context=3)` groups a script into hunks with `context`
  kept lines around each change; changes separated by at most `2 * context` kept lines share a
  hunk. Raises `ValueError("context must not be negative")`.
- `linediff_hunk_stats(hunks)` returns `(added, removed)` line totals.
- `linediff_hunk_covering(hunks, a_line)` returns the index of the hunk whose old range holds
  the line, or `None`.
- `linediff_trim_hunk_context(hunk, context)` returns the hunk with at most `context` kept
  lines before and after its changes (same `ValueError` for negative context).

## Rendering (`linediff.unified`)

- `linediff_format_range(start, count)` formats a 0-based start and count as `"4,2"`, `"4"`
  (count 1) or `"3,0"` (count 0, start unshifted).
- `linediff_render_hunk_header(hunk)` returns `"@@ -old +new @@"`.
- `linediff_render_hunk(hunk)` returns the header plus body lines prefixed with space, `-`, `+`.
- `linediff_render_patch(hunks, old_name, new_name)` returns patch text with `---`/`+++`
  headers and a final newline; no hunks give `""`.
- `linediff_render_stat_line(hunks)` returns e.g. `"2 hunks, 3 insertions(+), 1 deletion(-)"`.
- `linediff_render_numbered_hunk(hunk, width=4)` returns body lines with right-aligned 1-based
  old and new line numbers (blank where a side has no line).

## Parsing (`linediff.patchparse`)

All failures raise `LinediffPatchError` (a `ValueError`).

- `linediff_parse_range(text)` parses `"4"`, `"4,2"`, `"3,0"` to a 0-based `(start, count)`.
- `linediff_parse_hunk_header(line)` returns `(a_start, a_count, b_start, b_count)`.
- `linediff_parse_patch_set(text)` returns `(old_name, new_name, hunks)` for every file patch.
  Hunk bodies are read by the counts in their headers; hunks must not overlap.
- `linediff_parse_patch(text)` returns the single file patch or raises if there is not exactly one.

## Applying (`linediff.patchapply`)

- `LinediffConflict(hunk_index, message)`: raised when a hunk does not fit.
- `linediff_reverse_hunks(hunks)` returns hunks that undo the given ones.
- `linediff_locate_hunk(lines, hunk, expected, max_shift=0, floor=0)` returns the start where
  the hunk's old text sits, trying `expected`, then `expected+1`, `expected-1`, ... up to
  `max_shift`; never before `floor` or past the end; `None` when nothing fits.
- `linediff_place_hunks(lines, hunks, max_shift=0)` returns a start (or `None`) per hunk; the
  offset found for one hunk carries to the next and hunks cannot overlap.
- `linediff_conflicting_hunks(lines, hunks, max_shift=0)` returns indices of misfitting hunks.
- `linediff_apply_patch(lines, hunks, max_shift=0)` returns the patched text or raises
  `LinediffConflict` for the first misfit; the input is not modified.
- `linediff_unapply_patch(lines, hunks, max_shift=0)` rolls the hunks back.

## Three-way merge (`linediff.merge3`)

Changes from the base touching or overlapping each other form one cluster; a cluster is a
conflict unless both sides agree or only one side changed it.

- `linediff_changed_spans(base, other)` returns `(start, end, replacement)` per rewritten range.
- `linediff_merge_chunks(base, ours, theirs)` returns `("clean", lines)` and
  `("conflict", ours_lines, theirs_lines)` chunks.
- `linediff_has_conflicts(chunks)` tells whether any chunk is a conflict.
- `linediff_render_merge(chunks, ours_label="ours", theirs_label="theirs")` flattens chunks into
  lines with `<<<<<<<`, `=======`, `>>>>>>>` markers around conflicts.
- `linediff_merge_three(base, ours, theirs, ours_label, theirs_label)` returns
  `(lines, conflict_count)`.

## CSV (`linediff.csvio`)

`LinediffCsvError` (a `ValueError`) signals bad text or table shape. A delimiter must be one
character that is not a quote or line break (`ValueError` otherwise).

- `linediff_csv_read(text, delimiter=",")` returns rows of fields. Quoted fields may hold
  delimiters, line breaks and doubled quotes; `\n`, `\r\n` and `\r` end rows; a blank line is a
  row with one empty field; a quote inside a plain field is literal. Errors: unterminated
  quoted field, character after a closing quote (both report the line).
- `linediff_csv_quote_field(value, delimiter=",")` quotes a value only when needed.
- `linediff_csv_write(rows, delimiter=",", newline="\n")` renders rows; a row without fields is
  an error.
- `linediff_csv_ragged_rows(rows)` returns indices of rows not as wide as the first row.
- `linediff_csv_pad_rows(rows, fill="")` pads every row to the widest.
- `linediff_csv_records(rows)` returns `(header, records)` with dict records; short rows are
  padded with empty strings, wider rows or repeated header names are errors.

## Keyed row diff (`linediff.csvdiff`)

`LinediffKeyError` (a `ValueError`) signals missing or non-unique keys.

- `LinediffTableDiff(added_columns, removed_columns, added, removed, changed)`: `added` and
  `removed` are `(key, record)` tuples, `changed` is `(key, {column: (old, new)})` tuples.
- `linediff_index_records(records, key_columns)` maps key tuples to records.
- `linediff_cell_changes(old_record, new_record, columns)` returns the differing cells.
- `linediff_diff_tables(old_rows, new_rows, key_columns)` compares two tables (header first)
  by key; only columns present in both are compared. Needs at least one key column.
- `linediff_table_diff_is_empty(diff)`, `linediff_table_diff_counts(diff)` (rows added, rows
  removed, cells changed) and `linediff_render_table_diff(diff)` (one line per difference).

## Similarity (`linediff.similarity`)

Scores are exact `fractions.Fraction` values in `[0, 1]`; two empty inputs score 1.

- `linediff_split_words`, `linediff_normalize_space`, `linediff_same_ignoring_space`.
- `linediff_word_overlap(a, b)` (shared words over total words, doubled) and
  `linediff_word_jaccard(a, b)` (distinct words).
- `linediff_sequence_ratio(a, b)` twice the common subsequence length over the total length,
  for strings or line lists.
- `linediff_line_similarity(a, b)` mean of word overlap and character ratio.
- `linediff_rank_matches(line, candidates)`, `linediff_best_match(line, candidates, floor=1/2)`,
  `linediff_pair_changed_lines(removed, added, floor=1/2)`.

## Example

```python
from linediff import *

old = ["a", "b", "c"]
new = ["a", "x", "c"]
hunks = linediff_group_hunks(linediff_edit_script(old, new), old, new, context=1)
text = linediff_render_patch(hunks, "old.txt", "new.txt")
_, _, parsed = linediff_parse_patch(text)
assert linediff_apply_patch(old, parsed) == new
```

Swedish Embedded AB implements reliable comparison, patching and merging of text and tabular
data for its clients. If your team needs expertise in diff, merge and CSV tooling you can
procure our services by sending an email to info@swedishembedded.com.
