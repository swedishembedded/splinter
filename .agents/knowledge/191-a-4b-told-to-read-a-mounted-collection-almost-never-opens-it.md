<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 191. A 4B told to read a mounted collection almost never opens it

**Lesson:** a flat result for "collection + grounding instruction" against "none" on a small policy is first a result
about whether it read, not about the collection. Count the reads per trial (paths under the mount in the tool calls)
before you interpret a context arm. A null where nobody read the pages does not show that the pages are bad.

Seen 2026-11-20 (item 21 S1, 48 fresh practice tasks x 8, brain 5e25d5b82): A2s1 with the collection mounted and the
grounding text in its system prompt touched `/knowledge` in 6% of 382 trials and read `index.md` in 3%. It opened no page
of the task's domain. Several of its calls named paths that do not exist (`/knowledge/reference/rule_table.md`): it took
the mount for the task's own files. Without the instruction it never touched the mount (0 of 378). When the mount held
only the task's domain, A2s1 opened a domain page in 2 of 381 trials. Base Qwen3-4B followed the instruction more often
(index 13%, a domain page in 37 trials) but passed none of those 37. Pass@1 moved by at most 0.015 in any cell (A2s1 0.529
none, 0.527 collection + instruction). A2s1 was trained on 27B demonstrations that never read a collection, and its
habit wins over a system-prompt line.
