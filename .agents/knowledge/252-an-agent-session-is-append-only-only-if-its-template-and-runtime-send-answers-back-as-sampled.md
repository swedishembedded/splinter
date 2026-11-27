<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 252. An agent session is append-only only if its template and its runtime send each answer back as it was sampled

**Lesson:** a session merges into one training record (brain's rule `session_not_append_only`) only when every
call's prompt ids are the previous call's prompt ids, its generated ids and the turns between. Two things re-render an
earlier answer between calls, and either one makes the whole session one record per call:

- **The chat template.** Qwen3's (4B, 8B) writes an earlier assistant turn without its think block when that turn has
  no reasoning, while the model generated `<think>\n\n</think>\n\n` at that turn. Every Qwen3 session diverges at its
  second call, at the first id of the first answer. Qwen3.8's template keeps the earlier turns as generated.
- **The agent runtime.** sven sends an earlier call's arguments back re-serialised: whitespace compacted (Qwen3's
  template writes the argument text as given, so this alone also breaks the session) and, rarely, keys reordered
  (Qwen3.8's template writes the arguments parameter by parameter, so key order is the only thing that matters there).

**Evidence (2026-11-27, read-only sample of 178 recorded vim sessions from one archive of attempt passes, 59
Qwen3-4B, 59 Qwen3-8B, 60 Qwen3.8-27B-FP8):** each request rendered with its model's own chat template (arguments as
brain hands them to it), each answer as the template writes the conversation's last turn, then checked for the
append-only property. Mergeable: 0 of 59 (4B), 0 of 59 (8B), 58 of 60 (27B). Every Qwen3 failure is at call 1, the
dropped think block; the two Qwen3.8 failures are argument keys sven reordered (call 1 and call 7). At message level,
sven's next request repeated the generated argument text in 383 of 452 calls (4B), 370 of 462 (8B) and 3 of 406 (27B);
the rest differ in whitespace only, except 2 key reorders (27B), 31 arguments that were not JSON and 14 calls dropped
from the history (8B). Forwarded ids of the 58 mergeable sessions: 2,191,698 as one record per call against 444,003
merged (4.9x; median 3.85x per session). These sessions recorded no policy trace (no online run in the archive), so
the generated ids are taken as the template's rendering of the parsed answer; an online run's recorded ids are what
`SessionRecord::merge` checks at run time.

**Consequence:** merging is safe by construction (the producer falls back per call and counts why), but it saves
nothing on Qwen3 until the history is sent back as sampled: keep the earlier think block (a template choice the
policy was not trained under, so not a fix here) or train with thinking on, and have sven send each call's arguments
back as the model wrote them (text and key order), which is sven's to fix.
