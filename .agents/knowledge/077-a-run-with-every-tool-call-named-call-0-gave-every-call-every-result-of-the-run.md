<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 077. A run with every tool call named `call_0` gave every call every result of the run

`brain serve` numbered the calls of each response from 0, so a recorded agent run held many calls with the id
`call_0`. `serialise_run_step` (`dataset build --view sft-step --serialise-calls`) paired a call's results by
`source_call_id == id`, so with one id every call received every result of the whole run. 147 of the 231 examples of
the CE4 primary dataset had duplicated, reordered and future tool results in their context. Nothing refused them:
the unanswered-observation check passed (the id existed) and brain's record check only asked that a result name an
earlier call. The adapter trained on them memorised its data (loss 2.5 to 0.04) and looped on one listing.

Structure, not vigilance, now closes the class, in three independent places:

* the projection (`try_conversation`) refuses a step in which two calls share an id (`duplicate_call_id`), pairs results to calls within their step, scopes the ids it emits to `s<step>-<id>` and
  checks, after projecting, that the tool messages and tool calls out equal the results and calls of the steps put in
  (`not_conserved`); both are counted in the dataset's `excluded` map;
* brain's record check (`data::chat::check_tool_pairing`, run by the file parsers and by every encode) refuses a
  conversation with a repeated call id, a result answering a call twice or before it was made; `BrainDatasetCheck`
  runs it on every dataset before it is stored;
* `brain serve` mints ids that are unique across responses (`call_<hex>_<index>`).

Any one of them would have stopped the corrupt dataset. A producer id scheme can regress; the refusal cannot.
