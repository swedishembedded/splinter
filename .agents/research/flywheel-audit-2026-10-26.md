<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Flywheel audit, 2026-10-26: silent failures and wasted capacity

Scope: every workflow and every hand-over between them in the harness flywheel
(`harnesses/flywheel/{invent,solve,validate}`, the host's side as the test driver does it,
`HarnessProposer`) and in the acquire loop, judged by one question: can it fail, or spend
compute, without anyone being told? Each finding names what happens, how it was found, and
what was done.

## Fixed

| # | Where | What would have happened | Fix |
|---|---|---|---|
| 1 | sven shell tool | A model sending `command` (the common name) instead of `shell_command` failed every call: 40 of 64 calls in one 27B run, and the proposer spent its turns retrying. | sven b4d31e72: `command` accepted; a call with neither names the keys it gave. |
| 2 | sven OpenAI driver | `response_format` was left out only for a provider named `brain`. The same server reached through the generic OpenAI wire format answered 400, and the whole invent run failed at its first schema'd step: 480 s and 162k input tokens of the 27B lost. | sven fdc34d93: a 400 naming `response_format` retries once without it, and the provider remembers; test with a refusing server. |
| 3 | sven harness spec | A schema'd model-only step inside a `for_each` body did not compile. | sven c77e6020. |
| 4 | sven turn watchdog | A looping turn ended the run as "the user cancelled the run", and `conclusion` said success. | sven 36f93e6c: the stop status `watchdog`; `conclusion` agrees with `status`. |
| 5 | invent: required variables set by tool call | Proposer and reviewer had to call `harness_vars_set`; the 27B answered in prose instead, every idea failed after two reminders, and an unset variable in `explore` fails the whole run. | Each agent reports in prose; a schema'd model-only step (`plan`, `formalize`, `decide`) turns the report into the typed result. A trusted gate captures the plan's angles for the loop. |
| 6 | invent: admit gate | A rejected idea ran `rmtree(tasks/<slug>)` before the slug was validated; an empty or `..` slug would have deleted the whole task repository with its history. | The slug is validated before any path is formed, in every mode. |
| 7 | invent: outcomes | A duplicate, a rejection and a failed idea left no trace, so a run's yield could not be audited. | Every outcome is a line of `tasks/log.jsonl`, committed with the tasks. |
| 8 | invent: task harness inputs | A task's own `harness.yaml` that declares a required input fails every solve attempt at start; the task would look unsolvable rather than broken. | The task gate refuses a harness whose `sven harness schema` lists required inputs; the author is told it takes none. |
| 9 | invent: tools on the PATH | The gates call `sven`, `git`, `python3`; a host that leaves one off the PATH would make every idea fail its gate. | The setup gate checks them and fails the run at its first step, naming the missing tool. |
| 10 | validate | The keep-or-discard decision was set by tool call: a judge that forgot left the attempt undecided and it was dropped as if discarded. | Judge reports in prose; a schema'd `decide` step gives the typed decision. |
| 11 | host (test driver) | Only the last conversation of a run was kept: a multi-stage task harness would have lost every stage but the last from the training record. | All conversations, in order, are the record and what validators read. |
| 12 | host (test driver) | An attempt whose harness run did not complete (a broken process) and a validator run that gave no decision were counted as a failed attempt and as a discard. | Reported apart: `runs_not_completed`, `validator_runs_without_decision`. |
| 13 | brain resident full fine-tune | Documented specs (`grad=bf16`, `state=bf16`) were refused on the embedding, which is no longer a store tensor. | brain 0acae2970. |
| 15 | invent: typed steps | A copy step ran at temperature 0.9 and copied an 800-word map into JSON; the 27B ran to the 8192-token limit and the run failed (`output_truncated`). | Typed steps carry only short fields (the plan extracts the angles; prompts read the survey's prose), and every copy or decision step decodes at temperature 0. |
| 16 | invent: `forget` | It removed every directory of `tasks/` without a `task.json`, `.git` included: the first failed idea erased the task repository's history, and the next idea's log commit failed (`rollback_failed`), ending the run after an hour. | Only a directory named like a task slug is ever removed; the index skips dot entries. |
| 17 | invent: empty proposal | A proposer whose turn ended without text was formalized anyway, and the schema'd step invented the slug `empty-proposal`; the author then tried to build a task from nothing. | `propose` verifies it wrote a proposal (at least 30 words), with two repair turns; still empty, the idea fails and is logged. |
| 18 | all harnesses: `/tmp` | Agents wrote their experiments to `/tmp` on a shared node. | Every prompt sends scratch files to `scratch/` in the working directory. |
| 14 | gates on commit | Twice a chained command committed after a failing gate (text hygiene, the architecture ratchet), because a pipe or `;` hid the status. | Fixed by follow-up commits; gate chains now `&&` under `pipefail`. |

## Open, with the consequence

- **A malformed structured answer fails the whole run.** brain has no constrained decoding,
  and sven sends a schema'd step's answer to the post-parse only; one that does not parse
  ends the run (`invalid_answer`) with no repair turn, even inside a `for_each` with
  `on_item_failure`. Admitted tasks survive (they are commits); the rest of the run is lost.
  To do in sven: a bounded repair turn with the parse error, then an item failure. Then in
  brain: grammar-constrained decoding for `json_schema`.
- **Harness trajectories keep no reasoning, and a truncated answer leaves no text.** A harness
  run's ATIF conversations are built from the thread's messages, and a message holds text only:
  every harness step with thinking on loses its reasoning from the training record. A turn cut
  off inside a `<think>` block (the 27B opened one with thinking off, at temperature 0.9) is
  reclassified as reasoning and the answer cleared, so 8192 generated tokens left an empty step
  and the run ended `output_truncated` with nothing to show why. To do in sven: carry reasoning
  in the thread's assistant messages into `reasoning_content`; keep a truncated turn's partial
  text and reasoning in its step; give a truncated or unparsable structured answer a bounded
  repair turn, then an item failure.
- **A tool call cut off at the output limit vanishes.** A proposer's first turn generated 8192
  tokens and left neither text nor a tool call: most likely a long shell call whose arguments the
  limit cut, which the stream decoder drops as incomplete. To do in sven: keep the unfinished call
  in the step and report the truncation to the model, so it can retry with a shorter call. Turns
  may now emit 24576 tokens.
- **Conversations are written when a run ends.** A run stopped by a signal loses every
  transcript (sven roadmap, harness-workflows item 1).
- **`HarnessProposer` reads the old contract** (`tasks.json`, `sections/NN.md`,
  `{count, kind, runtime}`). Against the current invent it fails loudly (sven exit 2), not
  silently, but splinter cannot consume task directories yet, and the request deadline it
  passes would kill an invent run. Splinter roadmap: flywheel item 12.
- **One writer per task repository.** `forget` removes every task directory without a
  `task.json`, and two invent runs sharing `tasks/` would remove each other's tasks in
  progress and race on git. Run one invent per repository at a time, or add a lock.
- **An agent cannot run a harness it wrote** (sven roadmap, harness-workflows item 4). Until
  then a task harness is compiled by the gate but first runs at solve time; a harness that
  compiles but cannot work shows only as runs that do not complete (finding 12 keeps those
  apart from hard tasks).
- **Untrusted task harnesses run gates on the host.** A task harness written by a model can
  declare command gates. Solve must run it in a sandbox before tasks from another source are
  admitted.
- **Capacity in the acquire loop.** A skill whose tasks can never be verified (randomness,
  `math.randomseed`) is chosen until its per-skill round limit, each round writing and
  refusing about 16 tasks; in one run 8 of 14 rounds were such skips. Bounded and recorded
  with the refusal counts, but every skip is a wasted round.
- **Training dominates a round** (210-300 s for 2-6 records at 4B) and the resident path
  was 3.4x slower than fp32 (single-threaded host decode while the card idled): being
  profiled and fixed in brain.
