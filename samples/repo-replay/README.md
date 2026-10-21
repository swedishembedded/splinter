<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# repo-replay - a git history turned into verified agent training data

**Status: the harness and its gate are checked with scripted models (`make samples/repo-replay/audit`,
no model, no GPU, no network). The results of real runs are in [Results](#results).**

## The question

A repository's history holds thousands of small, correct engineering steps. Can an agent be made to
*redo* each of them, in the tree as it stood before, from a task written the way a person would ask
for it, and can every redo that provably ends where the commit ended be kept as supervised
fine-tuning data? The agent is told the answer (the diff is in its system prompt) so that it
succeeds often; the student is never told it (the records carry the serving prompt and the plan).

## Pipeline, per commit, oldest to newest on the first-parent chain

1. **Select.** Merges, root commits, binary files, submodules and symlinks, oversize commits, commits
   that are mostly generated or vendored (lock files, `vendor/`, `*.min.js`, a `@generated` header)
   commits whose diff holds a secret, and commits whose diff takes more than a share (default a quarter,
   `--max-diff-context-percent`) of the served model's context (`--context-tokens`, else what the server
   reports; tokens estimated at one per three characters) are skipped, each counted under its own class
   (`diff_over_context_share` for the last). The source
   repository is cloned bare into the state directory and only the clone is read. Every `git`
   process is started through one helper that ignores user and system configuration, never prompts or
   pages, and passes what the output depends on as options; `git --version` is recorded per run in
   `runs.jsonl` beside the ledger.
2. **Plan.** A model writes the task as the user would (a terse and a detailed style), from the
   commit message and the diff. A plan that repeats a diff line of `--plan-verbatim-words` words or
   more, carries a diff marker or the commit id, speaks of "the diff", is outside its style's length
   bounds or is mostly code is refused.
3. **Replay.** The agent (sven's coding tools, in process) works in a fresh single-commit repository
   holding the commit's parent: no later history, no refs, no remote. System prompt: the grounding
   prompt with the diff, masked: hunk offsets are removed, every unchanged line is `xxx`, paths and
   removed and added lines are exact, so the agent has to find each place by searching and reading. First user message: the plan.
4. **Gate.** Deterministic, on the final tree and the recorded trajectory (rules below).
5. **Correct (optional, `--corrections N`).** When an attempt ends with a wrong tree and nothing else
   is wrong with it, a supervisor (the same model, prompt in `harness/replay.yaml`) sees the task, the agent's
   changes and the wanted change, and writes a follow-up message as the user would: intent, behaviour
   and place in words, no diff line, no line number, no name or number nobody gave the agent. A message
   that breaks that (`diff_line`, `location_offset`, `diff_reference`, `leaked_target`,
   `unjustified_content`) is refused, counted in the ledger and rewritten once or twice; the agent never
   sees it. The agent then continues in the same conversation. In the stored trajectory the agent's turns
   before the last correction are context and are not trained on; the final tree must still match exactly.
   Records are labelled `first_try` (plan, edits) or `corrected` (plan, attempt, feedback, fix).
6. **Store.** Every attempt is an immutable experience with the gate's verdict as an executable
   annotation; failed attempts are the negatives. The next commit's tree is the real one, never the
   agent's result, so attempts are independent and run in parallel (`--jobs`); results are stored in
   commit order.
7. **Dataset.** splinter's own `sft-step` build over the stored attempts, packed, under the serving
   prompt, validated by brain's dataset parser. Names, trailers, addresses and secrets are scrubbed
   from everything stored.

## The gate

| Class | Rule |
|---|---|
| `tree_mismatch`, `junk` | The final tree equals the commit's tree on every path: same bytes and mode; nothing missing, nothing extra (generated and runtime paths excepted). |
| `missing_edit` | Each changed file got `--min-edits-per-file` edit calls; a deleted or moved file got an `rm`/`mv` naming it. |
| `forbidden_shell` | No `git apply`/`patch`, no history-reaching git command, no here-document, redirection, `tee`, `sed -i` or interpreter write, no network, no path of the source or its clone. |
| `outside_tree` | No shell path above or outside the attempt tree. |
| `unread_edit` | A file that existed was shown (read, `cat`, `sed -n`, or a search in that file with context lines) before its first edit. |
| `unmotivated_location` | An edited or removed file was named by the plan or shown by an earlier search or listing. |
| `leaked_target` | A name that only the diff introduces appears in no call but an edit. |
| `unjustified_content` | Every name, number and string literal the code of an edit introduces is in the plan, in a tool result the agent saw earlier, in its own earlier edits or in a user's correction, or is already in the parent tree (a name the repository has is not new). Comments and prose files (markdown, text) are not looked at. Otherwise the plan was under-specified: the attempt fails with this class (no plan regeneration: the plan-retry loop judges a plan before an attempt exists, and redoing a whole attempt on a new plan is a different experiment). Names compare without letter case and a dashed crate name equals its underscore spelling; strings must appear verbatim. Names from the language's standard library the repository never uses count as new: the rule over-rejects there, and the bias report shows how much. |
| `diff_offset_read` | A `read_file` with an explicit offset whose range covers where a hunk starts is refused unless an earlier search named the file. |
| `diff_reference` | The agent's words never refer to the diff, the system prompt or its instructions. |
| `tool_calls_out_of_range` | Between `--min-tool-calls` and `--max-tool-calls` calls. |
| `run_incomplete` | The run concluded by itself (no cap, repetition stop or deadline). |

Project checks given with `--check` run in the attempt tree after the edits; their outcome is
recorded in the ledger and never gated. They execute the history's code on this machine.

An attempt's tree is removed when the attempt ends. A killed run cannot, so every attempt holds a
lock on a marker in its directory and the next run removes the directories nobody holds.

## Commands

```text
repo-replay audit [--work DIR]            the gate checks itself (also: make samples/repo-replay/audit)
repo-replay run REPO --state DIR --base-url URL --api-key-file FILE --model NAME --seed N
            [--from REV] [--to REV] [--max-commits N] [--styles terse,detailed] [--jobs N] ...
repo-replay report --state DIR [--json]   bias report: pass rate by size and style, reasons, skips,
            [--versus CONTROL_DIR]        first_try and corrected passes; or grounded beside plan-only
repo-replay regate REPO --state DIR [--write]   judge the passed attempts again with the current gate, no
            model; report only unless --write (new ledger line + failing verdict; try it on a copy first)
repo-replay dataset --state DIR [--serving-prompt-file FILE] [--max-block TOKENS]
            [--min-corrections N] [--max-corrections N]   0 only is first_try, 1 and more corrected
            [--tokenizer-dir MODEL_DIR]   count record lengths (and --max-block) with that model's tokenizer
            [--splits OUT [--train-percent 80 --eval-percent 10] [--drop-overlaps]]
                                          temporal train/eval/test datasets, splits.json (boundary
                                          commits, split digest), dataset-manifest.json and one trace
                                          file per attempt
```

`--control plan-only` replays the same commits without the grounding diff (the plan and the repository
only, `harness/replay.yaml`); it needs a state directory of its own, and `report --state GROUNDED
--versus PLAN_ONLY` prints both pass rates side by side with the units only the grounded run passes
(what the diff carries) and the units both pass (the purest records). `--thinking` and
`--thinking-budget` turn on the model's reasoning for every request of a run.

The self-review step compares the agent's own `git diff` with the task (which the record holds), never
with the masked change (which it does not); the gate keeps flagging comparisons with the given change.

`--max-commits N` takes the N most recent replayable commits. `--seed` is required: no default is
fixed. Every number a run prints comes from the ledger (`state/ledger.jsonl`).

## Harness v2

The plan and the work run through sven's workflow language, and nothing else runs them. Every
prompt lives only in `harness/replay.yaml` (the plan stage and the work of both controls: the
agent's system prompt is chosen by the `control` input): edit it and rerun, no build;
`--harness-file` names an edited copy. The documents declare the planner's refusal
retry (the gate `check-plan`), the tool allowance, the repetition stop, the follow-up for a turn
that ends announcing a call it did not make, and the supervisor corrections: the gate `check`
judges the tree and the calls so far (exit 0 passed, 1 a correction can repair it, 2 it cannot),
the supervisor writes in a user's voice, the gate `message_ok` refuses a message that holds the
change. Sampling, thinking, the reply limit (default 16384 tokens: sven's own 4096 is spent
entirely on reasoning by a thinking model) and the call budgets are inputs of the documents.

Rust keeps what is specific to replay: mining and selection, the masking, laying the trees, the
rules of the gate (`repo-replay check`, `check-message` and `check-plan` expose them to the
documents over the hidden side of the attempt, `case.rs`, which lies outside the agent's tree),
the ledger, `regate`, the dataset and splits, the scrub, and the audit with scripted models, which
runs the whole pipeline through the documents. `tests/prompts_spec.rs` and `tests/harness_spec.rs`
read the documents; the system prompts and user turns the models receive are the documents' text.
The serving prompt of the dataset is splinter's own coding-agent prompt (`splinter_sdk::vocabulary::prompt`), not a replay prompt.

The gate `check` is told how the agent's last turn ended (`--stopped`), so a correction is earned only
by a turn that concluded. The sample needs sven at or after 3e1f6b4d, which Cargo.lock pins.

## What the audit proves

On a fixture history (root, add, edit, rename, delete, three-file change, lock file and source,
lock file only, binary, merge, secret): the honest scripted solver passes the gate on every
replayable commit in both styles and every other commit is skipped for the right class; each of
cheater fails with the reason of the rule meant to catch it (including an honest edit under a plan
that gave none of the names it needs, and supervisors that quote the wanted change, give a line
number or speak of the diff); a correction repairs an attempt that left a file out, and its record
trains only on the fix; a plan-only run is flagged and a state refuses a second control; the same history gives the
same verdicts; the source is byte-for-byte untouched; every record opens with the serving prompt,
holds no part of the grounding prompt, and holds neither the author nor the trailer.

## Suitable for training

The dataset this sample builds is meant to be trained on; training and evaluating an adapter on it is
a later step and is not part of this sample. What makes the data suitable, and where it is checked:

| criterion | checked by |
|---|---|
| every record's tree is exactly the commit's (the gate compares trees) | the gate, per attempt; `audit` |
| no record holds the grounding prompt or any part of the change the agent was not shown | `audit` searches every record |
| no author, address or trailer, and no secret, in any record; a commit whose diff holds a secret is skipped | the scrubber; `audit`; the manifest counts what was removed |
| every record opens with the serving prompt | `audit`; the manifest's trainer preflight |
| train, eval and test are a cut by time over the whole range; both plan styles of a commit are in one split | splinter's `temporal_bands` (verified when built), `split_digest` in `splits.json` |
| no record of an earlier split repeats a later one's target change (a change and its revert, a copy) | `content_overlaps` over the changed lines; the build fails with the pairs unless `--drop-overlaps` leaves the training-side record out |
| brain trains the dataset, by the supervised regime, and its records fit the block | `dataset-manifest.json`: `trainable`, the examples, token lengths counted with `--tokenizer-dir`'s tokenizer against `--max-block`, what was left out by reason |

Secrets are found by splinter's `knowledge::redact` and replaced by `[REDACTED:<kind>]` (before that the
sample's own patterns left `[redacted]`). This affects new runs only: a stored state is never scrubbed
again, so a state started earlier keeps its old markers, and `redact` recognises more shapes than the
old patterns did, so more commits whose diff holds a secret are skipped.

`dataset-manifest.json` (beside `splits.json`) states: the source name and revision range, the
splinter, sven and brain revisions and the binary's digest, the digests of the harness documents,
the model, seed and sampling; records per split by type (`first_try`, `corrected`) and plan style;
token lengths (minimum, median, 90th percentile, maximum, and which counter measured them); the
commits skipped and the attempts that failed, by reason; the supervisor messages refused; and the
scrub totals (trailers, people, secrets) as counts. It is computed from the ledger and the stored
attempts, with no model and no device.

## Limits

* The gate proves the tree is the commit's and the path to it was clean. It does not prove the
  agent understood the change: a plan that names the file and a diff small enough to copy make the
  task easy, which the bias report shows by size and style.
* Project checks are recorded, not gated, and run only when asked for.
* The agent's shell is the sandbox's (Landlock on the process sandbox): it sees the attempt tree and
  its scratch directory (`$HOME`, `$TMPDIR`), so the source repository, the mirror and the case
  directory cannot be read from it, and the gate still refuses attempts that name such paths. What it
  changes for a run: `git` works inside the tree, `/tmp` and the cargo registry are not readable
  from the agent's shell (the project checks the host runs are not affected), and the gate commands
  run on the host, so the case directory is readable by them only. The sandbox needs a kernel with
  Landlock; without it the shell refuses to run. The file tools (read, edit, search) resolve inside
  the tree as before.
* Generated paths are excluded, not regenerated.

## Results

**One small demonstration on a public history, to show that the pipeline produces a dataset a trainer
accepts; it says nothing about how well training on it would work.**

Repository: splinter's own history (a bare scratch clone), the 59 first-parent commits up to
`378be432b`. Model: Qwen3.8-27B (Q8_0 GGUF) served by brain, seed 20261021, sampling temperature 0.7,
top-p 0.8, top-k 20, no reasoning block; terse and detailed plans, `--max-commits 12`, two jobs. The
run used a binary built at splinter `5aed5ef9f` and the dataset a later one (`f21dbf61d`), both with sven
`3e1f6b4d` and brain `dff565152`; the harness document digest is `blake3:789259c6...`.

The first pass looked back over the whole history (1,167 first-parent commits) for the 12 newest replayable
ones: 754 commits were oversize, 34 mostly generated, 18 held a secret, 7 binary, 4 had an ambiguous
hunk, 1 was the root. The 12 are all within the last 54 commits; the second pass bounded the range to the
59 commits the splits are cut over (`--from` 59 commits back) and asked again for the two refused plans.

| outcome (12 commits, 24 attempts) | attempts |
|---|---|
| passed the gate (all `first_try`, no correction was asked) | 8 (4 terse, 4 detailed) |
| failed the gate | 16 (reasons, an attempt can have several: `unjustified_content` 12, `tree_mismatch` 5, `unread_edit` 4, `diff_reference` 2, `outside_tree` 2, and one each of `diff_offset_read`, `forbidden_shell`, `run_incomplete`, `tool_calls_out_of_range`, `unmotivated_location`) |
| plan refused, infrastructure errors | 0 (two plans were refused in the first pass and written again in the second) |

Passing by size of the commit: 1-10 changed lines 5 of 8, 11-50 lines 3 of 12, 51-200 lines 0 of 4. The
gate fails most attempts for writing a name, number or text that neither the plan nor anything the
agent had read holds (`unjustified_content`): the replayed agent is held to the plan it was given.

`repo-replay dataset --state demo-state --splits OUT --train-percent 60 --eval-percent 20
--tokenizer-dir Qwen3.8-27B-FP8 --max-block 32768` then built:

| split | commits of the range | attempts | records | tokens per record (min / median / max) |
|---|---|---|---|---|
| train | 35 | 16 | 3 | 7,369 / 7,697 / 9,647 |
| eval | 12 | 4 | 3 | 6,030 / 6,101 / 6,359 |
| test | 12 | 4 | 2 | 5,575 / 5,818 / 5,818 |

The split has its content address (`split_digest` in `splits.json`), no pair of records of different
splits repeats a target change, no record is over `--max-block`, every record opens with the serving
prompt, and `dataset-manifest.json` reports that brain's supervised regime takes each split's dataset
(19, 13 and 25 supervised turns). The scrubber removed 37 people (names and addresses), 3 secrets and
0 trailers from the 24 stored attempts. Eight records are too few to train anything; the point is the
shape of the output. The ledger, the splits and the manifest of the run are kept outside the repository.
