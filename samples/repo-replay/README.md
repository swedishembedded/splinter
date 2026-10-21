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
   is wrong with it, a supervisor (the same model, in `supervisor_prompt.rs`) sees the task, the agent's
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
            [--splits OUT [--train-percent 80 --eval-percent 10]]   temporal train/eval/test datasets,
                                          splits.json (boundary commits) and one trace file per attempt
```

`--control plan-only` replays the same commits without the grounding diff (the plan and the repository
only, `plan_only_prompt.rs`); it needs a state directory of its own, and `report --state GROUNDED
--versus PLAN_ONLY` prints both pass rates side by side with the units only the grounded run passes
(what the diff carries) and the units both pass (the purest records). `--thinking` and
`--thinking-budget` turn on the model's reasoning for every request of a run.

The self-review step compares the agent's own `git diff` with the task (which the record holds), never
with the masked change (which it does not); the gate keeps flagging comparisons with the given change.

`--max-commits N` takes the N most recent replayable commits. `--seed` is required: no default is
fixed. Every number a run prints comes from the ledger (`state/ledger.jsonl`).

## Harness v2

`--driver harness` runs the plan and the work through sven's workflow language instead of the
Rust loops. The prompts live in `harness/replay.yaml` (plan stage and grounded work) and
`harness/replay-plan-only.yaml` (the control's work; a sven agent task cannot carry its own system
prompt, so the control is a document of its own): edit them and rerun, no build. `--harness-file`
and `--harness-plan-only-file` name edited copies. The documents declare the planner's refusal
retry (the gate `check-plan`), the tool allowance, the repetition stop, the follow-up for a turn
that ends announcing a call it did not make, and the supervisor corrections: the gate `check`
judges the tree and the calls so far (exit 0 passed, 1 a correction can repair it, 2 it cannot),
the supervisor writes in a user's voice, the gate `message_ok` refuses a message that holds the
change. Sampling, thinking, the reply limit (default 16384 tokens: sven's own 4096 is spent
entirely on reasoning by a thinking model) and the call budgets are inputs of the documents.

Rust keeps what is specific to replay: mining and selection, the masking, laying the trees, the
rules of the gate (`repo-replay check`, `check-message`, `check-plan` expose them to the documents
over the hidden side of the attempt, `case.rs`, which lies outside the agent's tree), the ledger,
the dataset, the scrub and the audit. The audit runs the honest solver, every cheater and the
supervisors through the documents as well and requires the Rust path's verdicts;
`tests/harness_spec.rs` requires every system prompt and user turn, the supervisor's included, to
equal what the Rust path sends. Differences from the Rust path: the supervisor is shown the wanted
change before the agent's changes (the Rust path now orders it the same way); the refusals of
supervisor messages are counted by sven without the last one when the rewrites ran out; the gate
`check` cannot know why a run stopped, so a correction is only earned by a turn that concluded.

The sample needs sven at or after 5e2aea0a; Cargo.lock still pins the published revision, so build
it with `make deps/local`.

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

## Limits

* The gate proves the tree is the commit's and the path to it was clean. It does not prove the
  agent understood the change: a plan that names the file and a diff small enough to copy make the
  task easy, which the bias report shows by size and style.
* Project checks are recorded, not gated, and run only when asked for.
* The shell is not a sandbox: the gate refuses attempts that name paths outside the tree, it does
  not stop them.
* Generated paths are excluded, not regenerated.

## Results

**One pilot run, five attempts: it shows the pipeline works end to end against a served 27B model, not
how well the method works.** Repository: a local copy of a research repository's history. Model: the
27B Q8_0 GGUF served by brain on a card shared with other jobs, one session, seed 20261020, no
reasoning block, terse and detailed plans, `--max-commits 3 --max-changed-lines 40 --max-files 2`.
Selection looked at 39 first-parent commits: 28 were oversize and 8 more went unselected because
`--max-commits` keeps the three most recent replayable ones. That left three commits and six planned
attempts.

| outcome | attempts |
|---|---|
| passed the gate | 2 (one commit, both plan styles; 4 and 3 tool calls) |
| failed the gate | 1 (`unmotivated_location`) |
| infrastructure error (server silent past the request limit), in no rate | 2 |
| not run (the run was stopped when requests passed two minutes) | 1 |

The failure is the gate doing its job: the agent edited the file named in the diff's header, which the
terse plan did not name and no search had shown. The two passes produced two records through splinter's
dataset build (the failed attempt is counted as an exclusion). In the passed attempts the agent's first read used the line offset of the diff's hunk; that run predates the masking of the diff and the `diff_offset_read` rule, which close it. Wall times were dominated by the shared
server's queue and are not a property of the method. The ledger and log of the run are kept outside
the repository with the command that produced them.
