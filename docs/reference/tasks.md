<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Tasks reference

A task is a request with a reference and the verifiers that grade an answer to it, written
from sources and addressed by content. Tasks are grouped in task sets; `tasks generate`
makes them, `tasks list` and `tasks show` read them. This page lists the kinds and what
each is graded by; the checks themselves are described in
[verification](../concepts/verification.md).

## Task kinds


Task kinds: `recall`, `taught` (something a person told an agent; graded by the terms of its reference sentence an answer carries and adds, see `absorb`), `explain`, `predict` (`predict-python` and `predict-lua` name its runtime), `construct`, `debug`,
`counterexample`, `transform`, `classify`, `retrieve`, `multi-turn`,
`combine` (written by the generator model) and `denoise` (a corrupted
passage to restore, no model needed). Kinds that run code need `python3`.
A question answered from the source (`recall`, `explain`,
`counterexample`, `classify`, `retrieve`, `multi-turn`, `combine`) names
its subject - the product, document, tool, component or version it is
about - so that it has exactly one answer; a kind whose instruction shows
its material or whose answer is computed by running code carries what its
answer depends on and names none.

A `predict` task's reference is what the pinned interpreter prints when the generator's program
runs, never what the generator claims it prints: a program that runs to completion, prints
something non-empty and at most 2,000 characters, and prints the same on a second run is admitted
with its executed output as the reference, whatever the claim; any other program is refused as
`checks_failed`. When the claim differs (ignoring whitespace), the task carries a privileged
`relabelled-from-execution` record, JSON with the `claimed` and the `executed` output, and an
acquisition round counts such tasks in `relabelled`, beside `refused`, in its record and in each of
its `batches`. Only `predict` kinds are relabelled; the record never reaches the student.

## Concepts

A task's concepts: those it declares; otherwise the (source, section)
pairs its evidence falls in (the section of the source part containing
each span's first byte); otherwise its task kind. Every solve through
`policy:<alias>` records the release the alias was (or `base`), and
`status` ranks concepts by their rolling pass rate (the newest 32 graded
solves) under the current release, weakest first, with their rates under
earlier releases; a teacher's open-book solve and a retry helped by a
critique are not counted. When the release gate's retention check fails
on an earlier release's suite, the concepts of the tasks the candidate
forgot are queued under `<state>/curriculum/queue/` (the `release` report lists
them), and the next `learn` generates new tasks from their sections.

## Variants

`tasks variants TASKSET-ID` writes up to `--per-task` (default 3) differently worded
questions about each task of a set, into a task set of their own. Each keeps the task's
reference and evidence, so the same verifiers grade it. Variants are measured by the release
gate and never trained on.

## Canaries

`tasks canaries TASKSET-ID` makes canaries of the workspace tasks of a set: each task with
its hidden assertions also required to fail, so that no honest change passes it. A candidate
that passes one is refused when the set is given to `release --canaries`.

## Picking and dividing sets

`tasks check-requests TASKSET-ID` drops the workspace tasks whose request does not state the
contract the reference change restores (a returned value or a raised error that no title speaks
of; the same rule refuses such tasks at generation as `unstated_contract`) and writes the rest,
unchanged, as a task set of their own, with the dropped counted by class.

`tasks pick --seed N TASKSET-ID...` divides the union of task sets into a picked set and the
rest by a stable hash of the seed and each task's family, using the practice exam's ranking.
`--family-share F` picks that fraction of the families, `--per-family N` at most that many
tasks of each. Naming the sets in another order changes nothing.

## Combining faults

`tasks combine TASKSET-ID --faults K --per-family N` makes harder tasks of the admitted
single-fault workspace tasks of one snapshot: each holds `K` independent faults (2 by
default), in `K` different files, and is graded by all of their hidden tests at once. Its
tree is the clean snapshot with every fault applied and none of the hidden tests; its check
is the union of the members' fail-to-pass and pass-to-pass commands with every hidden file
laid over; its reference change is the union of the members'; its request lists every
member's symptoms, says that several independent faults exist, and names no file. Its family
is the member families sorted and joined with ` + `, and the member task ids are kept as a
`combined_task` document beside the task (`--json` lists them).

A combination is admitted only when each fail-to-pass command fails on its tree, the reference
change makes it pass on two runs, and no single member's reference change does, with the
exposure rules of every task (the request quotes no changed line and no hidden test, and names
no hidden file). The rest are counted in `refused` by reason (`shared_files`,
`different_snapshots`, `hidden_overlap`, `leaks_change`, `not_self_contained`, `not_failing`,
`not_restorable`, `flaky`, `single_suffices`); a single task that cannot be combined at all
(not a workspace task, no reference change, a request not in the original wording, ...) is
counted once in `unusable`. A pass-to-pass command is one a member passed on its own snapshot;
on the combined tree another member's fault can break it, and it is required to pass after
the change.

`--per-family N` is the most combined tasks any one file takes part in; files and tasks are
drawn so that every file, and every pair of files, is met about as often as any other, and
`--max-tasks` (default 64) bounds the whole. The draw is a function of `--seed` (drawn and
printed when absent) and the tasks, whatever order they are stored in, and the same seed
admits the same tasks. Three sets are written: all of them, those of the exam, and the rest.
A combined task is an **exam task when any file it holds a fault of is an exam family**, and
the exam reservation, the dataset seal and the trained-on record all read a composite family
that way, so such a task is never trained on. `tasks pick` still treats a composite family as
a family of its own. A combined task exports with `tasks harbor export` like any workspace
task.

## Comprehension drills

`tasks drills REPO --rev REV` makes small prerequisite tasks from a repository's clean snapshot,
so a policy that misreads part of the code can learn what that code does (not the answer to a
repair task). A drill shows a short program that imports the repository's module and calls one
of its functions on concrete inputs, and asks what it prints; it is a closed-book task of kind
`predict-python` (Python files) or `predict-lua` (Lua modules), graded by the formal verifier
like any predict task: the answer is the output, exactly, in one fenced block.

No model is asked for anything:

- **Functions.** The module-level functions of every source file that is not a test (`--file
  PATH`, repeated, narrows it). Python is read by the parser; a Lua module is a file that returns
  a table `M` of `function M.name(...)` declarations at the left margin, closed by `end` at the
  left margin.
- **Inputs.** The literals the repository's own tests call each function with, and
  recombinations of them per argument position, drawn by `--seed` (drawn and printed when
  absent; the same seed on the same tree makes the same drills). Only arguments are taken from a
  test, and only when every argument of the call is a literal (a number, a string, a list or
  table of literals, `None`/`nil`): never an assertion, an expected value, a test's name or text.
  A function no test calls with literals, and which needs arguments, gets no drill
  (`no_test_inputs`). A Python test finds a function through `from module import name` or an
  imported module alias; a Lua test through `M.name(...)` or `wrapper(M.name, ...)` on the module
  named by `MODULE = "name"` or `local M = require("name")`.
- **Reference.** What the pinned interpreter (`python3`, or `lua` from `--runtime-dir`) prints
  when the program shown runs against the clean snapshot of `--rev`: the tracked files of that
  commit, never a task's broken tree and never a hidden test. Python programs print the result's
  `repr`, or `raised ErrorType: message`; Lua programs print `pcall`'s results, with a table
  result printed by a small serialiser the program itself carries (a table's own text is an
  address).
- **Admission.** The program runs to completion, prints something non-empty and at most 2,000
  characters that is not just `None`, prints the same on a second run (Python under another
  `PYTHONHASHSEED`, so a set of strings is refused), does not read the clock, randomness or the
  environment, and its output is not one an earlier drill of the same function has (so a
  function's drills differ in their answers). Each candidate is run in a fresh tree, so a function
  that writes files leaves nothing behind. The rest are counted in `refused` by reason
  (`no_test_inputs`, `does_not_run`, `empty_output`, `output_too_long`, `nondeterministic`,
  `undefined_by_the_language`, `uninformative`, `same_output`).

`--per-function N` (default 3) bounds the drills of a function. A drill's first concept is the
file it exercises (its family, as a repair task's is) and its second `file::function`; its
privileged records are the reference and a `drill-origin` record (language, file, function, tree
id, interpreter version, a digest of the program). A drill of a file the practice exam reserved
is an **exam drill**: the sets written are all of them, the exam's, and the rest, and an answer
to an exam drill is refused when a dataset is built, as an attempt at a repair task of that file
is. `tasks harbor export` writes a drill as a Harbor task whose instruction adds that the answer
goes to `answer.txt`, whose verifier rewards exactly the reference (whitespace aside) and keeps
the reference under `tests/`.

`--for-tasks TASKSET-ID` is the hook for a loop that retries repair tasks: drills are made only
for the functions that contain a line the reference change of each workspace task of the set
changes (the lines of the clean file, from the change's `+` side), so a policy can be trained on
its verified answers to them before it attempts those tasks again. The revision must be the one
the tasks were made from: a change that does not match the revision is refused. Files changed
outside any function are counted in the report's `scope`.

## Repository families

Eight small original projects, each learned by `splinter practice` from its own tests, are
catalogued in `tasks/catalog.toml`. Their layout, how the admitted task sets are regenerated and
the funnel counts are in [Task families](../../tasks/README.md).
