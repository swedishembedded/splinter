<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Tasks reference

A task is a request with a reference and the verifiers that grade an answer to it, written
from sources and addressed by content. Tasks are grouped in task sets; `tasks generate`
makes them, `tasks list` and `tasks show` read them. This page lists the kinds and what
each is graded by; the checks themselves are described in
[verification](../concepts/verification.md).

## Task kinds


Task kinds: `recall`, `taught` (something a person told an agent; graded by the terms of its reference sentence an answer carries and adds, see `absorb`), `explain`, `predict`, `construct`, `debug`,
`counterexample`, `transform`, `classify`, `retrieve`, `multi-turn`,
`combine` (written by the generator model) and `denoise` (a corrupted
passage to restore, no model needed). Kinds that run code need `python3`.
A question answered from the source (`recall`, `explain`,
`counterexample`, `classify`, `retrieve`, `multi-turn`, `combine`) names
its subject - the product, document, tool, component or version it is
about - so that it has exactly one answer; a kind whose instruction shows
its material or whose answer is computed by running code carries what its
answer depends on and names none.

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

## Repository families

Eight small original projects, each learned by `splinter practice` from its own tests, are
catalogued in `tasks/catalog.toml`. Their layout, how the admitted task sets are regenerated and
the funnel counts are in [Task families](../../tasks/README.md).
