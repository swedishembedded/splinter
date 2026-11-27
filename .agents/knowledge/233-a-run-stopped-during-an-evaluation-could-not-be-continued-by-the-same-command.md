<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 233. A run stopped during an evaluation could not be continued by the same command

**Lesson:** a worker stopped mid-trial leaves the trial's sandbox on disk (`<trial>/sandbox-<session>`: the image
copy's overlay, with directories it made unreadable and links into host directories it had mounted). The controller
publishes every batch directory of an attempt stage as part of its product, hard-linking the tree, so the next
invocation finished every lost attempt in a new batch and then failed on the old batch's dangling link
(`overlay/upper/splinter/host/ld.so: No such file or directory`); the stage could never commit. A worker's reconcile is
the one step that sees an interrupted batch, so it removes what the sandboxes of its unfinished trials left.

## How it showed (2026-11-27)

Testing the paper's reproduction procedure (`papers/generalized-continuous-learning/appendix-reproduction.tex`): the
smoke run's arm-b fork was stopped by a time limit while both evaluation arms had trials running. Running the same
`splinter fork` again reused every committed stage, trained nothing again, ran the four lost T1 attempts in batch 2
and failed publishing `eval-unseen` on batch 1's leftover sandbox (two trials, each with an overlay `work/work` of
mode 000 and a `splinter/host/ld.so` link to the host loader the sandbox had bound).

## What changed

`splinter-harbor-worker reconcile` removes every `sandbox-*` directory of a trial that has no finished result
(`runner.remove_sandboxes`, with the tree removal that restores the owner's permissions first); the trial's records
stay. The controller reconciles each batch of its unfinished attempts before it launches anything, so the batch it
later publishes holds no sandbox. Spec: `workers/harbor/tests/test_run.py`,
`test_reconcile_removes_what_the_sandbox_of_an_unfinished_trial_left`.

Not covered: a product that already failed this way. Its lost attempts were finished in a later batch, so no restart
reconciles the old batch again; remove the old batch's `sandbox-*` directories by hand, or fork the run again.
