<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 239. Stages run at once produced a stage identity they share twice, and offered one adapter over each other

**Lesson:** one stage identity can stand at several keys of a run. An evaluation's baseline arm is named by what it is
(plan, agent, baseline model, tasks), so every evaluation of that baseline on those tasks shares it, and slots are
namespaced by the arm's name; every computed stage that reads a recipe input pins it as the same `_inputs/NAME` stage.
Run one after another, the second finds the product committed and reuses it. Run at once, both begin it: the second
transition of the shared run-state record fails with `Conflict { expected: 0, actual: 1 }`, and two baseline arms would
claim one set of slots. With the controller's per-identity claim removed, 5 of the 8 stage-concurrency tests failed
this way (three independent stages reading one recipe input; two evaluations of one baseline).

A second race was in the adapter offer: `AdapterOffer::Directory` copied the adapter to `.<sha>.partial` and renamed it
into place. Eight threads offering one adapter (three evaluations of one trained LoRA started together) failed 3 of 3
runs with `No such file or directory` on the rename: one thread renamed the partial file another was still writing.

**Fix:** the controller claims a stage identity for the thread that produces it (`run::Producing`): a second thread
that begins it waits, then reuses the committed product. The offer, and the release-manifest writer, write aside under a
name unique in the process (`manifest::aside`).

**How to apply:** before running anything at once, list what is named by content rather than by its caller (a shared
arm, a pinned input, an offered adapter, a cache entry); each is produced under a claim or a lock, and written under a
name no concurrent writer uses. The harbor worker's layer cache had the same shape (a partial file named by the
process alone, written by threads of one process).
