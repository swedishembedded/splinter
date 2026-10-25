<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# On-policy expert corrections: why our SFT cannot transmit, and what to build

Status: design note, nothing implemented. Written 2026-10-24 from measurements on this
stack, not from a literature claim.

## The measurement that motivates it

Four teacher-trajectory SFT recipes failed to improve task completion on the pilot set S1:
CE8-T (packed), its shuffled control, u1b (decision-level), and r2 (twice the data). The
positive control run on 2026-10-24 is the decisive one: an adapter trained on **the
verified successful trajectories of the very tasks it is then evaluated on** bought one net
task out of fourteen (greedy decode: base 6/14, control 7/14, 3 fail -> pass and 2
pass -> fail). Untrained tasks were identical. Training loss fell, held-out likelihood fell,
and four epochs of the same corpus overfit it (train 0.042, held-out 0.154 against base
0.128). More data of the same shape did nothing (r2: 9 verified passes, same as base).

So the gap is not corpus size and not the block bound (though that dropped 47% of the
longest trajectories, and is now fixed). The gap is that a demonstration from a clean start
does not tell the student what to do from the state it actually reaches.

## The mechanism

A student that could solve the task by following the teacher would not need training: the
failure is that its own actions lead it to states the teacher's trajectory never visits.
Supervised imitation from a teacher-only distribution is exactly the covariate-shift
setting: the student is trained on states it does not encounter at inference, and inference
states are unseen in training. Its error compounds and the episode ends in a stop, a
repeated call or a wrong patch, none of which the training data shows how to recover from.

This is visible in our own numbers: the adapters shorten episodes (median 32 steps for the
base, 12-16 for the adapters) while raising exposure (hidden-test failures 8 -> 9 -> 11 as
training grows). They learn to conclude, not to recover.

## The design

At the first point where a student attempt is stuck, freeze its state and have a stronger
teacher continue *from there*, through the same tool interface. Train on the teacher's
continuation with the student's prefix masked.

1. **Attempt.** The student runs a task in the normal practice environment and fails (or is
   stuck: no progress over N actions, a repeated call, a failing test loop).
2. **Fork.** Capture the workspace exactly as the student left it (the task already
   snapshots a tree) and the transcript up to the fork step. This must be a fresh process
   replaying the recorded actions, not a live session: the workspace, the tool results and
   the visible transcript hash must match the student's at that step, or the fork is refused.
3. **Teach.** The teacher continues from the fork under the same system prompt, tool
   schemas and workspace, with the same step budget as the student's remaining budget. It
   never sees hidden tests, the reference diff or the task's privileged evidence - only what
   the student could see.
4. **Verify.** The final workspace is graded in a separate workspace by the task's hidden
   checks, exactly as a student attempt is. A teacher continuation that does not verify is
   discarded, not softened into a weaker label.
5. **Record.** One sft-step record per admitted teacher action: the student transcript is
   context (`train: false`), the teacher's actions are supervised (`train: true`), the
   provenance names which model produced each action. Token ids, tool schemas and sampling
   parameters are the same shape the existing view already emits.
6. **Train.** The registered recipe (rank 16, alpha 32, lr 5e-5, one epoch) on the
   corrected records, then the frozen evaluation build at the registered protocol *and* at
   greedy.

## What must change in each repository

- **sven**: a session must be able to start from a recorded transcript and a workspace
  snapshot, driven by a different model than the one that produced the transcript, and to
  continue the same tool loop. Today a session starts clean; the harness has no fork.
- **splinter**: a `correction` stage that selects failures, requests forks, records the
  teacher continuation with per-action provenance, verifies it, and projects it through the
  existing sft-step view. Reuse the practice environment and the experience store; do not
  add a second trajectory format.
- **brain**: nothing new. The record shape is the one the trainer already reads; the masked
  prefix is what `train: false` already means.

## Acceptance

Run on D46 (46 families, greedy) and S1, against the same-token ordinary teacher corpus:
a paired exact sign test over families, plus retention on the previous generation. The
criterion is a positive held-out gain at the same inference budget; a flat result is
reported as flat and the next single-variable change is named before the run.

## Risks

- A teacher that is only slightly stronger adds little; measure the teacher's own fork
  success rate first, and report it beside the result.
- Corrections at failure states oversample hard tasks; weight or cap per family.
- The prefix must be the student's *actual* transcript, not a reconstruction that flatters
  it; a mismatch here reproduces the same covariate shift with extra steps.
- The teacher must never see grading material, or the correction teaches an unavailable
  solution - the same rule the practice environment already enforces for the student.

## What the first collection measured (2026-10-25)

Fourteen selected attempts from the wins-first adapter's D46 failures (one per task, 34 failed
attempts in the set), teacher the served 27B, fork verified against each attempt's own
recording:

| outcome | count | why |
|---|---|---|
| the student's transcript compacted | 8 of 14 | `step N is a system turn inside the conversation`: the episode replaced its own context before it failed, so there is no faithful transcript to continue |
| an observation does not replay | 4 of 14 | a test run's wall clock (which replays when the timing coincides - refusal is a coin flip there), a `grep -r` whose matches came in another order, and two listing totals (fixed: a copied tree allocates other blocks) |
| forkable | 2 of 14 | one clean, one whose timing-dependent observation replayed |
| continued and verified | 3 of 6 continuations, on 2 tasks | over three runs (32k and 64k teachers); the teacher's continuation has to pass the task's hidden checks, and it does so about half the time |

Two consequences for the design, both measured rather than assumed:

- **The corpus is limited by the recorder, not the teacher.** Most failures that a teacher
  could correct cannot be *continued* at all, because the student compacted its context while
  it worked. A solve run whose failures are meant to be corrected has to record the episode
  without proactive compaction (fail on the context limit instead), and that is a change to
  the recording protocol, not to the correction stage.
- **A continuation must not compact either.** At a 32k teacher, both of the first run's
  verified corrections had compacted their own context, and the dataset rightly left them out
  (`Compacted`), since the prefix a record would supervise is a summary then. The continuation
  now runs with proactive compaction at 1.0 of the budget: an episode that outgrows its window
  ends at the budget gate with every message.
- **The student's episode usually ends near the context limit**, so the continuation needs
  room of its own: the teacher bank serves 64k (`CTX=65536 ./serve-bank.sh 0 8791`) and a
  continuation that still compacts is refused by the projection rather than taught, since the
  prefix it would supervise is no longer the student's.

Admission is also stochastic: one attempt verified on one continuation and not on another.
A correction corpus therefore has to be collected over more attempts than the number of
records needed, and the fork success rate, the compaction rate and the verification rate are
reported together.
