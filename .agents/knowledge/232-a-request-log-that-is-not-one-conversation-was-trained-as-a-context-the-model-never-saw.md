<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 232. A request log that did not read as one conversation was trained as a context the model never saw

**Lesson:** when a single agent's recorded requests did not read as one conversation, the policy-gradient and
step views fell back to the record its trajectory makes: Splinter's generic system prompt, no tool schema, the
trajectory's repaired arguments. Those records were supervised with the rollout's advantage, so a policy-gradient
dataset taught a conditional the model was never sent and set GRPO group baselines from it. The fallback was counted
only in the token report (`context_reconstructed`), which the factory's `dataset.json` and `report.md` do not carry.
Most of these logs were not rewritten at all: the reader refused one whole conversation over one answer it could not
carry. Read what a log holds before deciding it is not one conversation; never fill the gap from the trajectory.

## What the logs held (sven 5.9.2, two Vim runs, 2026-11-26)

Of the rollouts whose actions were trained from the trajectory (forward 134 of 2,906 written, 667 action records;
backward 79 of 1,052, 437 records), by first cause (forward / backward):

| Cause | Forward | Backward |
|---|---|---|
| An answer whose tool-call arguments are not JSON (an invalid escape, a literal tab): sven sent the call back repaired (or as `{}`), and the reader refused the whole conversation | 98 | 33 |
| An answer sven's runaway-argument guard stopped: dropped from the history, a note in its place, no trajectory step, so one request more than steps | 24 | 12 |
| A last request the run's time limit cut before it was answered, filed by sven as a step that says nothing: the policy-gradient reader paired calls with every agent step (native-sft already skipped silent ones) | 11 | 11 |
| The context rewritten between requests (compaction; a request sent again after an empty answer with the time line appended a second time to its last message, later requests without it; parallel calls sent back in the order they finished) | 1 | 23 |

## What changed

The request reader keeps one conversation where the requests extend one another: a silent step and an answer that
made no step are passed over, a dropped answer is no action, and an answer whose arguments are not JSON is no action,
counted `unparsed_arguments`, its repaired turn context (what the model was then sent). Where the requests do not
extend one another, each is rendered on its own as sent, its answer the one supervised turn, paired with its agent
step by the step's tool-call ids (in any order) or text; a step no request answered is left out and counted
`requests_unpaired`, and a rollout none of whose steps pairs keeps only its baseline record. Where requests were
recorded, the trajectory is never rendered. The views that read a conversation whole (feedback, preference,
transition) refuse one holding a left-out step. The online path (`project_calls`) shares the per-request rendering and
counts an unparsed answer `unparsed_arguments` (it was `unreadable_request`).

## Measured (the same corpora projected by the old and the new build)

The old build reproduces the exported products exactly (forward 14,598 records, backward 6,306, same exclusions).
Forward: all 134 rollouts now render from their requests (551 action records; 123 unparsed answers left out), none
baseline-only; 18 compacted rollouts that were baseline-only now render request by request (732 records); 1 step
counted `requests_unpaired`; 0 records keep the generic prompt. Backward: 78 of 79 render (391 records); the other has
every answer's arguments not JSON and keeps its baseline. Every other rollout's records are byte-identical. native-sft
refusals of passing trials: forward 21 to 0 (ReasoningInText 2 unchanged), backward 92 to 85; the 85 left are
rewritten contexts, which one record cannot hold.
