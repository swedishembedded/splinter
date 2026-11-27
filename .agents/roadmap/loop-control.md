<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Loop control: observing and steering a generational learning loop

Scope: treating a generational learning loop as a control system: its state measured at every generation from the run's records (`splinter observe`), every recipe knob declared with the state variable it is meant to move and proven to move it (`splinter probe`), and what each measured knob did.

## Principle

A loop is observable when someone who did not run it can read off, from the tool's output alone, why a generation
did or did not improve; it is controllable when every knob a recipe turns names the state variable it moves and a
probe shows that it does. A knob that does not move its variable is a defect, whatever capability did.

## State vector (done, 2026-11-27)

`splinter observe RUN [RUN ...]` (`splinter_factory::observe`, SDK `splinter_sdk::factory::observe`) reads per
generation, from the recipe graph and the committed products, never changing the run:

| Group | Variables (JSON path under `generations[i]`) |
|---|---|
| data | `candidates[].data`: records and supervised characters per source stage and policy (own/other), in train and validation; `own_share_records`, `own_share_characters`, `validation_own_share_*`; `frontier` (tasks failed/sometimes/passed by the own policy, records from each by policy, `frontier_share`); per-source `pass_rate` and `mean_task_pass_rate` on the new tasks; `action_sequences`; `supervised_tokens` absent (not recorded) |
| training | `candidates[].training`: `steps`, `steps_completed`, `selected_step`, `stopped_early`, `records_per_step` (recorded or derived), `draws_per_epoch`, `epochs_at_selected`, `epochs_completed`, `tokens`, `tokens_per_second`, validation loss base/first/selected/last |
| capability | `evaluations[]`: purpose (or measurement only), task-set label, both arms' pass rates with Wilson intervals, wins/losses/p, difference, paired per-task bootstrap interval, uncounted share; `release`: promoted, champion, each candidate's gates |
| conduct | per arm and per attempt stage (`attempts[]`): failure classes, by side (interface, policy, infrastructure, unattributed), self-verification (undetected failures, disagreement), solvability of the task set |
| system | per attempt stage and arm: attempts per hour, uncounted by cause, completion and prompt tokens per second; training tokens per second; card use joined from an optional `nvidia-smi dmon -o DT` log |

Measured on the 4B loop (generations 1-3) and the 8B loop (generations 1-2): knowledge 244 (the stall) and 245
(undetected failures). Open:

- Supervised tokens per record: the records hold characters; brain's tokenizer count per record would need a
  `measure` call per dataset (the trainer seam has one) and a field in `composition.json`.
- The four undecided failure classes need records the run does not keep: which tool calls write and which read
  (a tool effect contract in sven), and the final state against the agent's account. Unverified successes wait on
  the same contract.
- The repeat rule (`failure_to_recover`) fires on 13% of the 8B champion's passes against 38% of its failures:
  a repeat whose result is the same is not always a failure to recover (a read twice). An error status per call
  (knowledge 226: it is in the request log, not the trajectory) would let the rule require an error first.

## Knobs (Part 2)

A recipe declares its knobs in a top-level `knobs:` section; each names the recipe parameter it sets and the state
variable (an observation path) it is meant to move. `splinter probe` perturbs one knob of a run's recipe, holds every
other parameter fixed, runs the perturbed recipe as a fork of the run (every stage upstream of the knob is reused,
so the cost is the stages downstream of it: keep the probe recipe a cheap slice), observes both, and reports
whether the variable moved and whether held-out capability followed, with intervals. A knob whose variable did not
move is reported as a defect. Not built yet.

| Knob | Moves | Measured |
|---|---|---|
| `model.train` `params.epochs` (or `steps`) | `training.epochs_at_selected` | 4B generation 2 retrained from 0.97 to 1.89 epochs at the selected step: the budget moved, capability did not follow (against the champion: exam 6/399 against 11/400, own held-out 53/260 against 57/264; the 0.97-epoch candidate had 13/400 and 66/264 against the same champion). |
| a frontier rule (which tasks' records to keep) | `data.frontier.frontier_share`, `data.own_share_records` | not yet a recipe parameter |
| difficulty steering (authoring) | `data.sources[].pass_rate` of the champion | not yet a recipe parameter |
| `model.train` `replay_share`, the teacher's share | `data.own_share_records`, `training.draws_per_epoch` | not measured |
