<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Build datasets and train a candidate

Datasets and training are stages of a recipe of the [learning factory](learning-factory.md): a
`dataset.project` stage projects attempts to a dataset, and a `model.train` stage trains a candidate on it. This page
describes how that training behaves: the parameters, the regimes, the evaluation and the replay. Datasets
are described in the [datasets reference](../reference/datasets.md); the [release
gate](../concepts/release-policy.md) decides whether a candidate is ever used.

```bash
splinter plan recipes/agent-sft.yaml
splinter run recipes/agent-sft.yaml --out runs/agent-sft
splinter inspect runs/agent-sft --stage lora
```

Training runs in the process unless the run has brain endpoints (`--brain-endpoint`): then a supervised
or policy-gradient run is a job of the brain service, which schedules it on the cards it serves from, and
the datasets and the adapter move through the artifact store `--brain-artifact-store` names. See
[Training on a brain endpoint](../getting-started/configuration.md#training-on-a-brain-endpoint).

## Parameters

A `model.train` stage names its `base` role, its `method` (`lora`), its `seed` and the trainer's own
`params`. For `lora` the `params` are checked before the run:

- `rank` is required, and so is a budget: `steps` (optimiser steps) or `epochs` (passes over the dataset),
  never both; there is no default budget.
- `epochs` (positive, fractional allowed) states the budget in passes, so it keeps its meaning when the
  dataset grows: the run trains `ceil(epochs * draws / records_per_step)` steps. The draws of one epoch are
  the dataset's training records plus the replayed ones (the plain union, each record seen `epochs` times);
  with a `replay_share` they are the dataset's records over `1 - replay_share`, so each record of the
  dataset is seen `epochs` times and the replay takes its share beside them. `records_per_step` is
  `grad_accum` times the data-parallel ranks the run spreads over (`--brain-train-cards`). The resolved
  `steps`, `epochs`, the record counts and `records_per_step` are part of the candidate's identity (the
  same epochs over other records is another candidate) and are written as `budget` into the stage's
  `report.json` and the training record `training.json`.
- `alpha` defaults to twice the rank, so the update's scale does not change with the rank.
- `grad_accum` is the records per optimiser step on each data-parallel rank and defaults to 1.
- `learning_rate` is brain's default when absent; `weight_decay` defaults to 0.
- `max_block` bounds the longest training row in tokens.
- `base_precision` is how the frozen base is held: `fp32` (brain's default), `bf16`, `int8`, `nf4` or `fp8`;
  a CPU backend trains `fp32` only.
- `eval_every` is the steps between evaluations of the validation records. Its default 0 evaluates nothing
  and exports the last step. With a value above 0 and validation records, the run evaluates every
  `eval_every` steps and exports the step with the best validation loss.
- `patience` is the evaluations without improvement before the run stops. Its default 0 runs the whole
  budget; the best evaluation is exported either way.
- For preference datasets: `loss` (`dpo`, `ipo`, `simpo` or `orpo`; `dpo` when absent), `beta`,
  `nll_weight` (the weight of the chosen answer's likelihood beside the pair loss, 0 when absent) and
  `gamma` (SimPO's target margin).
- For rollout datasets: `behaviour`, where the importance ratio's behaviour log-probabilities come from
  (`recorded`, the default, or `start`: the policy the run starts from, exact only when the rollouts were
  sampled from it).

The stage records every parameter as the trainer resolved it, defaults written in, so a change of a default
is a change of the stage's identity. Brain's own settings that the stage does not name are left at brain's
defaults.

## Regimes and what a candidate continues

`model.train` trains the regime the dataset's objective names, and one stage's datasets are all of one
regime:

- `sft` and `cpt` datasets by supervised fine-tuning.
- `preference` datasets by a pair loss against the model the run starts from (DPO by default).
- `feedback` datasets by KTO on every judged completion, with class weights from the balance of desirable and
  undesirable records.
- `policy-gradient` and `distill` datasets by brain's clipped policy gradient, below.

A stage trains a fresh adapter on the base unless it names `start:`, a champion role (see
[the learning factory](learning-factory.md)) served from the same base, whose adapter a preference, feedback or
policy-gradient run continues: a preference run polishes the policy that made its pairs. A stage with no
`start:` does not continue any earlier release. The validation records are what the held-out scores and the
step choice are read over; a dataset with none is scored on its training records, which the candidate's
report records as `held_out`.

## Training on rollouts

A `policy-gradient` dataset trains with a clipped policy gradient, the way GRPO and PPO do:
brain turns each group's rewards into advantages by the normalisation the dataset was built
for (which its manifest records), a record with a negative advantage lowers the probability of its action, one with a
positive advantage raises it, and a KL penalty keeps the adapter near a reference policy.
Nothing is imitated. Brain trains it on `qwen35` bases only and refuses any other, and refuses
a dataset with rewards under supervised training, so neither can silently drop them. The
run replays and rehearses nothing: a `replay` input is refused with it. Of brain's policy-gradient settings
the stage names only `behaviour`; the rest are brain's defaults.

The candidate's report carries brain's per-step statistics under `statistics`: the loss, clip fraction, KL and
importance ratio of each step. The KL costs one forward pass over the dataset before the first step, none per step.

### Self-distillation on the cliff

GRPO's advantage is zero where every attempt at a task failed, so the hardest tasks teach it
nothing. Self-distillation gives those attempts dense credit instead, on the student's own
distribution. A dataset projected with `objective: distill` makes every attempt of every group,
passed or failed, a distilled record (see the [learning
factory](learning-factory.md#what-each-stage-guarantees)); a `model.train` stage trains it with the
policy-gradient job.

A distilled record is the attempt's own conversation, exactly as the student saw it, plus a
`teacher_context`: a turn right after the instruction that only the teacher sees, built from what
the verifier recorded (the hidden checks that failed and the tests seen to fail, the verifier's own
statement of why, the task's reference change when the task carries one). It never enters the
student's messages. An attempt with nothing of that to show is left out and counted as
`no_privileged_context`. Brain scores the student's own sampled tokens with the frozen model shown
that context and trains each token with `A_t = log q_teacher(y_t | privileged context, y_<t) - log p_student(y_t)`
through the same clipped step as any advantage (reverse KL, the dense signal). The teacher's
rendering is longer than the student's, so the block size counts the longer of the two.

## Evaluation and the step carried

A supervised run evaluates the validation records only when `eval_every` is above 0. The candidate then
carries the adapter of the evaluation with the lowest validation loss, not the last step's, and the run stops
once that loss has gone `patience` evaluations without improving (`patience: 0` runs the whole budget and
still carries the best). With `eval_every: 0` nothing is evaluated and the last step is carried. The
candidate's report holds the base and tuned scores and the training curve (`curve`). The validation records
come from the dataset's own `validation` split, so no step is chosen on training records.

## Interrupting a run

A brain training job keeps running when the client that submitted it exits, and keeps its card.
So Ctrl-C (or SIGTERM) on `splinter run`, or on any command whose run submits a training job,
cancels the job through brain's API (`DELETE /v1/jobs/{id}`, which stops it at its next optimiser
step) before the command exits, and the error it ends with says which job was cancelled and
whether brain confirmed it; one brain did not confirm is named with how to find it
(`GET /v1/jobs`) and stop it. The run is recorded as cancelled. A second Ctrl-C exits at once.
A killed run continues where it stopped (`splinter resume`), and a training job continues from its last saved state
(its resume key).

## Replay and rehearsal

A `model.train` stage may name a `replay` input: datasets of earlier chat records mixed into the
training draws. With `replay_share` (between 0 and 1) that share of the draws comes from the replay records;
without it the replay and the new records are one plain union. Replay is for `sft` and `cpt` datasets only: a
preference, feedback or rollout dataset is refused with a `replay` input, and so is a `replay_share` with
nothing replayed. The stage's `seed` seeds the adapter's initialisation, the batch order and the replay draw;
the same seed writes the same adapter on the same data, and two runs that differ only in the seed show how
much of a measured difference is the draw.

## Seeds

Every random choice a pipeline makes takes a seed the run states. A recipe writes down the seed of every stage. The
exam's families (`exam-set create --seed`) are kept with the families, so every later run reads the seed back and
naming another is refused; the exam's bootstrap intervals draw a seed when none is given, write it to the run record (`seeds`) and keep
it in the artifact it decides. The release gate's bootstrap intervals (the tasks' own evidence, the anchor's drop) are
resampled from a seed of their own, recorded as `gate-interval` and kept in the gate report and so in the release
manifest. A local model that samples draws a seed for each request that names none; the run that made the requests
records them in order (`runs show` lists the count, the first sixteen and a digest of all of them, and the whole list
is kept in the run's events), and a seed passed back as the request's replays its decoding. Greedy decoding draws
nothing. Passing the seed again reproduces the choice. Two runs reserve different families unless they share a seed, so
results are comparable across runs only when each names a recorded one.
