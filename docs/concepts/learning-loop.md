<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# The learning loop

Splinter turns a request to learn something into a recorded run of stages. Each stage
reads what the previous one stored and writes content-addressed objects under the state root,
so any stage can be rerun or inspected alone.

```text
   task --> experience --> evidence --> dataset --> training --> evaluation --> release
```

1. **Tasks** are written from the sources you point it at: a question or a job, with a
   reference answer and the verifiers that grade an answer. See [Tasks](../reference/tasks.md).
2. **Experience** is an agent's run of a task, recorded whole and never edited. See
   [Experience](experience.md).
3. **Evidence** is a verdict about an experience from a verifier the policy cannot influence.
   See [Verification](verification.md).
4. **Datasets** are projected from the experiences worth training on, for one objective. See the
   [datasets reference](../reference/datasets.md).
5. **Training** fits a LoRA adapter on a dataset, optionally continuing a champion and replaying earlier material.
   See [Build datasets and train](../guides/training.md).
6. **Evaluation and release** decide whether the candidate becomes the policy, in a recipe that has
   evaluation and promotion stages. See [Release policy](release-policy.md).

## Why questions name their subject


Every question Splinter writes names what it is about - the product,
document, tool or version - so that someone who has never seen the source
gets exactly one answer. A question whose answer would differ for another
product is refused, and so are two tasks that give one question different
answers.

## Where the training budget goes

Budget goes where learning happens, and the recipe says where. An `attempt.collect` or `answer.collect`
stage runs the policy at each task (`attempts: k` makes k graded attempts, each an experience numbered by its
attempt) and grades every attempt by the task's own verifiers. A `dataset.project` stage then projects
attempts to a dataset: an `sft` dataset admits an attempt only when it was counted and passed at full credit
with no veto, and writes the student's record, the instruction alone and the verified answer. A task whose
attempts all failed contributes nothing to an `sft` dataset, and an attempt stage can be pointed at the tasks
an earlier one left unsolved (`unsolved_by`) to try them again.

For reinforcement learning, a `dataset.rlvr` stage selects the tasks worth rolling out again: those whose
measured pass rate lies strictly between 0 and 1, or inside the `min_pass_rate` and `max_pass_rate` the stage
names, since a task the policy always or never solves gives its groups the same reward and no gradient.
`splinter status` lists the concepts the policy has mastered least, from its closed-book attempts alone.

## One prompt for asking and training

What the policy is trained on is what it sees when it answers: every
attempt, `ask`, judge and critique runs under one short system
prompt of Splinter's own, and every training record starts with that same
system turn.

## Learning a person's way of thinking

The same pipeline, planned from a directory of someone's writing: see [Learn how a person
thinks](../guides/personas.md). Learning a repository from its own tests is the same loop with
an agent working in copies of the tree: see [Learn a repository](../guides/repositories.md).

## Rollouts over a served policy

A learning run that keeps going (many generations, each one a cycle of rollout, train, gate and account) is a forked
recipe run of the learning factory: a generation is a run that forks the previous one and names the previous
generation's model as its `champion:`. That workflow, its recipes and its commands are described in
`docs/guides/learning-factory.md`. What this page describes is where such a run's training episodes come from and
what Splinter keeps of them.

**Where rollouts come from.** Training episodes come through one of these paths and no other:

- the episode path: practice task folders in the Harbor format (instruction, starting files, tests that write a
  reward) run as workspace episodes through sven's agent against the served policy (`tasks harbor episodes`, and
  `episodes::SvenEpisodes` as the episode executor; see [Harbor tasks and trials](../guides/harbor-tasks.md)).
  Each attempt is graded by the task's own tests outside the agent's reach and kept as an experience whose trajectory
  holds the policy trace (the prompt and generated token ids and their log-probabilities as the serving endpoint
  returned them). An attempt with no reward (the infrastructure or the verifier failed) enters no group, and one whose
  trace is not provable is never trained on, never re-tokenized.
- the served path: a model brain serves, sampled at temperature 1 against a verifier the policy cannot reach.

**Where its tasks come from.** A run's tasks may be environments from the environment factory ([Environments
from an objective](../guides/environment-factory.md)). Only an `Admitted` environment whose recorded base pass rate
lies inside the learnable band it was admitted at may enter (`served::factory::select` refuses the rest with the
reason); its task folders are staged per attempt and removed. A host that feeds a run from the factory labels each
task with its environment (`served::EnvironmentLabels`) so that the weakest verifier class forces its label on the
verdict (`proxy only` is never lost on the way to a "better" model), and so that each environment's training reward and
dev pass rate can go to the factory's divergence alarm: a reward that climbs while the dev measure does not is reward
hacking or memorisation, and that environment is quarantined.

Training data is never taken by running a benchmark's own runner on its collections. A benchmark is the sealed
measurement authority: it holds the exams, and the proposer must not be able to see or influence them. Bench is asked
for verdicts on finished candidates; a run's own gate is a dev-validation gate on tasks of its own, and it does not move
a release alias.

The bench-mediated flow (a benchmark run exported and admitted with `experiences ingest-bench`, see
[Benchmark trials as training data](../guides/bench-trials.md)) is **legacy** for new work: it remains for the
recorded arms and for evidence from benchmarks that cannot be re-run, and it trains on the stored actions with a
proxy for the behaviour policy (`--behaviour start`) because a benchmark export carries no sampler log-probabilities.
