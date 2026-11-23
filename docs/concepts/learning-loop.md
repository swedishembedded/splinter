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
5. **Training** continues the current release with a LoRA adapter, replaying earlier material.
   See [Build datasets and train](../guides/training.md).
6. **Evaluation and release** decide whether the candidate becomes the policy. See [Release
   policy](release-policy.md).

## Why questions name their subject


Every question Splinter writes names what it is about - the product,
document, tool or version - so that someone who has never seen the source
gets exactly one answer. A question whose answer would differ for another
product is refused, and so are two tasks that give one question different
answers.

## Where the training budget goes

Budget goes where learning happens: before a task's attempts become
training data, `learn` measures the policy's pass@k on it closed-book.
A task it never solves is solved once more by a teacher - the same
policy unless `--teacher` names another model - shown the source
sections the task is grounded in; the teacher's answer, once the task's
verifiers pass it, is what the policy learns a new fact from. `learn`
keeps the tasks the policy fails at least sometimes and has a verified
answer to - its own or the teacher's - and drops those it always solves
(nothing to learn) and those nobody answered verifiably (nothing to learn
from). The training records are the student's: the instruction alone and
the verified answer, never the passage the teacher saw. The training set
is capped per concept, task kind and verification strength; `splinter
status` lists the concepts the policy has mastered least, from its
closed-book solves alone; and concepts the release gate sees forgotten are
queued for new tasks.

## One prompt for asking and training

What the policy is trained on is what it sees when it answers: every
solve, `ask`, judge, critique and gate probe runs under one short system
prompt of Splinter's own, and every training record starts with that same
system turn.

## The curriculum

`solve --frontier` measures the solver's pass@k on every task: k solves
(default 4), each an experience numbered by its attempt, graded by the
task's own verifiers. Each task no graded attempt solved is then solved
once by a teacher - the solver, or the model `--teacher` names -
open-book: its prompt shows, before the instruction, the task's
grounding material (the source section each evidence span falls in, its
passages and hints; never its reference). The teacher's solve is an
experience of the task like any other, graded by the same verifiers, its
provenance marked `teacher`; a task grounded in nothing a teacher could
be shown is skipped. A task is worth training on when the solver fails it
at least sometimes and a verified answer exists: on the frontier (some
attempts passed) or taught (none passed, the teacher's did). A task
every graded attempt solved carries no signal, and one with no verified
answer nothing to learn from: both are dropped, as is one with no graded
attempt (it has no rate; unmeasured is never 0). The tasks kept are
written as a task set and an experience set of their attempts and
verified teacher answers. The measurement is recorded under
`<state>/curriculum/measurements/`, per task: its kind, concepts,
attempts, graded attempts, passes, rate, the teacher's attempts, graded
attempts and passes, and class (`always`, `frontier`, `taught`, `never`,
`unmeasured`), with the solver, the teacher, the release `policy:<alias>`
was, k and the sampling. The teacher's solves are training data, never a
measurement of the policy: they count toward neither a task's rate nor
concept mastery. A local model samples
its attempts at temperature 0.8 unless `--temperature`/`--top-k` say
otherwise; a model whose sampling cannot be set (a remote one) samples as
its server does, and naming a sampling for it is refused.

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
