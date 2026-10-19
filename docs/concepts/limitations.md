<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Limitations

What Splinter does not do, and what its results do not show. Each statement is backed by the
program's behavior or by the project's own samples; where a limit has a flag or check, it is named.

## Model size and hardware

- The default policy is Qwen3-0.6B from brain's model store. One 24 GB GPU holds its base for
  serving and trains LoRA adapters on it at agent-trajectory lengths. Qwen3-1.7B also trains on one
  such GPU, at shorter records; larger bases need more memory than one card has. `--bf16-base`
  holds the frozen base at bf16, which is what a 7B base needs to train on one 24 GiB card.
- A full consolidation (`consolidate --mode full`) trains every weight and keeps the optimizer's state
  in system memory; the cards hold the weights and their gradients, about 4.4 GiB for Qwen3-0.6B and
  61 GiB for Qwen3-8B, which fits no two 24 GiB cards. `--mode merge` folds the adapter into the weights
  on the host at any size and trains nothing.
- If no card has room for the policy's weights, brain runs the model on the CPU backend without a
  word. That is correct and too slow to use for real runs; `splinter doctor` warns about it. A
  model brain must place on the CPU backend timed out in the first-run example.
- The serve check, and so any release, serves an adapter on a Qwen3 base: brain serves it as the
  model `brain/qwen3` and refuses to start when the base's digest is not the one the adapter records.
- Linux is assumed for repository tasks: the kernel confinement needs Landlock ABI 4, and a kernel
  that cannot confine makes the command refuse to run, unless you pass `--allow-unconfined-workspace`.

## Evaluation contamination

- A held-out result is only as clean as the separation of its data. Splinter reserves exam families
  before anything is generated, leaves their text out of every source the run reads, and checks each
  dataset for them; it refuses a family a candidate's lineage was trained on. It cannot know what
  the base model saw in pretraining. The `synthetic-author` sample exists because a model may have met
  a historical author in pretraining; it measures against ground truth that cannot have been seen.
- The exam measures access to what the weights never saw, not recall of it, when retrieval is
  used: the sources hold the held-out text too.
- Sealed families are recognised by a repository's tree id: a copy of a sealed family with one file
  changed has another tree id and is not recognised.
- An exam of a few families cannot show a gain. Use `exam-set power` to size an exam before paying for
  it; a result on an underpowered or not-frozen-in-advance evaluation is a pilot.

## Cost

- A run needs a model to write tasks, a teacher for what the policy never solves and, for kinds
  only a judge can decide, a judge. With nothing configured the 0.6B policy plays all of them, and
  in the first-run example that was not enough to write usable questions or to solve a fill-in task
  (see [your first learning run](../getting-started/first-learning-run.md)). Naming stronger models
  costs GPU memory and time, or money and a network opt-in for a `remote:` model.
- `--budget` bounds a run's wall-clock time, and generation stops early to leave time to train. A
  run over a few percent of a large corpus is reported as such, with what covering the rest would
  take; it is not a run over the corpus.
- Local models share the device. A task is failed after `--solve-deadline` seconds on the wall
  clock, so raise it when many tasks share one served model and a task would otherwise fail for
  waiting in the queue.

## Determinism

- Splinter draws, prints and records the seed of every random choice (exam families, rehearsal
  tasks, bootstrap intervals, training). Passing a recorded seed again reproduces the choice, and
  results are comparable across runs only when each names a recorded one.
- Two processes decoding one model need not give identical text: summation order changes the
  last digits. The serve check therefore accepts the same text or the same meaning.
- A local model that samples draws a seed per request; the run records them, and decoding with the
  seed replays. Greedy decoding draws nothing. Splinter does not claim bit-identical training
  across different hardware or library versions.

## Verification strength

- A verifier either establishes that an answer is right or only refutes it. Grounding (every number,
  name and quotation is in the source) refutes what invents and decides nothing about what is right.
- A judge's verdicts count only if its calibration on controls shows it precise enough for the use. The
  controls are made from the tasks' own references; an agreement of a judge with a person is measured
  only when a person supplies labels (`exam-report labels-import`).
- The claim gate matches the person's words by terms: the first word of a sentence is not taken for a
  name and numbers written in words are not recognised, so a statement may carry them without the person's
  words having said them. The secret filter recognises shapes; it is not proof that nothing sensitive remains.
- Voice and correctness are scored separately; neither the likelihood of the writer's text nor
  nearest-passage similarity establishes authorship.

## External teachers and judges

A `remote:` model is refused unless `--allow-remote` or `SPLINTER_ALLOW_REMOTE=1` is given; that is the
only way Splinter uses the network. A teacher's answers are training data and never a measurement of the
policy. A strong teacher raises what the policy can learn and also makes the result depend on a model
Splinter does not control. A judge must be another model than the teacher, the generator and the policy.

## Security of agent-generated code

Splinter runs code that a small model wrote. Where no container is configured (`SPLINTER_DOCKER` and a
digest-pinned `SPLINTER_WORKSPACE_IMAGE`), the commands an agent works with run on the host,
bounded but not isolated, and every environment record says `isolation: none`.

On the host the kernel confines a command to read-only system access, its own tree and a private directory,
with no network, a private user and pid namespace and no secret-named variables in its environment. A kernel
that cannot do this makes the command refuse to run. That is confinement by the kernel, not a container's
boundary: use a container for code you do not trust. The exposure audit is detection, not prevention. Details
are in [Learn a repository](../guides/repositories.md#confinement).

## Controlled demonstrations versus general capability

Every sample is a controlled measurement of one question and states what it measured. None shows that the loop
makes a model generally better, and several state that they have not (the `tool-syntax` sample has no before and
after number yet, and the first-run example reaches no trained candidate on its single page of material).
A run can train a candidate that the gate then blocks; that is a result, not a failure of the tool.
Learning a fact is not learning a skill, and a policy trained on a person's writing is not that person.
