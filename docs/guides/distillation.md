<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Distil a teacher's verified trajectories

Supervised fine-tuning of a small student on the trajectories a large teacher passed is the simplest recipe that
generalises here. The examiner grades every trial, only the passes are kept, and the student learns each of the
teacher's actions in the context the teacher saw. You can start from a published bundle of teacher trajectories,
or produce your own.

**Reference result.** Qwen3-4B fine-tuned on 420 packed trajectories of Qwen3.8-27B-FP8 (550 practice trials on the
train side of four collections, 445 of them passed) was examined by the examiner on sealed task families it never
trained on: pass rate 0.042 for the base and 0.425 for the student, a gain of +0.383 [+0.298, +0.469] (326 items,
54 families, three attempts each), with retention non-inferior at a margin of 0.10. The verdict was Promote. The signed
aggregate verdicts are in the bundle below. A gain on another teacher, student or task distribution has to be measured
again; this one shows that the recipe can work.

## Who does what

| Project | Owns |
|---|---|
| bench (the examiner) | the task collections and their train/eval split, the practice material, every grade, the leak audit, the export of graded trials, the sealed exams and their signed verdicts |
| sven | running the agent in each trial and writing its trajectory (ATIF) and its model requests |
| brain | serving the teacher and the student, reading datasets with its own trainer, training the adapter |
| splinter | admitting graded trials as experiences, curating the dataset, submitting the training job, recording the candidate with its provenance, and promoting it on a verdict |

Splinter never grades a trial and never sees exam items. A trial reaches a dataset only as the examiner exported it,
and only when it is on the train side.

## From published trajectories

### 1. Ingest the bundle

```bash
splinter experiences ingest-hf swedishembedded/experience-to-competence-evidence@splinter-b0f65b12fffa370c40fa \
    --token-file ~/.cache/huggingface/token
```

The bundle is a private Hugging Face dataset repository written by [`splinter publish`](../reference/storage.md#bundles-on-the-hugging-face-hub).
Name it by its tag or commit, never by a branch you expect to stay put. The tag above resolves to commit
`01dc4011237cf347a7c5d23c636d9149ac08038c`, and every file is checked against the bundle manifest before anything is
stored. The command prints the experience set (the teacher's 550 graded trials), the dataset, and the candidate with
its adapter file:

```text
experience set blake3:c2ff30619a140af905508a660c55afd7cd960e737e1e284a2d5d131779a2904e
dataset blake3:e2eb1efb27e7c3d8e144448a4dad3b7c9806e8c680c2a838aae2eca2620d5388
candidate candidate-20261118T042734.740-8514: adapter $STATE/artifacts/76/76242730f4201f735173d0803fd77696011b976e04ccafd0973f09a8f68d9b19.safetensors
experiences: 550 new, 0 held; datasets: 1 new, 0 held; candidates: 1; artifacts: 1 new; verdicts: 5
```

Ingesting it again stores nothing new.

### 2. Build the dataset

```bash
splinter --brain-bin "$BRAIN" --device cpu --backend cpu dataset build --view sft-step --pack \
    --max-block 16384 --from local:Qwen/Qwen3-4B blake3:c2ff3061
```

`sft-step --pack` makes one record per passing trajectory. Every assistant turn (its reasoning, its tool calls and
its answer) is supervised, and the system prompt, the instruction and the tool results are context. Failed trials,
repeated actions and records longer than `--max-block` are left out and counted. `--from` names the base the
records are for: that base's trainer reads every record (brain's own check, through `--brain-bin` or
`--brain-endpoint`), counts its tokens the way training will, and a record it refuses (its chat template or its
record contract) is left out and counted as `trainer_refused`, so training never meets it. With `--from` and no brain
to ask, the build is refused. Without `--from`, a build with no brain counts by a floor, says so, and can keep a
record that the trainer later refuses. Checking needs no card.

`--repository-history` sets what happens to the teacher's reads of git history (see the pitfalls below). The
default, `drop-step`, keeps such a read as context and does not supervise it. `truncate` ends the demonstration
before the read. `keep` supervises it, as the reference dataset did. On this set, `drop-step` leaves 14 actions
unsupervised and keeps 420 records (`blake3:2d505ae1...`). To rebuild the reference dataset itself, pass
`--repository-history keep`. The result (`blake3:0efa3688...`, the same id in any state root) then holds the same
420 records with the same exclusions (105 failed, 7 repeated actions, 25 over the block) and, as brain's trainer
reads it, the same 2,894,518 tokens with 1,544,541 supervised. Its id is not the
bundle's, for two reasons
([knowledge 193](../../.agents/knowledge/193-a-rebuilt-distillation-dataset-has-the-same-records-but-not-the-same-id.md)).
The records now carry the seeded, stratified held-out split
([the held-out split](../reference/datasets.md#held-out-and-monitoring-split)). Also, a long listing's `agent` owner
is kept rather than renamed `user`, because `agent` is the account the student meets again in the examiner's sandbox.

### 3. Check it with the trainer

```bash
splinter --brain-bin "$BRAIN" --device cpu --backend cpu dataset check "$DATASET" \
    --from local:Qwen/Qwen3-4B --max-block 16384
```

brain's own trainer reads every record for that base, in the mode a run would use (reasoning on, one example per
packed record). It reports the tokens and supervised tokens per example, the records that supervise nothing, and the
longest records. The command exits 1 if the trainer refuses any record. Fix the dataset before you claim a card.

### 4. Train

```bash
splinter --brain-endpoint http://127.0.0.1:8811/v1 --brain-key-file brain.key \
    --brain-artifact-store /srv/brain-artifacts \
    train "$DATASET" --from local:Qwen/Qwen3-4B --base-precision bf16 --max-block 16384 \
    --rank 16 --alpha 32 --lr 2e-4 --weight-decay 0 --steps 210 --records-per-step 4 \
    --eval-every 14 --patience 4 --seed 1
```

These are the reference run's settings, read from its candidate record:

- LoRA rank 16 and alpha 32 on every adapter site of the model (all seven Qwen3 projections, brain's default);
- a peak learning rate of 2e-4 and no weight decay;
- 210 steps of 4 records each, with the base held in bf16;
- the monitoring set evaluated every 14 steps, and the best evaluation kept;
- seeds 1, 2 and 3, one run each (the examined candidate is seed 1). Without `--seed`, a seed is drawn and printed.

Name `--records-per-step`, because a run that names `--steps` otherwise takes one record per step. The job runs on the
brain service, which you start with `--training --artifact-store` set to the directory you pass as
`--brain-artifact-store` ([training on a brain endpoint](../getting-started/configuration.md#training-on-a-brain-endpoint)):

```text
brain --device gpu0 --backend cuda serve --openai 127.0.0.1:8811 --api-key-file brain.key \
      --models-dir ~/models --training --artifact-store /srv/brain-artifacts --serve-adapters /srv/adapters
```

The candidate records:

- the held-out loss of the base and of the adapter;
- the training curve and the step it kept;
- the seed;
- the splinter binary that trained it, with its digest;
- brain's training record.

The split now draws held-out and monitoring records per domain rather than from the tail of the file, so a retrain
lands within the seed band, not on the reference adapter bit for bit. The reference run's three seeds came out within
0.002 of each other in held-out loss (1.528 for the base, about 0.567 for each adapter). How training is scheduled,
interrupted and resumed is in [Build datasets and train a candidate](training.md).

### 5. Serve the adapter and measure it

A brain started with `--serve-adapters DIR` serves every adapter file in `DIR` as a model of its own and lists it in
`GET /v1/models`. To measure the candidate and the base on the same tasks and attempts, run a recipe of the learning
factory (`docs/guides/learning-factory.md`) that rolls out both models over the same collection, for example a forked
run whose `champion:` is the base and whose model is the candidate (`$ADAPTER` is the file `train` or `ingest-hf`
printed; `--brain-adapter-dir` writes it into the directory the server follows). Each run admits its graded trials as an
experience set, and `experiences compare` pairs two sets task by task and counts each family once:

```bash
splinter experiences compare "$CANDIDATE_SET" "$BASE_SET"
```

Rollouts over the train side measure fit, not generalisation. For a development measurement, run the eval side of a
public collection through a recipe's development plan, with the candidate served as a model of its own.

### 6. The sealed exam (examiner side)

A claim that the student generalises needs the examiner's sealed exam, which the proposer never runs or reads. The
examiner:

1. serves the base and the candidate on one brain build;
2. runs both on the sealed families under an exam plan committed before the runs;
3. issues a signed verdict of aggregates only (`bench verdict --plan ... --signing-key ...`, or `bench examine`; see
   bench's `docs/verdicts.md` and `docs/examination.md`).

What comes back is the verdict file and the examiner's public key, never items or trials. Check the signature with
`bench verify-verdict verdict.json --public-key KEY.pub`. A bundle carries the verdict with the candidate
(`splinter publish --verdict verdict.json --examiner-key KEY.pub`), and `ingest-hf` verifies it again. A sealed exam is
spent by its first verdict, so draw a new exam for the next decision.

## From scratch: your own teacher trajectories

### 1. Draw practice tasks

The examiner owns the train side of a collection, and a recipe names the collection a run draws its tasks from
(`docs/guides/learning-factory.md`). Only the train side is ever drawn for practice; the eval side and sealed exams are
never rolled out for training.

### 2. Run the teacher

Serve the teacher on a brain endpoint and run a recipe whose rollout model is the teacher. sven is the agent in every
trial, the task's own tests grade it, and the graded trials are admitted as experience. If the examiner graded the
trials itself instead, ingest its export with `splinter experiences ingest-bench EXPORT`. Each trial gets a typed
outcome, and an infrastructure failure is never a reward of 0. The details are in [Benchmark trials as training
data](bench-trials.md).

### 3. Keep the verified passes only

- The dataset admits only trials the examiner passed. A failed, ungraded or infrastructure-failed trial yields no
  record.
- A trial the examiner's leak audit flagged (a security violation, such as reading history beyond `HEAD`) is never
  exported, so it never becomes an experience. The run's report counts such a trial as `exposed`.
- An experience of an eval-side or sealed task is refused by every dataset build.
- Reads of history that the audit allows can still teach probing, so choose `--repository-history` deliberately (see
  below).

Then continue at step 2 of the first section, naming every run's experience set in one
`dataset build`. To share the result, `splinter publish` writes the sets, the dataset, the candidate and its verdicts as
one tagged revision of a private repository.

## Pitfalls

- **Git-history probing.** A student imitates the teacher's exploration and generalises it past the audit's line. Seven
  allowed `git log` and `git show` reads taught a student `git log --all`, `git reflog` and reads of `.git/refs`
  ([knowledge 170](../../.agents/knowledge/170-a-student-learned-to-dig-through-git-history-from-seven-teacher-log-reads.md)).
  Dropping the supervision of those reads did not lower the probing
  ([knowledge 179](../../.agents/knowledge/179-dropping-supervised-history-reads-did-not-stop-a-student-probing-git-history.md)).
  Measure the examiner's security-violation rate on the trained arm, and use `truncate` when it has to fall.
- **Serve/train parity.** A record must render, token for token, the way the served model is prompted: the same chat
  template, the same system prompt and tools, the same JSON spelling of arguments. The `sft-step` view takes the system
  prompt and tool schemas from the trial's first recorded model request for this reason
  ([knowledge 003](../../.agents/knowledge/003-training-and-evaluation-must-render-the-same-prompt.md),
  [knowledge 085](../../.agents/knowledge/085-a-dataset-from-the-27b-teacher-taught-the-host-login-the-scratch-path-and-the-loop-tail-under-a-prompt-it-was-not-served-with.md),
  [knowledge 159](../../.agents/knowledge/159-the-pretraining-text-spelled-tool-call-arguments-compactly-while-the-served-template-spaced-them.md)).
  `dataset check` with the base you train is the test.
- **Compare within one serving build.** A brain build changed what a served Qwen3 prompt carries: an injected
  reasoning instruction that the trainer never renders was later removed. So pass rates measured on two builds differ
  for reasons that have nothing to do with the adapter. Measure the base and every candidate on the same build. The
  examiner records its sha256 in the verdict.
- **A frozen brain binary per run.** Copy the brain build a run uses, record its commit and digest, and never switch it
  mid-run. The candidate records the splinter binary that trained it. Rebuilding brain under a running server changes
  what serves the next request.
- **Pin the bundle's revision.** `ingest-hf` records the commit it read. Name a tag or a commit in scripts and papers,
  so a later publish to the same repository cannot change what your run ingests.
