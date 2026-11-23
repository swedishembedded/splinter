<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# minimind: from random weights to a small agent

One command takes a tiny model from random weights to an agent that solves
elementary workspace tool-use tasks (read a value from a file in one of
several formats, follow a chain of lookups, run a helper, edit a key, write
the answer), and probes it on development and held-out tasks. The command
is idempotent and resumable: run it again after an abort and it continues;
run it again after it finished and it trains nothing.

The sample is thin on purpose. What it holds is what is specific to this
demonstration: the configuration and profiles, a deterministic synthetic
world, the task skills with their verifier, a programmatic expert, the data
checks and the admission rules,
the model's chat template, the stage list and the report.

Swedish Embedded AB implements end-to-end learning systems that take a model
from random weights to a working tool-using agent for its clients. If your
team needs expertise in training small agents from scratch, you can procure our
services by sending an email to info@swedishembedded.com. Everything else is
Splinter's: the stage ledger, sven episodes under the compact profile, the
recording endpoint, training records from recorded calls and brain over its API.

## How the pieces connect

```text
 splinter-minimind (stages, ledger in the run directory)
   |                                   |
   | brain/train jobs, install,        | sven agent (compact
   | /v1/chat/completions              | profile) in a workspace
   v                                   v
 brain serve --training  <------  recording endpoint  <---  sven
   (one service, HTTP only)        (expert policy, or a proxy
                                    to brain for the model)
```

- brain is one `brain serve --training` process: given with `--brain-endpoint`,
  or started on a CUDA card for the run with `--brain-bin` and stopped when it ends.
  The sample reaches it only over HTTP.
- sven runs each episode against an OpenAI-compatible endpoint: the expert's
  for demonstrations, a recording proxy to brain for the model's own episodes
  and the probes. The requests sven sends are what the training records hold.
- Every episode here runs under the conditions of a headless sven run: sven's compact profile, the
  task file's text (final newline included), a per-task time limit
  (`agent.deadline_secs`), announced as a headless run announces it (a `Time:`
  line ending the request and every tool result), no sampling in the probes'
  requests, so the model's declared sampling decides (greedy: the checkpoint
  is made with `do_sample: false`). The pretraining documents shaped as agent
  runs carry the same notice and list the tools in sven's order.

## Prerequisites

- This repository built with `make deps/local` against local sven and brain
  checkouts, or plainly against the pinned remotes. sven is linked through the
  SDK (its compact run profile, `compact/1`); nothing else of sven is needed to
  collect episodes.
- A brain service: either a running `brain serve --training` given with
  `--brain-endpoint http://host:port/v1`, or a `brain` binary given with
  `--brain-bin PATH --device gpuN`, which the run starts on that CUDA card
  (`--device gpuN,cpu --backend cuda`; the CPU only for making a checkpoint
  from nothing, which is host work) on a free port, with its model store and
  artifact store under `RUN_DIR/brain`, and stops when it ends. Training never
  runs on the CPU: a brain that does not schedule the card is refused before
  any work. `run.sh` finds NVRTC for it (the loader cache, the library path,
  beside `nvcc`, or `--nvrtc-dir DIR`) and puts only that library first on
  the brain process's library path; it puts brain's scratch (`TMPDIR`) in the
  run directory and refuses a disk with less than 10 GB free (brain's
  training floor), naming what is missing. The server dies with the run.
- On a shared card, run the command under the card's lock, for example
  `flock <locks>/gpu3.lock samples/minimind/run.sh ...`.
## The one command

```bash
samples/minimind/run.sh --run-dir ~/runs/minimind-smoke --brain-bin /path/to/brain --device gpu3
```

(or `--brain-endpoint URL --brain-artifact-store DIR` for a `brain serve
--training` that is already running).

`run.sh` builds the sample and runs `splinter-minimind run` with every option it
was given; `--profile` (default `smoke`) picks the profile and `--run-dir`
defaults to `target/minimind/<profile>`. The configuration is
`samples/minimind/minimind.yaml` (`--config FILE` for another): it overrides
the profile key by key and refuses a key it does not know. The run seed is
drawn and printed on a fresh run (`--seed N` sets it) and read back from the
run's manifest afterwards.

## Stages and what the terminal shows

| # | Unit | What it does | Shown after it |
|---|---|---|---|
| 1 | `init` | writes `manifest.json`: profile, seed, configuration, code revisions, template, licenses of fetched corpora | profile and seed |
| 2 | `fetch/<name>` | only when configured: a pinned URL, checked against its SHA-256, cached, sliced, split into documents on its separator | bytes and license |
| 3 | `world` | the synthetic world corpus from the seed: towns, people, expressions, tiny programs, conversations in the chat template, copy drills, and worked tool-use trajectories (the expert with the tools run for real); repeated documents over `data_checks.max_duplicate_share` fail the stage | documents, bytes, the duplicate share |
| 4 | `tasks` | training, exploration, development and held-out instances of five skills (`src/skills.rs`: read a value, follow a chain of two to four lookups, compute with a helper whose usage comes from its own `--help`, edit a key, count lines) over seven file formats with values that are numbers of one to six digits, words or letter-and-digit codes and data files at the root as often as in a directory (a training request names a file in a directory by its name alone a third of the time, so its place comes from the listing); names are made-up words, random letters, or, with a fetched corpus, its common words, requests drawn from a grammar of frames and synonyms; the held-out split uses two formats and frames no other split does; an index per split and the Harbor task folders | counts |
| 5 | `demonstrate/<task>`, `demonstrate` | the expert drives real sven on every training instance by one procedure (list the workspace, read, discover the helper, act, read an edited file back, answer); the verifier judges the workspace; an episode that passed but rewrote the task's files, looked for verifier material or answered before reading its sources is not admitted (`src/admission.rs`); admitted episodes become `data/demonstrate.jsonl`; one request template or final reply dominating fails the stage (`src/datacheck.rs`) | accepted, flagged, the data shares |
| 5b | `teach/<task>`, `teach`, `sft-data` | with `tasks.teach` above 0 and `--teacher DIR` (a checkpoint brain serves, linked into its model store): the teacher works its own split of training-family instances through sven under the same conditions, `teach.samples` times each; admitted episodes join the demonstrations in `data/sft.jsonl` (records too long for the chat block are left out and counted); without a teacher `data/sft.jsonl` is the demonstrations | accepted, records per source |
| 6 | `<model>/checkpoint` | brain learns a byte-level vocabulary on the corpus (digits split) and makes a random-init checkpoint of the configured architecture | the installed model |
| 7 | `probe/<model>/init`, `retrieval/<model>/init` | the probe: the development and held-out tasks through sven, a raw-text continuation; the retrieval probe: exact copies of 24 held-out values from context | passes by skill, development and held-out, how often the first reply is a tool call, a solved and a failed transcript |
| 8 | `<model>/pretrain-<i>`, `<model>/pretrain` | raw-text pretraining in `pretrain_segments` segments, each continuing the last; after each, the retrieval probe and, with `headroom_steps`, a fixed small fine-tune and the probe (the checkpoint's post-fine-tune headroom) | steps, losses, rate; the probes |
| 9 | `<model>/midtrain/data`, `<model>/midtrain` | mid-training: raw text with the demonstrations rendered in the template as `midtrain.agentic_share` of it | the same |
| 10 | `<model>/sft-<i>`, `<model>/sft` | the fine-tune on `data/sft.jsonl` (brain's `full_train`, only the replies supervised) in `sft_segments` segments, the probe after each: the points of the learning-aptitude fit | the same; at the end `s(N) = b + k log10(N/N0)`, `k` and `N*` |
| 11 | `<model>/explore/<task>`, `<model>/explore` | the model itself as a sven agent on fresh instances, through a recording proxy; admitted episodes become records | accepted, flagged |
| 12 | `<model>/improve/data`, `<model>/improve`, probe | its own admitted episodes mixed with `experience.rehearsal` demonstration records per own record, fine-tuned on (expert iteration with rehearsal), then the probe | the same |
| 13 | `report.md` | written from the ledger; the terminal gets the table across stages, the retrieval curve and the aptitude fit | its path and the table |

`chat_training.as_text: true` trains the fine-tune and the improvement on
whole conversations as text instead; whole trajectories as text belong in the
mid-training stage, and the fine-tune proper stays role-masked.

## Measured

Seed 7, one GH200 (gpu3), brain 16c2db09e, 2026-11-16. The smoke profile
(about 13 minutes to the improved checkpoint; the sample then also sat bench's
sealed exam, which it no longer does):

```text
$ samples/minimind/run.sh --run-dir RUN --seed SEED --brain-bin BRAIN --device gpu3
[  done] world: bytes=13208671 documents=33300
[  done] tasks: dev=12 explore=96 train=1600
[  done] demonstrate: accepted=1600 episodes=1600 records=5868
  probe ouro/sft: 2/12 development tasks solved (combine 0/4, lookup 0/4, count 2/4)
    PASS dev-00002-count  How many lines of log.txt contain ERROR? Write the number to answer.txt.
         shell({"shell_command": "grep -c ERROR log.txt || true"})
         write_file({"path": "answer.txt", "text": "3\n"})
         say "The answer is 30. It is written to answer.txt."
         answer.txt: "3"  (answer matches)
    fail dev-00000-combine  Run ./calc combine on the values stored in inputs/glais.txt and inputs/vaimgler.txt (first...
         read_file({"path": "inputs/kaim.txt"})
         write_file({"path": "answer.txt", "text": "412\n"})
         answer.txt: "412"  (answer.txt holds "412")
[  done] ouro/explore: accepted=15 episodes=96 records=45
[  done] ouro/improve/data: episodes=15 own_records=45 rehearsal_records=45 verified=15

ouro: what each checkpoint can do
  stage       probe   by family                          training
  init        0/12    combine 0/4, lookup 0/4, count 0/4
  pretrained  0/12    combine 0/4, lookup 0/4, count 0/4 loss 0.624
  sft         2/12    combine 0/4, lookup 0/4, count 2/4 loss 0.193
  improved    3/12    combine 0/4, lookup 0/4, count 3/4 loss 0.082
  retrieval while pretraining (listing+path exact): 0M tok: 0+0/24, 20M tok: 0+0/24
```

<!-- perf-number: the measured result of the smoke run above -->
Times of that run: world 29 s, tasks 17 s, pretraining 354 s at 55k tok/s,
<!-- perf-number: the measured result of the smoke run above -->
fine-tune 239 s, improvement 6 s.

The smoke model learns the protocol (read, run the helper, write
`answer.txt`, answer) and the task that needs no copying (count); it does not
copy a file name or a code out of its context, which every other task needs.
That ability appears abruptly with pretraining tokens: the same model
pretrained on copy drills alone copied none of 24 held-out values after 16,
33 and 49M tokens and 10 of 24 after 65M (the retrieval probe,
`src/retrieval.rs`). Smoke pretrains on 20M; the default profile pretrains
the same model on about 200M tokens of a world dense in copy drills, past
that point. Pretraining runs at about 55k tokens a second on this
model whatever the batch (64 x 512 to 256 x 1024).

## Abort, resume, re-run

Every stage is a unit of the run directory's ledger (`ledger.jsonl`), keyed by
what its result depends on and nothing else:

- its configuration slice and the run seed;
- the digests of the stage outputs it reads (a dataset, a checkpoint);
- its **recipe version**, a constant in `src/recipe.rs` for the sample code
  that decides its result (world, tasks, episode, gather, checkpoint, train,
  probe), bumped in the change that alters that behaviour; a spec
  pins the generators' output, so a change without a bump fails it;
- the revisions of the **services it consumes**: sven's for every episode,
  the brain binary's digest for every stage brain computes or serves.

This repository's revision is in no key: a change to the report, the
terminal output or a comment recomputes nothing (only the manifest, which
records every revision, is written again). A new sven reruns the episodes,
and because an episode's records are the same bytes for the same episode
(a shell result's duration, the server's tool-call ids and the attempt
number are made canonical), the datasets keep their digests and no training
run reads anything new. A new brain recomputes what brain computes or serves
and none of the data before it.

- **Abort** with Ctrl-C at any point. A unit that began and did not commit is
  resumed by the next run under a new attempt number; an interrupted episode's
  workspace is discarded and the episode runs again, counted once. A training
  run is resubmitted under the same job identity, so brain continues it from
  its last saved state.
- **Resume** by running the same command again. Measured (default profile of
  the time, 2026-11-16, brain 16c2db09e): Ctrl-C at step 100 of a 500-step
  pretraining segment, then the same command:

  ```text
           ouro/pretrain-1: step 50/500 loss 2.191
           ouro/pretrain-1: step 100/500 loss 1.298
  ^C
  $ (the same command)
  [reused] retrieval/ouro/init: listing=0 path=0 per_kind=12
           ouro/pretrain-1: step 150/500 loss 1.258
           ...
           ouro/pretrain-1: brain continued the run from its saved state at step 100
  [  done] ouro/pretrain-1: ... resume={"resumed_at":100,"state":"continued"} resumed=true steps=500
  ```

  <!-- perf-number: the measured result of the run above -->
  (The resumed segment took 566 s for its remaining 400 steps.)

  The managed brain server is a child in the run's process group, so Ctrl-C
  reaches both; a server whose run is killed alone gets SIGINT from the
  kernel and saves its job's state the same way.
- **Re-run** a finished run and every unit is reused after its outputs are
  hashed again; brain is not even started, no training and no episode is
  repeated, and the probes are shown from the ledger.
- A **changed setting** recomputes the units that read it and what follows them
  (a longer fine-tune retrains the fine-tune and what comes after, nothing
  before); a **changed seed** is a different run. A **corrupt output** (a
  flipped byte in a checkpoint) is reported and recomputed.
- `splinter-minimind status --run-dir DIR` prints what the ledger holds, the
  probes included, and computes nothing. One process holds a run directory at a
  time.

## Profiles

| Profile | Model (Ouro, looped) | Data | Where |
|---|---|---|---|
| `smoke` | hidden 256, 4 shared layers, 2 passes, vocabulary 1024 learned without digits, context 2048 (2.1M parameters) | 33300 documents (6000 worked tool-use conversations, 20000 copy drills), 1600/96/12 task instances; pretraining 600 x 64 x 512 (20M tokens), the chat stages as text | one card, about 13 minutes; the specs run it, smaller, against a double |
| `default` | the smoke model | 12000 worked conversations and 150000 copy drills, 3000/384/12 instances; pretraining 6000 x 64 x 512 (197M tokens) in six segments with the retrieval probe after each | one card, about an hour and a half |
| `64m` | hidden 512, 8 shared layers, 4 passes, vocabulary 8192, context 2048 | 200 towns, 2000/1000/8 instances | not run |

The architecture is configuration: `models.<label>` names an `arch` brain
knows and that architecture's own `config`, handed to brain unread. A second
label (for example a small `qwen3`) runs the same model stages on the same
data, budgets and probes, beside the first, and the report compares them
stage by stage.

## Where outputs live

Everything is generated at run time into the run directory, never into the
repository: `manifest.json`, `ledger.jsonl`, `world/corpus.jsonl`,
`tasks/{train,explore,dev}.jsonl`, `tasks/harbor.jsonl` (the Harbor task
folders), `episodes/<stage>.jsonl` (each episode's outcome, records, sven
trajectory and audit trail), `data/*.jsonl` (training records),
`checkpoints/<model>/<stage>/` and `report.md`.

A stage's episodes and task folders are thousands of small files, so they
are never made in the run directory: agent workspaces, episode files and task
folders are made in the run's scratch (`--scratch-dir`, default `/dev/shm`:
a private directory, mode 700, removed when the run ends; the run directory
holds a link to a running stage's tree). Each episode's kept files (outcome,
records, sven trajectory, audit trail) are appended to the stage's one
JSON-lines pack (`{"path", "mode", "text"}` per file) as soon as it finishes,
so a kill mid-stage leaves every finished episode there; once the stage
commits the pack is rewritten in path order. A reused stage reads its pack,
each file checked against the digest its episode committed; a stage that
must run more episodes unpacks it into scratch first. Shared file systems
bound the number of files, so a run directory stays at a few hundred.

## Limits

- The smoke model does not copy values out of its context; the default
  profile is sized to pretrain past the point where that appears, and its
  measured outcome is in the roadmap ledger.
- Brain's chat fine-tune (`full_train`) of a from-scratch Ouro made it ignore
  its context in every setting tried, so the chat stages train on the
  conversations as text by default; `chat_training.as_text: false` selects
  the chat fine-tune again.
- The probes are a handful of development tasks of the training families: they
  show what a checkpoint does, they are not evidence of learning. Learning from
  scratch and learning from experience are reported apart, and a finished run
  is not a learning claim; only a verified Promote is.
- A segment of pretraining restarts the learning-rate schedule; the segments
  exist to measure the curve.
