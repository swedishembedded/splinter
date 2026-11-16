<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# minimind: from random weights to a small agent

One command takes a tiny model from random weights to an agent that solves
elementary workspace tool-use tasks (read a value from a file, run a helper,
write the answer), and examines it on a sealed exam it never saw. The command
is idempotent and resumable: run it again after an abort and it continues;
run it again after it finished and it trains nothing.

The sample is thin on purpose. What it holds is what is specific to this
demonstration: the configuration and profiles, a deterministic synthetic
world, the training task families with their verifier, a programmatic expert,
the model's chat template, the stage list and the report.

Swedish Embedded AB implements end-to-end learning systems that take a model
from random weights to a working tool-using agent for its clients. If your
team needs expertise in training small agents from scratch, you can procure our
services by sending an email to info@swedishembedded.com. Everything else is
Splinter's: the stage ledger, sven episodes under the compact profile, the
recording endpoint, training records from recorded calls, brain over its API
and bench's examination.

## How the pieces connect

```text
 splinter-minimind (stages, ledger in the run directory)
   |                                   |                         |
   | brain/train jobs, install,        | sven agent (compact     | bench run / verdict
   | /v1/chat/completions              | profile) in a workspace |
   v                                   v                         v
 brain serve --training  <------  recording endpoint  <---  sven  ----> brain (exam arms)
   (one service, HTTP only)        (expert policy, or a proxy
                                    to brain for the model)
```

- brain is one `brain serve --training` process: given with `--brain-endpoint`,
  or started on a CUDA card for the run with `--brain-bin` and stopped when it ends.
  The sample reaches it only over HTTP.
- sven runs each episode against an OpenAI-compatible endpoint: the expert's
  for demonstrations, a recording proxy to brain for the model's own episodes
  and the probes. The requests sven sends are what the training records hold.
- bench runs sven itself for the exam, against the same brain, and returns one
  signed document of aggregates.

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
  any work. The brain process needs NVRTC findable, for example
  `export LD_LIBRARY_PATH=<cuda>/lib-with-libnvrtc:$LD_LIBRARY_PATH`, and the
  run directory must be on a disk with 10 GB free (brain's training floor).
- On a shared card, run the command under the card's lock, for example
  `flock <locks>/gpu3.lock samples/minimind/run.sh ...`.
- For the examination (optional): a bench checkout (`--bench DIR`), its private
  exam seed file (`--bench-seed-file`), the verdict signing key (`--bench-key`,
  made once with `./bench keygen --out KEY`) and the sven binary bench runs the
  agent with (`--sven-bin`). Without `--bench` the run says that it makes no
  promotion claim.

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
| 2 | `fetch/<name>` | only when configured: a pinned URL, checked against its SHA-256, cached, sliced | bytes and license |
| 3 | `world` | the synthetic world corpus from the seed: towns, people, expressions, tiny programs, conversations in the chat template | documents and bytes |
| 4 | `tasks` | training (`train`), exploration (`explore`) and development (`dev`) instances of three families, as an index and as Harbor task folders | counts |
| 5 | `demonstrate/<task>`, `demonstrate` | the expert drives real sven on every training instance; the verifier judges the workspace; accepted episodes become `data/demonstrate.jsonl`, one record per model call | each episode's answer and verdict, then accepted of total |
| 6 | `<model>/checkpoint` | brain trains a tokenizer on the corpus and makes a random-init checkpoint of the configured architecture | the installed model |
| 7 | `probe/<model>/init` | the development tasks through sven, and a raw-text continuation | `probe <model>/init: 0/4 development tasks solved`, the text, two example tasks |
| 8 | `<model>/pretrain`, probe | raw-text pretraining on the corpus, then the probe | steps, first and last loss; the probe |
| 9 | `<model>/sft`, probe | a full chat fine-tune on the demonstrations, then the probe | the same |
| 10 | `<model>/scratch/examine` | bench: the fine-tuned model against its random init | the verdict |
| 11 | `<model>/explore/<task>`, `<model>/explore` | the model itself as a sven agent on fresh instances, through a recording proxy; verified episodes become records | accepted of total |
| 12 | `<model>/improve`, probe | training on its own verified episodes (expert iteration), then the probe | the same |
| 13 | `<model>/experience/examine` | bench: the improved model against the fine-tuned one | the verdict |
| 14 | `report.md` | written from the ledger | its path |

Measured on the smoke profile, seed 7, one GH200 (gpu3), brain d84390ff5,
2026-11-16 (`gpu3` run, 32 minutes end to end without the exam):

```text
[  done] world: bytes=8501052 documents=13300
[  done] tasks: dev=12 explore=96 train=1600
[  done] demonstrate: accepted=1600 episodes=1600 records=5868
[  done] ouro/checkpoint: arch="ouro" model="local/minimind-ouro-init-7c69a3e524bf"
  probe ouro/init: 0/12 development tasks solved
    text  "Velmora is a town by" -> "oultethtethtethteth ... = = = ="
[  done] ouro/pretrain: first_loss=6.98 last_loss=0.34 seconds=687 steps=600
  probe ouro/pretrained: 1/12 development tasks solved
    text  "Velmora is a town by" -> " a forest. Its mayor is Trom Tromsoth. 568 people live in Vethmon"
[  done] ouro/sft: first_loss=1.43 last_loss=0.11 seconds=834 steps=4000
  probe ouro/sft: 3/12 development tasks solved
[  done] ouro/explore: accepted=13 episodes=96 records=40
[  done] ouro/improve: steps=1000
  probe ouro/improved: 4/12 development tasks solved
```

The fine-tuned model has the protocol (it reads, runs the helper, writes
`answer.txt`, answers); what it gets wrong is copying names and numbers out
of tool results. Pretraining ran at about 50k tokens a second (600 steps of
128 rows of 256 tokens); a sweep of pretraining batch on the card gave
27k, 56k, 69k, 75k and 77k tokens a second at batch 8, 32, 128, 512 and 2048,
so 128 is used. On the CPU the same pretraining ran at about 700 tokens a
second. The whole sequence is also exercised by the acceptance specs against
a model-service double (`cargo test --release -p splinter-minimind`).

## Abort, resume, re-run

Every stage is a unit of the run directory's ledger (`ledger.jsonl`), keyed by
what its result depends on and nothing else:

- its configuration slice and the run seed;
- the digests of the stage outputs it reads (a dataset, a checkpoint);
- its **recipe version**, a constant in `src/recipe.rs` for the sample code
  that decides its result (world, tasks, episode, gather, checkpoint, train,
  probe, examine), bumped in the change that alters that behaviour; a spec
  pins the generators' output, so a change without a bump fails it;
- the revisions of the **services it consumes**: sven's for every episode,
  the brain binary's digest for every stage brain computes or serves,
  bench's for the examination.

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
- **Resume** by running the same command again.
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
| `smoke` | hidden 256, 4 shared layers, 2 passes, vocab 1024, context 2048 | 13300 documents (6000 worked tool-use conversations), 1600/96/12 task instances | one GPU, about half an hour; the tests run it, smaller, against a double |
| `default` | hidden 256, 4 shared layers, 4 passes, vocab 4096, context 2048 | 60 towns and some 30k documents, 400/200/8 instances | a GPU; not run yet |
| `64m` | hidden 512, 8 shared layers, 4 passes, vocab 8192, context 2048 | 200 towns, 2000/1000/8 instances | a GPU; not run yet |

The architecture is configuration: `models.<label>` names an `arch` brain
knows and that architecture's own `config`, handed to brain unread. A second
label (for example a small `qwen3`) runs the same model stages on the same
data, budgets, probes and exam, beside the first, and the report compares them
stage by stage.

## Where outputs live

Everything is generated at run time into the run directory, never into the
repository: `manifest.json`, `ledger.jsonl`, `world/corpus.jsonl`,
`tasks/{train,explore,dev}.jsonl` and `tasks/harbor/`, `episodes/<unit>/`
(sven's trajectory and audit trail, the verdict, the records of an accepted
episode), `data/*.jsonl` (training records), `checkpoints/<model>/<stage>/`,
`exam/<model>/<pair>/verdict.json` and `report.md`. The smoke run's data stages
take well under a megabyte.

## The examination

With `--bench`, bench generates its sealed workspace tool-use exam and anchor
from the private seed file (the sample never opens it), runs both arms through
sven against the same brain (`bench run --endpoint`), and issues one signed
verdict of aggregates (`./bench verdict --min-gain X`). The arm profiles are
copies under the run directory; bench's jobs, which hold item-level records,
stay in bench's own data directory. The verdict is verified with
`./bench verify-verdict`; only a verified **Promote** promotes. Reject,
Inconclusive, Invalid or a document that does not verify promote nothing, and
the report says "not promoted".

## Limits

- The bench examination has not been run against a trained checkpoint yet:
  the measured run above makes no promotion claim.
- A rerun after a code change recomputes the data stages; their records hold
  the shell tool's measured duration, so the demonstrations' digest changes and
  fine-tuning is repeated. A rerun of unchanged code reuses every stage.
- The probes are a handful of development tasks of the training families: they
  show what a checkpoint does, they are not evidence of learning. Learning from
  scratch and learning from experience are reported apart, and a finished run
  is not a learning claim; only a verified Promote is.
- A tiny model may learn only part of the skill; the report says what was
  measured and nothing more.
- The tiny model learns the protocol and only part of the values; the improve
  stage trains on few verified episodes and fits them almost exactly.
