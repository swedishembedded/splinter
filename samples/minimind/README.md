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
  or started on the CPU for the run with `--brain-bin` and stopped when it ends.
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
  `--brain-bin PATH --models-dir DIR`, which the run starts on the CPU
  (`--device cpu --backend cpu`) on a free port, with its state under
  `RUN_DIR/brain`, and stops when it ends.
- For the examination (optional): a bench checkout (`--bench DIR`), its private
  exam seed file (`--bench-seed-file`), the verdict signing key (`--bench-key`,
  made once with `./bench keygen --out KEY`) and the sven binary bench runs the
  agent with (`--sven-bin`). Without `--bench` the run says that it makes no
  promotion claim.

## The one command

```bash
samples/minimind/run.sh --run-dir ~/runs/minimind-smoke --brain-endpoint http://127.0.0.1:8811/v1
```

(or `--brain-bin PATH --models-dir DIR` for a brain the run starts and owns).

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

Stages 1 to 5 need no model. Today, against brain, the run does them and then
stops at stage 6 with the capability brain does not offer yet (measured on the
smoke profile, CPU):

```text
[  done] init: profile="smoke" seed=7
[  done] world: bytes=22602 documents=344
[  done] tasks: dev=4 explore=8 train=8
[  done] demonstrate/train-00000-combine: answer="735" conclusion="Success" passed=true tool_calls=4
...
[  done] demonstrate: accepted=8 episodes=8 records=30
[failed] ouro: ouro/checkpoint (attempt 1): needs brain's from-scratch action over brain/train: ...
```

The whole sequence, every stage above, is exercised by the acceptance specs
against a model-service double (`cargo test --release -p splinter-minimind`).

## Abort, resume, re-run

Every stage is a unit of the run directory's ledger (`ledger.jsonl`), keyed by
the content address of its configuration slice, the run seed, the digests of
what it reads and the code revisions it depends on.

- **Abort** with Ctrl-C at any point. A unit that began and did not commit is
  resumed by the next run under a new attempt number; an interrupted episode's
  workspace is discarded and the episode runs again, counted once. A training
  run is resubmitted under the same job identity, so brain continues it.
- **Resume** by running the same command again.
- **Re-run** a finished run and every unit is reused after its outputs are
  hashed again; no training and no episode is repeated, and the probes are
  shown from the ledger.
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
| `smoke` | hidden 64, 2 shared layers, 2 passes, vocab 1024, context 1024 | 344 documents, 8/8/4 task instances | the CPU, minutes; what the tests and CI run |
| `default` | hidden 256, 4 shared layers, 4 passes, vocab 4096, context 2048 | 60 towns and some 30k documents, 400/200/8 instances | sized for a fast turnaround |
| `64m` | hidden 512, 8 shared layers, 4 passes, vocab 8192, context 2048 | 200 towns, 2000/1000/8 instances | a GPU |

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

- Brain does not yet offer, over its API, the from-scratch checkpoint (tokenizer
  plus random init), raw-text pretraining or a full chat fine-tune of an Ouro
  checkpoint; the run stops at the checkpoint stage and names what it needs.
- The probes are a handful of development tasks of the training families: they
  show what a checkpoint does, they are not evidence of learning. Learning from
  scratch and learning from experience are reported apart, and a finished run
  is not a learning claim; only a verified Promote is.
- A tiny model may learn only part of the skill; the report says what was
  measured and nothing more.
- Episodes run one at a time.
