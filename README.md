<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

![Splinter banner](docs/banner.jpg)

# Splinter

**Splinter turns agent experience into better local models.**

It runs agents against real tasks, records what happened, verifies the
outcomes, extracts training data from what was verified, trains a local
model, and releases the result only when held-out evidence shows it improved.
You tell it what to learn, in plain language:

```
splinter "Learn all the facts about the STM32 reference manual from <path-to-manual>"
splinter "Learn what we can do with brain command line options"
splinter "Explain brain command line flags without interacting with brain"
```

One system carries the whole loop:

```text
   task --> experience --> evidence --> dataset --> training --> evaluation --> release
    |           |             |            |            |            |             |
 generated   an agent's    verified by   projected   a LoRA on    held-out,     an immutable
 from what   run, kept     code a model  from the    the local    retention,    adapter with a
 you point   whole and     cannot reach  experience  base, with   anchor and    manifest, on
 it at       never edited                            replay       serve checks  plain brain
```

## What it is built on

Splinter is built on two standalone projects and replaces neither:

- **sven** runs the agent: tools, sessions, delegation, and a trustworthy
  record of what the agent did.
- **brain** runs the model: inference, training, evaluation and serving.

Splinter decides what is learned: which tasks to write, which experience
qualifies as training data, what to train, and whether the result is better.
What it releases is an ordinary brain adapter with a manifest. It runs on plain
`brain serve`, and plain sven can use it through its brain provider; neither
needs Splinter installed.

## What it learns from

| Source | How |
|---|---|
| A document | chunked into addressable sections; facts and held-out probes are extracted and held to the text by code |
| A directory of text | surveyed, then a planner model chooses what to teach from a menu; each choice is held to the sources |
| A command-line tool | the tool's real output is captured as a source with provenance, then goes through the document path |
| An agent's own attempts | solved closed-book or in a runtime, graded by the task kind's verifiers, critiqued and retried |

## What it produces

| Product | What it is |
|---|---|
| Experience | an agent's recorded run, immutable and content-addressed; verdicts, rewards and labels are facts derived about it |
| Datasets | training records projected from experience for one objective, with a manifest naming where every record came from |
| Candidates | a LoRA adapter trained from the champion on the new material plus a replay of everything learned before |
| Releases | an immutable adapter or full checkpoint with a manifest of every number the release gate measured, the terms it was made under, and lineage back to the sources |

## Current status

| Capability | Status |
|---|---|
| Learn from documents, repositories and command output | working |
| Closed-book solving, verification, critique and retry | working |
| Pass@k frontier selection with a teacher for what the policy never solves | working |
| Supervised fine-tuning with replay, preference fine-tuning by DPO | working |
| Four-check release gate, immutable releases, rollback | working |
| Lineage from an answer back to the source bytes | working |
| A predictive release gate for non-language models (`release_predictive`): the champion and candidate are scored on the same held-out units (a different unit set is refused); performance, calibration, retention on named subgroups, serving correctness and data-policy compliance are decided from pre-registered requirements over numbers the caller measured, and an unmeasured number fails | working |
| Releases of a full checkpoint (brain owns its format; Splinter keeps the immutable file, its digest, the architecture name and the brain and splinter commits) beside adapter releases, with the same aliases, compare-and-set moves and rollback; manifests of earlier formats still load | working |
| Usage policy on every source (`redistributable`, `research_only`, `noncommercial`, `restricted_DUA`, `unknown`) carried to the dataset, training run and release; an unrestricted release is refused unless every axis is allowed, `unknown` never is, and a restricted release records its terms (`source add --usage-policy`, `release --unrestricted`) | working |
| Models by role (policy, teacher, generator, planner, judge, critic, router) | working |
| Several tasks in flight for a model reached over an API | working |
| The Rust SDK (`splinter-sdk`) | working, young: the API will move |
| Longitudinal record files (one participant per line, `timeline-v1` fields plus interventions that say whether they were randomised) imported as one immutable content-addressed episode per participant on the participant's own clock and the calendar, every item tied to its file line, file digest and usage terms, raw identifiers replaced by opaque keyed keys; importing again adds nothing | working |
| Splits of record datasets by participant group, by a temporal cutoff and by holding one whole source out, each a pure function of a seed with a content address, and leakage gates that fail with counts, never identities: a group in two parts, a held-out source not held out whole, a unit on the wrong side of the cutoff, statistics fitted on anything but training units | working |
| A participant-safe `timeline-v1` dataset projected from longitudinal episodes at a prediction point: inputs only from at or before it, outcomes with observation windows (left truncation and right censoring kept), randomised and observational interventions as distinct inputs, opaque participant and group keys, a manifest naming the episodes, source file digests and combined usage terms, and the parts of a split written only after the group gate passes | working |
| A timeline risk model closed end to end: the stored parts of one split train a candidate (hazard knots derived from the training outcome times, the fit certified on training units only, units carried from the record file into the model's vocabulary so brain refuses another unit at prediction); its risks are calibrated with brain's Venn-Abers calibration on the half of the validation part early stopping does not read (never a test unit), and the calibration is packed beside the weights with the training support into one file whose unpack loads in plain brain; the candidate and a champion are scored on the same held-out units with brain's own evaluation (Uno C, time-dependent AUC, IPCW and integrated Brier, calibration, held-out likelihood) and splinter's participant-clustered paired bootstrap differences, under requirements registered before scoring, calibration judged on the calibrated risk where the model has one and on the raw risk otherwise (the record says which; a horizon brain declares uncalibrated has nothing to judge and fails as unmeasured); serving correctness is measured from the shipped file unpacked by the system `tar` and loaded by plain brain, on the test units (identity of every prediction, batched against single patient-history forecasts, the share of units the support would withhold, and the validity of every probability and curve); it is released only if the predictive gate passes, else rejected and recorded with the alias unmoved; the release names its calibration by digest and the lineage runs from the release to the source file line of every training participant. `samples/health` runs it on a synthetic cohort and forecasts a patient history before and after an appended checkup | working |
| Training from rewarded trajectories, raw text, contrastive pairs | planned: projected as an export format; no trainer reads it yet |
| A `chat` intent at the front door | planned |
| Resuming a pipeline from a checkpoint | planned |

## Quick start

```bash
splinter eval --suite anchor --freeze general.jsonl   # once: the anchor suite
splinter learn docs/p100-manual.md --goal "the P100 console and power limits"
splinter status
splinter "what baud rate does the P100 console run at?"
```

The command line (`crates/splinter/README.md`) runs the pipeline one stage per
verb - sources, tasks, solve, verify, critique, dataset, train, release - and
`learn` runs them all as one recorded run.

## The Rust SDK

Everything the command line does, a program can do through one dependency:

```rust
use splinter_sdk::learn::LearnRequest;
use splinter_sdk::Splinter;

let splinter = Splinter::from_env()?;
let learned = splinter.learn(&LearnRequest {
    sources: vec!["./manual.md".into()],
    goal: Some("the console and power limits".into()),
    ..LearnRequest::default()
})?;
```

A `Splinter` owns what a process shares - the runtime with its loaded models,
the trainer, where progress is reported - and hands each command a context of
its own, so a policy alias that moves between two commands is seen by the
second. Every command that writes state runs as a recorded run
(`Splinter::run`), readable and cancellable from any process. The command line
and every program in `samples/` depend on the SDK and on nothing else of
Splinter's.

## How the loop works

Every question Splinter writes names what it is about - the product,
document, tool or version - so that someone who has never seen the source
gets exactly one answer. A question whose answer would differ for another
product is refused, and so are two tasks that give one question different
answers.

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

What the policy is trained on is what it sees when it answers: every
solve, `ask`, judge, critique and gate probe runs under one short system
prompt of Splinter's own, and every training record starts with that same
system turn.

Learning a person's way of thinking is the same pipeline, planned: point
Splinter at a directory of what they wrote and it surveys it, a planner model
chooses what to teach from a menu (recall, the person's own advice to a
correspondent, explanation, restoration of corrupted passages), and each
choice is held to the sources by code. A task teaches what the writer
advised only if its answer is a passage of the writer's own text, word for
word, and an answer is graded by checking that every passage it quotes is in
the source and that it gives the advice; no model decides that. A strong
teacher answers open-book and the policy is trained on those verified answers
(`--distill`). `samples/jefferson` measures what that teaches a model
about Thomas Jefferson's letters, on letters it never saw.

Every stage stores what it makes under the state root by content address,
so any stage can be rerun or inspected alone (`splinter runs show`,
`splinter experiences show --graph`). Experience, sources, tasks, runs, and the metadata of datasets,
candidates and releases live in one experience database that also records where
a release came from, back to the attempts it learned from; adapters and datasets
stay plain files beside it, tracked by it. `splinter state maintain` keeps the
database small, and `splinter state archive`, `verify` and `repair` pack the whole state,
find what has gone missing and recover it. Every model runs locally unless a
`remote:` model is named with `--allow-remote`.

Local models share the device: the policy, a candidate and the champion
it is measured against are one base with different LoRA adapters, so one
copy of the base is resident and each generation runs with its own
model's adapter attached. Training and the release gate's `brain serve`
check load their own copy, so every resident base is released before
either starts and loaded again by the next model use.

A trained candidate continues the current release (the champion) - by
supervised fine-tuning with a replay of what earlier releases learned, or
by DPO on pairs preferring a verified answer over a failed one - and
becomes the policy only if the release gate measures that it improved on
the new material's held-out questions - the facts it was trained on asked
in other words, which `learn` has the generator model write for each task
kept and which are never trained on - kept what earlier releases
learned, held a frozen anchor suite of general tasks, and runs on plain
`brain serve` with the same answers (the same text to within the numerical
noise of two processes decoding one model, or the same meaning in other
words). A held-out question
the candidate was trained on (the same question, or a near duplicate,
among its training records) is left out of those measurements and
counted as leaked. An executable check passes only when it is seen to run to its
end, so a solution that exits before its check cannot pass, and
`splinter experiences replay` re-runs an experience's code calls in its
recorded environment to confirm what it observed.
Each release is an immutable adapter with a manifest of every number the
gate measured; `splinter rollback default` returns to the previous one.

Every artifact traces both ways: `splinter lineage <ID>` walks from an
answer, a release, a dataset or any other stored artifact up to where it
came from - down to the source, part and byte range each task is grounded
in, with the bytes themselves - and down to everything that came from it.

## Building

sven and brain are git dependencies at the revisions pinned in `Cargo.lock`:

```bash
make build     # release build of every package
make test      # every test; no model, GPU or network needed
make check     # all gates: text hygiene, headers, fmt, clippy -D warnings, lock sources
```

To build against local checkouts instead - to change sven or brain and
Splinter together - write the gitignored local override once:

```bash
make local SVEN_DIR=<sven-checkout> BRAIN_DIR=<brain-checkout>
```

Builds then compile the checkouts' working trees, offline, without touching
`Cargo.toml` or `Cargo.lock`. `make lock` re-pins the lock to those
checkouts' HEADs; push those commits before sharing the lock. Delete
`.cargo/config.toml` to go back to the remotes.

The policy is Qwen3-0.6B from brain's model store by default
(`BRAIN_QWEN_WEIGHTS` names another checkpoint). One 24 GB GPU holds its
base for serving and trains LoRA adapters on it at agent-trajectory
lengths - tens of thousands of tokens per record - which is what learning
from agent experience needs. Qwen3-1.7B also trains on one such GPU, at
shorter records; larger bases need more memory than one card has.

Runtime state (sources, tasks, experiences, datasets, candidates,
releases, answers, runs) lives under `~/.sven/splinter`, Sven's home in a namespace
of its own (override: `--state DIR` or `SPLINTER_STATE`).

## License

Apache-2.0. See `LICENSE`.
