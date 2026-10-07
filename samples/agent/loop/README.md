<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->
<!--
Swedish Embedded AB implements self-improving coding agents whose every
step is auditable, for its clients. If your team needs expertise in agent
evaluation or locally operated coding agents, you can procure our services
by sending an email to info@swedishembedded.com.
-->

# agent-loop - can a local model complete a repository task unaided, and can a supervisor audit it?

**Status:** the plumbing is specified and passes; coding ability on a real
model is measured per task and reported in the paper kept beside this
repository, never inferred from the fixture.

The loop takes a task, a repository and the checks that decide success. A
model works in an isolated git checkout with sven's own file, search and
shell tools; the loop reads back from git what changed, judges the
candidate with the supervisor's checks, and on failure retries from a clean
checkout with diagnostic feedback. Everything a reviewer needs is kept under
`~/.sven/loop/runs/<run>/`.

Models are **local by default** (`local:Qwen/Qwen3-8B`, in-process through brain). A
model reached over an API is refused unless `--allow-api-models` is given;
`remote:brain/<model>` names a `brain serve` on this machine (the way the 27B
model is reached) and is recorded as `local_served`, unpriced. Nothing here
pretends that a locally trained adapter changes a remote model.

## Run it

```sh
# 1. a disposable repository with a seeded bug, and an acceptance script
#    that lives outside it
samples/agent/loop/fixtures/make-fixture.sh /tmp/fixture

# 2. the loop (the GPU is needed for a local model)
agent-loop run \
  --workspace /tmp/fixture/repo \
  --task "The function textstats.top_words does not behave as its docstring says in some cases. Find the bug in the library code and fix it." \
  --accept "hidden=python3 /tmp/fixture/accept.py" \
  --accept-visible "repo-tests=python3 -m unittest discover -s tests" \
  --protect tests/ \
  --model "local:Qwen/Qwen3-8B@32768" \
  --max-attempts 2

# 3. look, stop, continue, apply
agent-loop status <run>
agent-loop cancel <run>
agent-loop resume <run>
agent-loop apply  <run>
agent-loop runs
```

`run` exits 0 when a candidate was accepted, 3 when cancelled and 1
otherwise; `--json` prints the structured outcome.

The workspace must be a committed git repository; it is never edited. The
candidate lives in `work/` of the run; `apply` applies the patch of an
accepted run to the repository, which must still be clean at the baseline.

## Learning from accepted runs

Only a run that was **accepted, unaided and on a local model** offers
training records, and only its accepted attempt: a rejected patch teaches a
ranker, not a generator, and a hinted or remote-model run is not evidence of
what the local model can do alone. `dataset` writes the records in brain's
chat format with a manifest saying what became of every run offered.
`train` fits a LoRA adapter on them (the trainer, optimiser and adapter
format are brain's) and registers it as a **candidate**; a candidate is never
the model in use. `models judge` decides by a rule fixed in the code before
any result: the same tasks run once with the model in use and once with the
candidate (directories of `run --json` outputs, same file names), at least
eight paired tasks, and a one-sided sign test over the tasks only one of the
two solved. A rejected candidate stays on record with the numbers that
rejected it. Promotion keeps what it replaced, so `rollback` returns to it.

```sh
agent-loop dataset --outcomes runs-json/ --out train.jsonl
agent-loop train   --dataset train.jsonl --steps 60 --rank 16
agent-loop models list
agent-loop models judge --candidate <version> --baseline-results base-json/ --candidate-results cand-json/
agent-loop models current      # the model reference to run with now
agent-loop models rollback
```

## What a run keeps (`~/.sven/loop/runs/<run>/`)

| File | What |
|---|---|
| `contract.json` | the task, baseline revision, checks, protected paths, limits, model, system prompt digest |
| `events.jsonl` | the append-only event stream, a sven hash-chained log (`events.jsonl.lock` is its writer lock): schema version, run, attempt, ever-growing id, timestamp, parent, type, data |
| `artifacts/<sha256>` | payloads too large for an event, stored once by content |
| `trajectories/attempt-N.atif.json` | sven's own trajectory of each attempt |
| `patch-N.diff`, `patch.diff` | the candidate of each attempt, and the last |
| `checkpoint.json` | the last durable point; written after every attempt |
| `outcome.json` | the structured result |
| `work/` | the isolated checkout |

Event types: `run_started`, `model_selected`, `definitions`, `workspace_prepared`,
`baseline_validation`, `attempt_started`, `tool_request`, `tool_result`,
`file_mutation`, `assistant_message`, `plan_update`, `model_usage`,
`candidate`, `validation`, `attempt_finished`, `provider_retry`,
`provider_error`, `limit_reached`, `events_dropped`, `reconcile`,
`run_resumed`, `definition_changed`, `run_finished`. An event over 4096 bytes keeps a content
address, its size and a preview; credentials are redacted before anything is
written. A gap in what could be observed is an `events_dropped` event.

Splinter records each run too (`runs list`, cross-process cancel); the loop
adds the directory above.

## Sampling of a served model

`--temperature` (0 to 2) sets the sampling temperature asked of a model
reached through sven's providers, a model served by brain on this machine
included; unset keeps the provider's default, which for a served model can be
greedy decoding. A greedy 27B model working as an agent repeated one probe
command until the attempt was stopped, again and again, and wrote nothing;
the same task at a temperature of 0.7 edited files within minutes. A served
model is also asked for no reasoning block unless `--thinking` is given. Both
settings are written in the `model_selected` event of every run.

## Limits (all recorded in the contract)

| Flag | Default | Stops |
|---|---|---|
| `--attempt-secs` | 900 | one attempt |
| `--total-secs` | 3600 | the whole run |
| `--max-output-tokens` | 24000 | one attempt's model output |
| `--max-tool-calls` | 60 | one attempt's tool requests |
| `--max-repeats` | 4 | one call (same tool, same arguments) returning the same answer this many times: the attempt is stopped as `stagnation` and the next one is told which call looped; a repeated call whose answer changes (a test run after an edit) is not counted; 0 turns it off |
| `--max-attempts` | 3 | the attempts |
| `--provider-retries` | 2 | a model call that failed in transit (growing pause) |
| `--follow-ups` | 2 | rounds an attempt goes on in the same conversation, told what failed, after the worker stops with the checks red |
| `--max-cost-usd` | none | a remote model's reported cost; an unpriced call is counted, never treated as free |

## Honest accounting

* `--hint` text goes to the worker and marks the run **assisted**; an
  unaided result is one with no hint, no remote model and no supervisor
  repair during the attempt.
* Acceptance checks are the supervisor's: they live outside the checkout. A
  hidden check (`--accept`) is named to the worker and nothing more, so its
  script is not offered for reading; a visible one (`--accept-visible`) is
  shown with its command. The shell is not a sandbox, so this keeps the
  worker from being invited to read the answer, not from being able to.
  Files named by `--protect` may not change; a candidate that touches one is
  rejected even if every check passes.
* A resumed run does not trust the checkout an interrupted attempt left: it
  resets to the baseline and records a `reconcile` event. Nothing is
  replayed on top of unknown state.
* The checkout is isolation from accidents, not a sandbox: the shell tool
  runs as the user. The tools that reach the network or another model are
  withheld from the worker.
* File mutations during an attempt are attributed by comparing the checkout
  after each tool call and can lag by one call; the patch read back from git
  is the authority.

## Layout

```text
Cargo.toml        depends on splinter-sdk and nothing internal else
prompts/          the system prompt (an override in ~/.sven/loop/system.md wins)
fixtures/         make-fixture.sh: the disposable seeded-bug repository
src/              contract, trace, observe, repo, acceptance, attempt, run, cli
tests/            the plumbing specs, with a scripted model
```
