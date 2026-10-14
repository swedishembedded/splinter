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

The loop takes a task, a repository and the checks that decide success. It
is the supervision around the shared stages, and owns nothing they own:

| Stage | Shared implementation the loop calls |
|---|---|
| the repository at the baseline | `capture_tree`: the tracked files of the revision, a snapshot in Splinter's artifact store |
| the worker | `solve_in_workspace`: sven's file, search and shell tools in a fresh copy of the snapshot, the shell in the sandbox's container when there is one, the change read back from git as a patch, the failures of the checks the worker is shown put back as follow-up turns |
| the judgement | `grade` (the workspace verifier's own): the snapshot laid out again, the patch applied, the protected paths fenced, the checks run in the sandbox with their process-group kill |
| what the worker saw | the exposure audit `practice` runs over every trajectory |
| applying an accepted patch | the sandbox's `apply`, all or nothing |

What the loop adds is the supervision: the contract, the append-only trace,
checkpoints, cancel and resume from any process, retries with feedback across
attempts, the limits, and the outcome. Everything a reviewer needs is kept
under `~/.sven/loop/runs/<run>/`.

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

The workspace must be a committed git repository; it is never edited, and a
repository whose tracked files differ from `HEAD` is refused. No tree outlives
an attempt: the candidate is `patch.diff` of the run, and `apply` applies the
patch of an accepted run to the repository, which must still be at the
baseline with no uncommitted change to a tracked file.

## Where learning went

The loop supervises one task; it does not train. Training from agent work is
`splinter practice` (and the verbs it is made of), which solves tasks through
the same worker, grades them with hidden material, writes the `sft-step`
view of what its own checks confirmed, trains and releases through the gate:

| The loop used to | Now |
|---|---|
| `dataset`: records from accepted runs | `splinter dataset build` with the `sft-step` view, from the experiences `practice` verified |
| `train`: an adapter on those records | `splinter train` (and the training stage of `practice`) |
| `models list`, `judge`, `current`, `rollback` | the release store: `splinter release` runs the gate over the exam `practice` reserved and moves the alias; `splinter rollback` moves it back |

`fixtures/families.py` renders the twelve seeded-bug families; they are
imported as workspace tasks for `practice`, and `fixtures/eval-candidate.sh`
runs a model reference over a list of them as plain loop runs.

## What a run keeps (`~/.sven/loop/runs/<run>/`)

| File | What |
|---|---|
| `contract.json` | the task, baseline revision, checks, protected paths, limits, model, system prompt digest |
| `events.jsonl` | the append-only event stream, a sven hash-chained log (`events.jsonl.lock` is its writer lock): schema version, run, attempt, ever-growing id, timestamp, parent, type, data |
| `artifacts/<sha256>` | payloads too large for an event, stored once by content |
| `trajectories/attempt-N.atif.json` | sven's own trajectory of each attempt |
| `patch-N.diff`, `patch.diff` | the candidate of each attempt, and the last, exactly as judged (the trace redacts credential-shaped text; a patch is code to be applied and is not rewritten) |
| `checkpoint.json` | the last durable point; written after every attempt |
| `outcome.json` | the structured result |

Event types: `run_started`, `model_selected`, `definitions`, `workspace_prepared`,
`baseline_validation`, `attempt_started`, `tool_request`, `tool_result`,
`file_mutation`, `assistant_message`, `plan_update`, `model_usage`,
`follow_up`, `candidate`, `validation`, `exposure`, `attempt_finished`,
`provider_retry`, `provider_error`, `limit_reached`, `events_dropped`,
`reconcile`, `run_resumed`, `definition_changed`, `run_finished`. An event over 4096 bytes keeps a content
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
| `--follow-ups` | 2 | rounds an attempt goes on in the same conversation, told what failed, after the worker stops with the checks it is shown (`--accept-visible`) red |
| `--max-cost-usd` | none | a remote model's reported cost; an unpriced call is counted, never treated as free |

## Honest accounting

* `--hint` text goes to the worker and marks the run **assisted**; an
  unaided result is one with no hint, no remote model and no supervisor
  repair during the attempt.
* Acceptance checks are the supervisor's: they run in a copy of the snapshot
  the worker never had, and their scripts live wherever the supervisor keeps
  them. A hidden check (`--accept`) is named to the worker and nothing more:
  not its command, and not what it printed, neither in the prompt nor in the
  feedback of a failed attempt, nor in the follow-up turns inside one (those
  show the output of the visible checks only). A visible one
  (`--accept-visible`) is shown with its command and its output. Files named
  by `--protect` may not change; a candidate that touches one is rejected
  even if every check passes. A check is bounded by the sandbox's wall-time
  limit for a workspace command, not by a per-check setting.
* The shell is not a sandbox, so the worker is kept from the answer by what
  is not in its tree, and the trajectory is audited for any sight of it: an
  attempt whose calls or outputs name a hidden check's command or script, or
  the directories that hold the loop's records (every contract names its
  checks) or Splinter's state, is rejected as `hidden_material_exposed`
  whatever its checks said, and an `exposure` event says where. Where the
  workspace runs in a container the audit detects and the container prevents;
  on the host it detects.
* A resumed run has no partial state to trust: no tree outlives an attempt, so
  it records a `reconcile` event and goes on with the next attempt. A run
  started under an older contract schema is not resumed.
* The tools that reach the network or another model are withheld from the
  worker.
* File mutations during an attempt are attributed by reading the episode's
  tree back after each tool call and can lag by one call; the patch read back
  from git is the authority. sven's own tool audit log lived in the worker's
  tree and goes with it: the trajectory and the `tool_request` and
  `tool_result` events are the record.

## Layout

```text
Cargo.toml        depends on splinter-sdk and nothing else of Splinter's
prompts/          the system prompt (an override in ~/.sven/loop/system.md wins)
fixtures/         make-fixture.sh: the disposable seeded-bug repository; families.py: twelve seeded-bug
                  families rendered under a seed; eval-candidate.sh: one model over a task list
src/              contract, trace, observe, workspace, attempt, run, cli
tests/            the plumbing specs, with a scripted model
```
