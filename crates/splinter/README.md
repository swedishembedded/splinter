<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# splinter - the command line

One verb per pipeline stage. The thing a verb acts on is a positional
argument; every stage writes what it makes under the state root by content
address, so any stage can be rerun or inspected alone; `--json` works on
every command. `splinter <command> --help` is the authoritative reference.

```
splinter                          REPL on the current policy (a line is handled exactly like `splinter "<line>"`)
splinter "<sentence>"             the front door: a sentence becomes one of the commands below
splinter learn <SOURCE>... [--goal TEXT] [--kinds K,..] [--budget DUR] [--dry-run]
splinter ask <QUESTION> [--open-book SOURCE-ID] [--policy REF]
splinter status
splinter source add <PATH|cmd:COMMAND...> | list | show <ID>
splinter tasks generate <SOURCE-ID>... --kinds K,.. [--generator REF] | list | show <ID>
splinter solve <TASKSET-ID> [--solver REF]
splinter verify <EXPERIENCE-SET> [--judge REF]
splinter critique <EXPERIENCE-SET> [--critic REF] [--retry N]
splinter judge calibrate <LABELLED-FILE> --judge REF
splinter experiences list | show <ID> [--graph]
splinter dataset build <EXPERIENCE-SET>... --view VIEW [--strip all|keep:K,..|mix:F]
                       [--min-strength executable|formal|consistency|judged] [--export-only]
splinter dataset export <DATASET-ID> --out DIR
splinter train <DATASET-ID>... [--from REF] [--replay DATASET-ID] [--steps N] [--rank R]
splinter runs list | show <ID> | cancel <ID>

global: --state DIR  --json  -v  --allow-remote
```

An id is the full `sha256:<hex>`, the hex alone, or a prefix of at least
four hex digits that names exactly one stored object.

## Models

Every model is named the same way:

| Reference | Model |
|---|---|
| `policy:default` | the configured base (`BRAIN_QWEN_WEIGHTS`, else `Qwen/Qwen3-0.6B` in brain's model store); the champion adapter joins it once releases exist |
| `local:<checkpoint>[+<adapter>]` | a checkpoint path (absolute, or starting `./` or `../`) or a name in brain's model store, with an optional LoRA adapter file after the first `+` |
| `remote:<provider>/<name>` | a model reached over the network through sven's provider configuration |

A remote reference is refused unless `--allow-remote` is given or
`SPLINTER_ALLOW_REMOTE=1` is set; that is the only way Splinter uses the
network. OpenRouter models take their key from `AGENT_OPENROUTER_KEY`, any
other provider from `BRAIN_API_KEY`.

## The pipeline

| Stage | Command | Reads | Writes |
|---|---|---|---|
| sources | `source add` | a file, a directory, a command's run | a source |
| tasks | `tasks generate` | sources | a task set |
| solve | `solve` | a task set | an experience set |
| verify | `verify` | an experience set | verdicts on its experiences |
| critique | `critique` | an experience set | an experience set of critiques and revisions |
| dataset | `dataset build` | experience sets | a dataset and its manifest |
| train | `train` | datasets | a candidate adapter (not released) |

`learn` runs them all as one run on `policy:default` (for instance
`splinter learn crates/splinter/examples/stm32_datasheet.md`), printing
each stage as it finishes: the sources are captured; tasks of the requested kinds
(default `recall`) are generated from every text part and admitted by
code; each is solved in the environment it records; each experience is
graded by its task kind's verifiers (a judge only through `verify
--judge`); failures are critiqued and retried once; the first attempts and
revisions that passed become an `sft-final` dataset; and a candidate is
trained on it. The candidate is reported and not released. `--budget`
bounds the whole run's wall-clock time; `--goal` steers what tasks are
asked for; `--dry-run` prints the plan and writes nothing. A `learn` that
stops before training (nothing admitted, nothing passed, the budget spent)
says why and exits 1.

Task kinds: `recall`, `explain`, `predict`, `construct`, `debug`,
`counterexample`, `transform`, `classify`, `retrieve`, `multi-turn`,
`combine` (written by the generator model) and `denoise` (a corrupted
passage to restore, no model needed). Kinds that run code need `python3`.

`verify` appends verdicts from each task kind's own verifiers: formal
(exact match, lenient), executable checks, mutation-validated tests,
agreement among answers, and - only with `--judge REF` - a judge gated by
the calibration `judge calibrate` stored for it. A labelled file is JSON
Lines, `{"experience": "<id>", "label": "pass" | "fail"}`.

`dataset build` views: `sft-final`, `sft-step`, `critic`, `preference`,
`verifier`, `decision`, `retrieval`, `outcome`, `denoise`, `cpt`. The
default `--min-strength` is `consistency`; `--strip` defaults to `all`
(the student sees only the instruction). Objectives brain cannot train
(preference, contrastive, reward, raw text) need `--export-only`.

`train` concatenates the datasets in order and holds the newest records
out for scoring; `--replay` mixes a dataset into training whole, never held
out. `--from local:<checkpoint>+<adapter>` continues training that adapter.

## The front door

`splinter "learn the manual in ./docs"` asks the policy model, through
sven's typed method call, which command the sentence means; the reply is
a list of candidate intents (`learn`, `ask`, `status`, `add_source`,
`list_sources`, `list_runs`, `cancel_run`), each with its arguments and a
confidence. Code decides: a single reading at confidence 0.7 or more, and
0.2 ahead of any other, runs as its command (printed on stderr first).
Anything less is asked back - in the REPL at a terminal you pick one; with
`--json` or a sentence argument the candidates are printed and the exit is
3. A remote model without the opt-in is refused; cancelling a run, running
a command (`cmd:`) and using a remote model are never done on a guess.

## JSON output

With `--json` every command prints one JSON document on stdout; errors
print `{"error": string, "refused": bool}`. Commands that record a run add
`"run"` (its id) to their report. Two reports in detail:

`source list`: `{"sources": [source]}`, each source
`{"id", "kind", "origin", "parts", "bytes", "captured_at"}` - `kind` is
`document`, `repository` or `command`; `origin` is the stored origin
(`{"kind": "document", "path"}`, `{"kind": "repository", "path",
"revision", "skipped"}` or `{"kind": "command", "argv", "cwd",
"exit_code", "timed_out", "stdout_truncated", "stderr_truncated"}`);
`parts` counts files or output streams and `bytes` sums their sizes.

`status`: `{"state", "policy", "recent_runs", "counts"}` - `policy` is
`{"reference", "model", "base", "adapter"}` (`adapter` is `null` until
releases exist); `recent_runs` lists up to five runs as `{"id", "command",
"status", "started_at", "updated_at"}`; `counts` is `{"sources",
"task_sets", "tasks", "experiences", "experience_sets", "datasets",
"candidates"}`.

## Runs

Every command that writes pipeline state records a run under
`<state>/runs/<run-id>/run.json`: the command and its arguments, each
stage as it finishes, its status (`running`, `completed`, `failed`,
`cancelled`) and what it produced. `runs cancel <ID>` asks a run in
progress to stop: it stops the model run in progress and the stage at its
next check, and training at its next optimizer step. A run whose process
died stays `running`.

## Exit status

0 done; 1 the work failed or stopped short of what was asked; 2 refused
before anything ran (usage, an unknown id, a remote model without the
opt-in); 3 a sentence was asked back.
