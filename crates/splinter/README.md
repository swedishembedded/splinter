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
splinter learn <SOURCE>... [--goal TEXT] [--kinds K,.. | --planner REF] [--budget DUR] [--dry-run] [--no-release]
                         [--no-frontier | --distill | --k N [--temperature T] [--top-k N]] [--teacher REF] [--generator REF]
                         [--steps N] [--rank R] [--lr LR] [--bf16-base]
splinter ask <QUESTION> [--open-book SOURCE-ID | --retrieve SOURCE-ID... [--passages N]] [--policy REF]
splinter status
splinter source add <PATH|cmd:COMMAND...> | list | show <ID>
splinter tasks generate <SOURCE-ID>... --kinds K,.. [--generator REF] | variants <TASKSET-ID> [--generator REF] [--per-task N] | list | show <ID>
splinter solve <TASKSET-ID> [--solver REF] [--frontier [--k N] [--temperature T] [--top-k N] [--teacher REF]]
splinter verify <EXPERIENCE-SET> [--judge REF]
splinter critique <EXPERIENCE-SET> [--critic REF] [--retry N]
splinter judge calibrate <LABELLED-FILE> --judge REF
splinter experiences list | show <ID> [--graph] | replay <ID>
splinter dataset build <EXPERIENCE-SET>... --view VIEW [--strip all|keep:K,..|mix:F]
                       [--min-strength executable|formal|consistency|judged] [--export-only]
splinter dataset export <DATASET-ID> --out DIR
splinter train <DATASET-ID>... [--from REF] [--replay-fraction F] [--steps N] [--rank R] [--beta B]
splinter release <CANDIDATE-ID> [--alias NAME] | list
splinter rollback <ALIAS>
splinter eval [REF] [--suite held-out|retention|anchor|FILE] [--freeze FILE]
splinter exam CANDIDATE [--judge REF] [--prompt GOAL]
splinter runs list | show <ID> | cancel <ID>
splinter lineage <ID> [--up|--down|--both] [--depth N]

global: --state DIR  --json  -v  --allow-remote
```

An id is the full `blake3:<hex>`, the hex alone, or a prefix of at least
four hex digits that names exactly one stored object.

`ask` records every answer under `<state>/answers/` with the model that
gave it and, asked through `policy:<alias>`, the release the alias
resolved to; its report carries the answer's `id` and that `release`.

`--open-book` shows the model the whole text of one source. `--retrieve`
shows it instead the `--passages` (default 6) passages of the named sources
that bear on the question: ranked by meaning (Qwen3-Embedding), with a
fixed share of the tail given to the passages only the question's exact
words find (a name, a date), each under its part and section. The answer
records the sources and `splinter lineage` links it to them.

## Models

Every model is named the same way:

| Reference | Model |
|---|---|
| `policy:default` | the configured base (`BRAIN_QWEN_WEIGHTS`, else `Qwen/Qwen3-0.6B` in brain's model store) with the champion's adapter, the release `default` points at; the base alone before any release |
| `policy:<alias>` | the base with the adapter of the release `<alias>` points at (`release --alias`) |
| `local:<checkpoint>[+<adapter>]` | a checkpoint path (absolute, or starting `./` or `../`) or a name in brain's model store, with an optional LoRA adapter file after the first `+` |
| `remote:<provider>/<name>` | a model reached over the network through sven's provider configuration |

A policy alias is resolved once per command, when it is first used: a
run keeps the release it started with however the alias moves meanwhile,
and records it (`learn`'s first stage, a candidate's `parent`).

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
| frontier | `solve --frontier` | a task set | k graded attempts per task, a teacher's graded open-book solve of each task never solved, a pass@k measurement, the kept tasks' task and experience sets |
| variants | `tasks variants` | a task set | a task set of the same facts asked in other words, each recording the task it varies; measured by the gate, never trained on |
| critique | `critique` | an experience set | an experience set of critiques and revisions |
| dataset | `dataset build` | experience sets | a dataset and its manifest |
| train | `train` | datasets | a candidate adapter (not released) |
| release | `release` | a candidate | the gate's numbers; a release and the alias moved, if it passed |

`learn` runs them all as one run on `policy:default` (for instance
`splinter learn crates/splinter/examples/stm32_datasheet.md`), printing
each stage as it finishes: the sources are captured; tasks of the requested kinds
(default `recall`) are generated from every text part, and from the
sections of every concept queued for new tasks, and admitted by code
(the generator is shown what the source is - its file, directory or
command, its title and recorded commit - beside the sections, and every
question answered from the source must name its subject, which that
identity or a cited section must name; two tasks of the set that ask one
question of one subject with different answers are both left out);
each task is solved k times (`--k`) in the environment it records and
every attempt is graded by its task kind's verifiers (a judge only
through `verify --judge`); each task no graded attempt solved is solved
once more by the teacher (`teach`: the policy, or the model `--teacher`
names) open-book - shown its grounding material - and graded the same
way; only the tasks worth training on go on: those the policy fails at
least sometimes and has a verified answer to, its own (the frontier) or
the teacher's (taught); the model that wrote the tasks writes up to three
differently worded questions about each task kept (`variants`: one
request per task, shown the section its evidence falls in, replying in the
shape the tasks come in; each variant keeps the task's reference and
evidence, so the same verifiers grade it, and must state the reference,
stand on its own, still name the task's subject and be no repeat of the
question or of a sibling) - the
same facts in other words, which the gate measures and nothing trains on;
their failed attempts are critiqued and retried
once; the passing attempts, verified teacher answers and revisions, near
duplicates removed and each concept, task kind and verification strength
capped at its share (`select`), become an `sft-final` dataset; a
candidate is trained on it from the champion; and the release gate
decides whether it is released (`--no-release` stops at the candidate).
`--no-frontier` solves each task once, has the teacher solve each one
that failed, and keeps every task. `--budget` bounds the run's wall-clock time: task generation stops at
three tenths of it, the student's attempts at half and the teacher's answers at
four fifths, and training and the exam then run to the end; `--goal` steers what tasks are asked for; `--dry-run`
prints the plan and writes nothing. A `learn` that stops before training
(nothing admitted, no task worth training on, nothing passed, the budget
spent) or whose candidate the gate blocks says why and exits 1.

## Learning to think like a person

```bash
splinter "Learn to think like Thomas Jefferson based on the materials he has written in directory ./jefferson"
```

With no `--kinds`, `learn` plans. After the sources are captured it surveys
them by code (how many parts and sections, how many sections read as advice,
a few excerpts) and shows the survey and the goal to a planner model (the
model `--planner` names, else the configured assistant, else the generator).
The planner chooses from a menu of task kinds, whether to distil, and who the
learner is becoming; code holds the choice to the survey, so a plan cannot
teach advice the sources do not hold. A plan that breaks a rule is sent back
once with the rule it broke and then refused: a run that cannot be planned
says so, and no default replaces its plan. The survey and the plan are in the
report, and naming `--kinds` as well is refused: one of the two decides.

The `advise` kind is what teaches a person's own advice. Its task is a
predicament put to the writer, addressed to the writer (so it names no
document), and its reference is a passage of the writer's own text that
answers it. The generator is shown only the sections that read as advice, and
a task is admitted only when its reference is a passage of its evidence word
for word, so a paraphrase the model wrote is refused. An answer is graded by
the quotation verifier, with no model involved: every passage the answer puts
in quotation marks must be in the task's source text, the advice must be
reproduced, and an answer that quotes nothing or invents a quotation fails.

The exam (`splinter exam CANDIDATE`, and the `exam` stage of `learn`) measures
what the release gate's code-only grading cannot: whether a candidate gives, in
its own words, what a held-out task's reference says. It puts the same held-out
tasks to what the candidate continues (its parent release, else the base) and
to the candidate, closed-book. With `--prompt GOAL` (the `exam` stage of `learn`
passes its goal) the base is asked once more under the goal as one added system
line, and the candidate is also compared with that: training is worth what it
adds beyond telling the base what the goal is. A judge compares each answer
with the reference, and code separately checks whether the answer states a
number, name or quotation the task's source does not hold. Before the judge
grades an arm it is calibrated on controls made from the tasks' own references
- each task's reference answering it, and another's answering it - which no
model wrote; a judge that cannot tell them apart grades nothing and the report
says it makes no claim. A task an arm gave no answer to counts as not done and
is reported as unanswered. The result is a paired sign test of the judged
results, and it never reaches a training set. Each control the judge did not
judge as labelled is reported with the answer and the judge's reason, so a judge
that is not trusted can be seen failing.

The `converse` kind teaches how the writer talks and reasons. Its task is the
opening message of someone speaking to the writer, with a passage of the
writer's text as its reference. A teacher shown the passage answers as the
writer, and a model that is never shown it plays the other speaker, following
up for up to three exchanges; the training record is the whole conversation
with every reply of the writer supervised and no passage in it. A dialogue
is graded by what it states, with no model involved: every number, name and
quotation in the replies must be in the writer's text or in what the other
speaker said, and a dialogue with one that is not teaches nothing. One
dialogue in four ends by asking for a specific the exchange has not given, so
that the student also sees the writer decline to invent one.

Task generation is bounded by the budget and spread over the sources: each
text part is shown through at most four evenly spaced windows of sections,
parts are visited in a stable order that does not follow their names, and the
budget stops generation inside a part. A held-out split never divides texts
that print the same passage (two editions of a letter): records name the
group of overlapping source text they came from, and a group is held out or
trained on whole.

`--distill` skips the policy's own attempts: the teacher answers every task
open-book and the policy is trained on its verified answers, with no
frontier and no critique. It is for a policy that cannot answer a task
closed-book at all, where its attempts are the most expensive part of a run
and teach nothing. A plan can choose it too.

A run needs no flags. The configuration names what a machine knows:
`SPLINTER_ASSISTANT_MODEL` is the stronger model that plans, writes tasks and
teaches when the command names none, `SPLINTER_FRONT_DOOR_MODEL` reads the
sentence, and `SPLINTER_BUDGET` is how long a `learn` may take when it names no `--budget`
(a sentence names none), `SPLINTER_THINKING=1` lets local models reason before
they answer (off, a reasoning model is asked with its reasoning block closed, so
every reply is the answer and comes at once; training renders its answers the
same way, so both settings work), and `SPLINTER_BF16_BASE` holds a large policy's base at bf16 so it
trains on one card. `SPLINTER_REMOTE_CONCURRENCY` is how many requests to a model
reached over an API may be in flight at once (4 by default); a model on the local
device is asked one at a time. Unless `--steps` is given a run trains about two passes
over what it learned, within bounds, at a learning rate suited to a short
LoRA run.

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

A task's concepts: those it declares; otherwise the (source, section)
pairs its evidence falls in (the section of the source part containing
each span's first byte); otherwise its task kind. Every solve through
`policy:<alias>` records the release the alias was (or `base`), and
`status` ranks concepts by their rolling pass rate (the newest 32 graded
solves) under the current release, weakest first, with their rates under
earlier releases; a teacher's open-book solve and a retry helped by a
critique are not counted. When the release gate's retention check fails
on an earlier release's suite, the concepts of the tasks the candidate
forgot are queued under `<state>/curriculum/queue/` (the `release` report lists
them), and the next `learn` generates new tasks from their sections.

A round's training set is deduplicated with the task generator's own
near-duplicate rule, strongest verdict first, then capped: a concept may
take at most a quarter of it, a task kind half, a verification strength
three quarters - never less than an equal split among the groups the
pool actually has.

Task kinds: `recall`, `explain`, `predict`, `construct`, `debug`,
`counterexample`, `transform`, `classify`, `retrieve`, `multi-turn`,
`combine` (written by the generator model) and `denoise` (a corrupted
passage to restore, no model needed). Kinds that run code need `python3`.
A question answered from the source (`recall`, `explain`,
`counterexample`, `classify`, `retrieve`, `multi-turn`, `combine`) names
its subject - the product, document, tool, component or version it is
about - so that it has exactly one answer; a kind whose instruction shows
its material or whose answer is computed by running code carries what its
answer depends on and names none.

`verify` appends verdicts from each task kind's own verifiers: formal
(exact match, lenient), executable checks, mutation-validated tests,
agreement among answers, and - only with `--judge REF` - a judge gated by
the calibration `judge calibrate` stored for it. A labelled file is JSON
Lines, `{"experience": "<id>", "label": "pass" | "fail"}`.

`dataset build` views: `sft-final`, `sft-step`, `critic`, `preference`,
`verifier`, `decision`, `retrieval`, `outcome`, `denoise`, `cpt`. The
default `--min-strength` is `consistency`; `--strip` defaults to `all`
(the student sees only the instruction - also for a teacher's solve,
whatever it was shown). Every conversation starts with the system turn
every solve, `ask` and probe runs under: one short system prompt of
Splinter's own, in place of sven's coding-agent prompt, so the policy is
trained under the prompt it answers under. Chat views are written as
brain's `generic-messages-v2`, `preference` as brain's
`generic-preference-v1` (one pair per line: the student's prompt, the
chosen and the rejected answer), each checked by brain's own parser.
Objectives brain cannot train (contrastive, reward, raw text) need
`--export-only`.

`train` trains the regime its datasets' objective names - chat datasets
by supervised fine-tuning, preference datasets by DPO against the model
the run starts from (`--beta`, the DPO temperature, defaults to brain's);
one run's datasets are all one or all the other. It concatenates them in
order and holds the newest records out for scoring: a supervised candidate
reports the held-out loss of base and candidate, a preference candidate
brain's preference score on the held-out pairs (how often, and by how
many nats, it prefers the chosen answer more than its reference does).
From `policy:<alias>` (the default) it continues the adapter of the
release the alias points at - never the base weights once a release
exists - and a supervised run replays `--replay-fraction` (default 0.25)
of every earlier release's trained-on chat records, a seeded sample that
is the same on every run and never includes what that release held out;
replayed records are never held out. A preference run replays nothing:
brain's preference trainer trains on its pairs alone. `--from
local:<checkpoint>+<adapter>` continues that adapter instead, with no
replay, and such a candidate cannot replace a champion. The candidate
records its regime; the release gate grades either the same way.

## Releases

`release <CANDIDATE-ID>` decides a candidate against the release its alias
(`--alias`, default `default`) points at - the champion, or the base before
any release - and refuses a candidate that was not trained from it. Both
are graded closed-book by each task's own verifiers (no judge) on the same
suites, and the candidate is released only if all four checks pass; each
is printed with its numbers, and a check that could not be measured fails:

| Check | Passes when |
|---|---|
| improvement | on the new datasets' held-out tasks - the records training held out, and the variants of the tasks it trained on - a one-sided paired sign test over the tasks only one model got right is significant at alpha 0.05; ties and tasks without a verdict for both are excluded and counted |
| retention | on each earlier release's held-out tasks, the candidate's accuracy is at most 0.05 below the champion's (each release reported) |
| anchor | on the anchor suite in force, the candidate's accuracy is at most 0.02 below the champion's |
| serve | `brain serve --adapter <candidate>` (the `brain` on `PATH`, or `SPLINTER_BRAIN_BIN`), with the base checkpoint the candidate was trained on as its `BRAIN_QWEN_WEIGHTS`, starts - brain binds the adapter only to the base whose digest training recorded on it - reports the candidate's adapter digest, and re-answers up to 8 held-out tasks through its OpenAI-compatible endpoint, both sides decoding greedily and without a reasoning block, with the same answers as in-process on at least three quarters of them (the same verdict, and either the same text over its first nine tenths, runs of whitespace aside, or - worded otherwise, as a sampling server or another summation order will - the same meaning: embedded by Qwen3-Embedding, an answer must be nearer its own in-process answer than to the in-process answer of any other task, so the comparison cannot call everything alike); each task answered differently is reported with both answers |

The improvement check measures whether the candidate learned the facts it
was trained on, on questions it was not trained on: the held-out records
(a tenth of the new data, the newest) alone are too few for a sign test to
reach significance, and questions about other facts cannot improve for a
campaign that teaches facts. So the suite is completed with the stored
variants of the tasks the candidate's trained-on records were projected
from (`learn` writes them; `tasks variants` writes them for any task
set). A variant of a task that was held out measures nothing the
candidate learned and is not used. A variant whose wording the candidate
trained on - the same question, a near duplicate, or its words inside a
longer training prompt - is left out like any leaked held-out task. The
gate prints how many variants were measured and how many were left out,
by reason; the manifest records them. Retention, anchor and serve are
measured as before.

A release is written once under `<state>/releases/<hex>/`: the adapter
file, read-only, and `manifest.json` in canonical JSON - the base model and
its digest, the adapter digest, the parent release, the candidate, the new
datasets and the replay sample with their digests, the training record
(brain's included), every gate number with the anchor suite's version and
digest, and `created_at`. `<hex>` is the manifest's digest: the release
id. `<state>/releases/aliases/<name>` names the release an alias points
at; it moves only from the champion the gate measured against. `release
list` shows every release with its parent and aliases; `rollback <ALIAS>`
points the alias at the release its current one was trained from, and
refuses when there is none.

An adapter belongs to one base model. To serve a release without
Splinter, name that base - the manifest's `base_model` and `base_digest`
say which checkpoint it is - and the release's adapter, as the serve check
does:

```bash
BRAIN_QWEN_WEIGHTS=<base checkpoint> brain serve --openai --adapter <state>/releases/<hex>/adapter.safetensors
```

brain serves the adapter on that one Qwen3 base, as the model
`brain/qwen3`, and refuses to start when the base's digest is not the one
the adapter records.

`eval <REF>` grades one model - a candidate id, or a model reference -
closed-book on `--suite held-out` (the default: a candidate's new data, or
the release a policy alias points at), `retention` (every earlier release,
each reported), `anchor`, or a FILE of tasks. The anchor suite is frozen
from a file with `eval --suite anchor --freeze FILE` (each different file
is the next version; the same tasks are the same version) and shown with
`eval --suite anchor`. An anchor file is JSON Lines, `{"instruction",
"reference", "kind"?}`, `kind` (default `recall`) a closed-book kind a
formal verifier grades.

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
A sentence naming a person to think like and a directory of their writing is
read as a `learn` of the directory; a path the model copied wrongly is
replaced, by code, by the one path the sentence names that exists, and never
by a guess. `SPLINTER_FRONT_DOOR_MODEL` names a model for reading sentences
other than the policy, since a larger model reads them better.

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

`status`: `{"state", "policy", "recent_runs", "counts", "concepts"}` - `policy` is
`{"reference", "model", "base", "adapter", "release"}` (`adapter` and
`release` are `null` until releases exist); `recent_runs` lists up to five runs as `{"id", "command",
"status", "started_at", "updated_at"}`; `counts` is `{"sources",
"task_sets", "tasks", "experiences", "experience_sets", "datasets",
"candidates"}`; `concepts` is `{"policy", "concepts", "measured",
"weakest", "queued"}`, each of `weakest` `{"concept", "current",
"releases"}` with `{"release", "graded", "passes", "rate", "since"}` per
release.

## Runs

Every command that writes pipeline state records a run in the experience
database: the command and its arguments, each stage as it finishes, its
status (`running`, `completed`, `failed`, `cancelled`) and what it
produced. `runs cancel <ID>` asks a run in progress to stop, by raising a
signal any process can see: it stops the model run in progress and the stage
at its next check, and training at its next optimizer step. A run whose
process died stays `running`.

## State

Everything Splinter knows lives in one experience database under
`<state>/expdb`: sources, tasks, experiences, annotations, sets, runs, and the
metadata of datasets, candidates, releases, answers, suites, calibrations and
the curriculum queue. The mutable names (an alias, the anchor suite in force)
are pointers whose whole history is kept; two processes moving one from the same
value cannot both win. Every experience is also an attempt in the database's
graph, a verdict is evidence about that attempt, and a dataset, a training run
and a release record where they came from, so `lineage` traces a release back to
what it learned from.

The bulk files a tool needs as real files (adapters, dataset JSONL) are kept
under `<state>/artifacts/<aa>/<digest>` and tracked by the database: a file is
written first, then one commit makes it official, so a crash leaves at most an
unrecorded file that `state maintain --collect` removes. `<state>/work` holds
scratch that exists only while a command runs.

`state status` reports what the database holds as files and what was written
off: `{"state", "storage": {"segments", "blob_packs", "index_runs", "pins",
"history", "artifacts", "losses"}, "losses", "pins"}`. A dataset keeps the
database as it was alive under a name (`pins` lists them); `state unpin <holder>`
lets one go so `state maintain --collect` can free its files. Every commit leaves small files
behind, and a command that opens the database reads them all, so `state
maintain` merges them, indexes what is not indexed and retires finished
writers; nothing stored changes, and it is safe beside a running command.
`state maintain --collect` also deletes files nothing reaches that are past
their grace period (a snapshot a dataset pinned is never touched) and artifact
files no commit made official. Its report is `{"run", "before", "after",
"segment_groups", "blob_groups", "writers_retired", "removed",
"orphan_artifacts"}` with the last two `null` unless `--collect` was given.
State written before the experience database is not read: regenerate it.

### Archive and recovery

`state archive FILE` packs the database and the artifacts it tracks into one
`.tar.zst`: the same state always gives the same bytes. `--no-artifacts` leaves
the files out, and `--since PREV` carries only what an earlier archive lacks (such an
archive restores together with it). `state restore FILE [BASE...]` unpacks into
an empty state root, checks every member against its digest, verifies the whole
state deeply next to the root and moves it into place only if it passes; a
restore that fails leaves the root as it was.

`state verify [--deep]` reports every missing or damaged file of the database and
of the artifacts without stopping at the first, and exits 1 if there is
one. `state repair [--from PATH]... [--accept-loss]` fills holes from copies (an
archive, or another state root): every file is named by the hash of its bytes, so
a copy is used only if it is exactly right. Index files are rebuilt from the
records. What no copy has stays reported unless `--accept-loss` is given, which
withdraws damaged database files and writes lost artifacts off in a ledger that
`state status` lists; a lost artifact is from then on refused with its digest
wherever it would have been used.

`state verify`: `{"database": {"manifests", "segments", "blob_packs",
"index_runs", "problems"}, "artifacts", "artifacts_unchecked"}`. `state repair`:
`{"filled", "rebuilt", "quarantined", "lost", "unresolved"}`. `state archive`:
`{"snapshot", "files", "carried", "artifacts", "bytes"}`. `state restore`:
`{"files", "artifacts", "verified"}`.

`release list`: `{"releases": [{"id", "created_at", "candidate",
"parent", "adapter_digest", "aliases"}]}`. `release`: `{"run",
"candidate", "alias", "champion", "gate", "release", "dir"}`, where `gate`
is `{"config", "improvement", "retention", "anchor", "serve", "passed"}`
and each check is `{"passed", "measured", "reason"}` (`measured` is `null`
when it could not be measured). `eval`: `{"model", "reference", "anchor",
"scores"}` (plus `"run"` when it froze a suite or graded a model).

## Lineage

`lineage <ID>` takes any artifact's id - a source, a source part's
content, a task or task set, an experience or experience set, a dataset, a
candidate, a replay sample, a release, an adapter digest, an answer - as
the full id, its hex, or a prefix naming exactly one; a prefix naming
artifacts in more than one store is refused with every one it names. It
prints two trees: where the artifact came from (`--up`), down to the source
part and byte range each task is grounded in with those bytes quoted, and
what came from it (`--down`): tasks, experiences, datasets, candidates,
releases and answers. Both by default; `--depth N` stops N edges out. The
graph is derived from what the stores already record, on every call:

| From | Relation | To |
|---|---|---|
| content | `part_of` | the source holding it as a part |
| span | `span_of` | the source it names a part of, else the content it indexes |
| task | `evidence` | each span it is grounded in |
| task set, experience set | `member` | each task, experience |
| experience | `attempts`, `ran_in`, `solved_by` | its task, environment snapshot, solver model |
| experience | `critique_of`, `retry_of`, `revision_of`, `preferred_over`, `variant_of` | the experience its relation names |
| verdict | `verdict_on`, `produced_by` | the experience it grades, the verifier that gave it |
| dataset | `projected_from` | each experience, task and source content its manifest names |
| candidate, release | `trained_on`, `trained_from`, `replayed`, `adapter` | its datasets, parent release, replay sample, adapter digest |
| replay sample | `sampled_from` | each earlier release it drew from |
| release | `release_of` | its candidate |
| answer | `answered_with`, `answered_by`, `open_book` | the release, the model, the source shown |

With `--json`: `{"nodes": [{"id", "kind", "label"}], "edges": [{"from",
"to", "relation"}]}`. The artifact asked about is the first node; every
node reached follows once, in walk order; an edge means `from` was derived
from `to`, whichever way it was walked. `kind` is one of `source`,
`content`, `span`, `task`, `task_set`, `experience`, `experience_set`,
`environment`, `model`, `verdict`, `producer`, `dataset`, `candidate`,
`replay`, `release`, `adapter`, `answer`. Artifacts without an address of
their own have ids of their kind: `span:<content-hex>:<start>-<end>`,
`model:<identity>`, `verdict:<experience-hex>:<n>`,
`producer:<name>@<version>`.

## Exit status

0 done; 1 the work failed or stopped short of what was asked (a candidate
the gate blocked included); 2 refused
before anything ran (usage, an unknown id, a remote model without the
opt-in); 3 a sentence was asked back.
