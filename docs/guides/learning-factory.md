<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# The learning factory

A recipe in a task repository says what an experiment does: which tasks, which
agents on which models attempt them, which trials become training data, which
candidates are trained, and how each is judged. `splinter run` carries the whole
recipe out into a run directory, and runs it again without repeating anything:
every stage whose inputs did not change is reused.

```sh
cd bench                                   # the task repository
splinter generate collections/vim-headless-mastery.yaml
splinter validate recipes/vim-hosted-sft.yaml
splinter plan recipes/vim-hosted-sft.yaml
splinter run recipes/vim-hosted-sft.yaml --out runs/vim-hosted-sft
splinter status runs/vim-hosted-sft
splinter inspect runs/vim-hosted-sft --stage sft
splinter inspect runs/vim-hosted-sft --stage hosted --viewer harbor
splinter verify runs/vim-hosted-sft
splinter report runs/vim-hosted-sft
```

## Starting from a template

`splinter templates` lists the task repositories Splinter can write as a
starting point, each working as written: its recipes, the collection, tasks,
environment and plans they name. `splinter init --template NAME DIR` writes
one into a directory and writes over no file; it refuses, naming the files,
before writing any. `repository-practice` ships an example project, which `init` makes a git
repository of and names by its commit in the recipe.

| Template | What its recipe does |
|---|---|
| `agent-sft` | a model learns a task collection from its own verified attempts |
| `knowledge-learning` | a model learns what documents say from closed-book tasks generated from them |
| `generations` | a generation of a model, promoted only when it improves, transfers and retains |
| `harness-search` | the harness that verifiably beats a baseline, among its mutations |
| `repository-practice` | a model learns a repository from its own tests, by faults seeded into it |

```sh
splinter init --template agent-sft my-repo
cd my-repo && splinter validate recipes/agent-sft.yaml
```

## What a run needs on this machine

- **The execution worker.** `make worker/install` in Splinter's checkout installs
  `splinter-harbor-worker` (Harbor's job API, pinned) into `workers/harbor/.venv`;
  put its `bin/` on `PATH` or set `SPLINTER_HARBOR_WORKER` to the program.
- **Connections**: how each model is reached. The task repository's
  `connections.yaml` gives them for every machine; this machine's
  `~/.config/splinter/connections.yaml` (or `$XDG_CONFIG_HOME/splinter/...`)
  overrides any of them by name. A credential is named by the variable that
  holds it, never written in either file; the worker reads the variable when it
  routes the agent's requests. A model whose connection neither file defines is
  refused, naming the files read.

```yaml
connections:
  brain: { protocol: openai, url: "http://127.0.0.1:8990/v1" }
  hosted: { protocol: openai, url: "https://example.invalid/v1", token: "env:SVEN_KEY" }
```

- **brain**, for a recipe that trains or evaluates: `--brain-endpoint URL` with
  `--brain-artifact-store DIR` and `--brain-adapter-dir DIR` (or
  `--brain-adapter-manifest FILE`).

## The recipe

A recipe (`splinter.recipe/v2`) is a graph of stages over typed products. It
names its inputs (files of the repository), binds model roles, and lists stages:
each names the operation it `uses`, binds the operation's inputs to products of
the recipe (`$inputs.NAME`) or of stages above it (`$stages.ID.PORT`), and gives
the operation's parameters under `with:`. A stage's id is its key in the run.

```yaml
schema: splinter.recipe/v2
name: vim-mastery
inputs:
  course: { kind: files, path: collections/vim-headless-mastery.yaml }
models:
  student: "brain:Qwen/Qwen3-4B"
  teacher: "hosted:deepseek/deepseek-v4.1-flash"
stages:
  - id: tasks
    uses: tasks.pin
    inputs: { collection: $inputs.course }
    with: { split: train }
  - id: local
    uses: attempt.collect
    inputs: { tasks: $stages.tasks.tasks }
    with: { agent: sven, model: student, environment: environments/vim/environment.yaml,
            attempts: 4, seed: 1 }
  - id: hosted
    uses: attempt.collect
    inputs: { tasks: $stages.tasks.tasks, unsolved_by: [$stages.local.attempts] }
    with: { agent: sven, model: teacher, environment: environments/vim/environment.yaml,
            attempts: 4, seed: 2 }
  - id: sft
    uses: dataset.project
    inputs: { attempts: [$stages.local.attempts, $stages.hosted.attempts] }
    with: { target: student, seed: 3 }
  - id: lora-r16
    uses: model.train
    inputs: { dataset: $stages.sft.dataset }
    with: { base: student, method: lora, params: { rank: 16, steps: 500 }, seed: 4 }
  - id: eval-r16
    uses: evaluation
    inputs: { candidate: $stages.lora-r16.model }
    with: { plan: eval/vim-dev.yaml, agent: sven, baseline: student }
  - id: release
    uses: promotion
    inputs: { evaluations: [$stages.eval-r16.evaluation] }
    with: { promote: false }
outputs: { dataset: $stages.sft.dataset }
```

| operation | inputs | output |
|---|---|---|
| `tasks.pin` | `collection` (a recipe input) | `tasks` |
| `attempt.collect` | `tasks`, `unsolved_by` (attempts, optional) | `attempts` |
| `attempt.import` | `tasks`, `jobs` (Harbor job directories, recipe inputs) | `attempts` |
| `dataset.import` | `records` (a recipe input) | `dataset` |
| `dataset.project` | `attempts`, `answers` (one or more of either) | `dataset` |
| `dataset.extend` | `base`, `candidates` (datasets) | `dataset` |
| `dataset.text` | `documents` (a recipe input) | `dataset` |
| `harness.search` | `tasks` | `harness` |
| `task.mutate` | (a repository and a commit) | `tasks` |
| `model.train` | `dataset` | `model` |
| `evaluation` | `candidate` (a model), `tasks` (optional) | `evaluation` |
| `promotion` | `evaluations` (one or more) | `release` |
| `agent` | `input` (files, one or more) | `output` |
| `harness` | `input` (files, any number) | `output` |
| `attempts.failures` | `tasks`, `attempts` (one or more) | `failures` |
| `dataset.rlvr` | `tasks`, `attempts` (one or more) | `tasks` |
| `source.ingest` | `documents` (files) | `sections` |
| `knowledge.admit` | `sections`, `proposals` | `knowledge` |
| `knowledge.query` | `sections` | `passages` |
| `task.generate` | `documents` (one or more) | `tasks` |
| `answer.collect` | `task` | `answers` |
| `environment.build` | `documents` (optional) | `environments` |
| `environment.qualify` | `environments`, `attempts` | `environments` |

A stage consumes only stages above it, so a recipe reads top to bottom and runs
in that order. A stage that `uses: recipe` (`with: { recipe: PATH }`) is a
subrecipe: its stages join the graph under the stage's id (`book.sft`), its
inputs bound to the stage's and its outputs the stage's ports. Each candidate's
evaluation is a stage; several evaluations of one experiment state how the
plan's significance level is shared (`comparisons: N`), and those with the same
baseline and plan share one baseline arm, run once.

Unknown operations, ports and parameters are refused, every seed is written
down, and `splinter validate` reports every problem at once: a missing file, a
model role with no connection, a reference to a stage below, a product of the
wrong kind, a credential written as a literal.

- **Collections** name a directory of Harbor tasks, and optionally the command
  that compiles them (`generate:`, which `splinter generate` runs).
- **Environments** say what the sandbox must enforce - resources, read-only
  mounts, the network and the least `isolation` (`strong`, `confined` or `any`) -
  and leave the sandbox to the machine: with no `provider` (`auto`) the worker
  picks the strongest it has that enforces all of it: `docker`, then `userns`,
  then `confined` (proot under Landlock and seccomp through `splinter-confine`:
  reads limited to the system directories, the task's image and its mounts,
  writes and the network confined, the host's processes still visible), then
  plain `proot` (open network only). The same recipe runs on a
  full machine and in a hardened container; each trial records the sandbox it
  ran in and what it enforced, the run report lists them per pass, and an
  environment no sandbox here can honour is refused, never run weaker.
- **Import stages** (`attempt.import`) read the trials of Harbor jobs run
  elsewhere as attempts and run nothing ([below](#importing-harbor-jobs)).
- **Attempt stages** (`attempt.collect`) name an agent and its own argv (`args`, passed as given,
  never through a shell), the files those arguments refer to, an appended system
  prompt, environment variables (`env:NAME` passes this machine's variable), a
  model role or none (the agent's own provider), attempts, retries and seed. A
  stage given `unsolved_by` runs only the tasks those stages left unsolved, frozen
  when it starts. A pass with `interactive: true` is played by people (below). A pass
  with `hint: solution` works each task from the run's own copy of it whose
  instruction carries the task's reference solution (`solution/solve.sh`), in
  the task's own sandbox and graded by its own verifier (below).

## Training records made elsewhere

A published dataset of chat records trains without being rebuilt from trials. A `dataset.import` stage reads a
directory of the task repository, named as a recipe input, that holds `train.jsonl` and optionally
`validation.jsonl`, and carries the lines over byte for byte as a dataset that `model.train` reads like any other:

```yaml
inputs:
  records: { kind: files, path: data/vim-sft }       # train.jsonl, validation.jsonl
stages:
  - id: sft
    uses: dataset.import
    inputs: { records: $inputs.records }
    with: { target: student, max_block: 16384 }
```

Every line must be a chat record with a supervised message (`train: true`); one that is not is refused with its
file and line, and the stage fails rather than drop it. A task in both files is refused, and so is a record that
keeps a turn's reasoning while `thinking` is false (the default, no-think mode). A record longer than `max_block`
tokens of the target is excluded and listed in `excluded.jsonl`. The records are the caller's word: the stage cannot
tell whether a trial was verified or a task is examination material, so keep exam tasks out of the files.

An `evaluation` stage takes `files`, `system_prompt` and `budget` as an attempt stage does, for every arm. A
candidate trained on records whose system prompt carries an appended instruction, or in a mode a configuration
file sets, is then evaluated under the same.

## Growing a dataset with new solutions

New verified trajectories make a dataset better only when they are other solutions. A `dataset.extend` stage takes a
`base` dataset and a `candidates` dataset (new attempts projected by `dataset.project`, or imported records) and makes
a dataset that holds every record of the base, byte for byte and in order, followed by the candidates that qualify:

```yaml
  - id: more
    uses: dataset.project
    inputs: { attempts: [$stages.local.attempts] }
    with: { projection: native-sft, target: student, max_block: 16384, seed: 4 }
  - id: grown
    uses: dataset.extend
    inputs: { base: $stages.sft.dataset, candidates: $stages.more.dataset }
    with: { target: student, per_task: 4, ignore_call_arguments: [intent],
            task_prefixes: ["bench:", "swedishembedded/vim-", "train/"], holdout_seed: 17925836769974212992 }
```

Records name their task in different ways (`swedishembedded/vim-001-x` in published records, `train/001-x` in records projected from a run): `task_prefixes` lists prefixes taken off a record's `metadata.task` (or its `metadata.group`) so both are one task. A candidate is admitted when it was made under the base's system prompt and tool schema, the sequence of its tool
calls (name and arguments, whitespace normalised, `ignore_call_arguments` left out) is not one its task already has,
and the task has fewer than `per_task` records. Otherwise it is left out and listed in `excluded.jsonl` as
`prompt_mismatch`, `tools_mismatch`, `duplicate_solution` or `over_task_cap`. The candidates are weighed in the order
of their task and content digest, so what is admitted does not depend on the order the attempts were produced in.

A task's side never changes: a task the base holds out keeps its new records in `validation`, a task the base trains
on keeps them in `train`, and a task new to the base is held out as a whole, or trained on as a whole, by a hash of
`holdout_seed` and the task at `holdout_percent` (10 by default); use the seed the base was split with. A task on both
sides stops the stage. `lineage.json` names the parents, the counts and every left-out reason.

## Harness stages

An attempt stage may run a sven harness in place of the agent's own
conversation: `harness:` names a harness file of the task repository, and each
attempt runs it (`sven harness run`) on the task's workspace, given the task's
instruction as its `task` input. The task's own verifier decides, as for any
attempt, so a harness and the plain agent are measured on the same tasks:

```yaml
  - id: planned
    uses: attempt.collect
    inputs: { tasks: $stages.tasks.tasks }
    with: { agent: sven, model: student, harness: harnesses/plan-then-edit.yaml,
            environment: environments/vim/environment.yaml, attempts: 4, seed: 6 }
```

The harness is pinned like any file the agent is handed: a changed harness is
a changed stage, and a fork reuses everything else. `splinter validate`
refuses a harness that does not compile or declares no `task` input, one run by
an agent other than sven, without a model role, interactively or beside a
`system_prompt` (a harness carries its own instructions). A harness run records
its trajectory but not its model requests, so a dataset over a harness stage
uses `projection: portable-sft`.

## Searching for a harness

`harness.search` looks for the harness that verifiably beats a baseline. The
baseline harness runs on the pinned tasks as an arm of its own; each of
`candidates` candidates is one mutation of the baseline (`strategy:
best-of-n`) or of the best found so far (`hill-climb`), drawn by the seed from
a mutation space (knobs: a path into the harness and the values it may take)
and kept only when Sven's compiler accepts it. Each runs as an arm with the
same agent, model, environment, attempts and attempt seeds, and is compared
with the baseline by the evaluation's paired sign test at `alpha` divided by
the candidates (the best is chosen, so the split keeps the chance that a
candidate no better than the baseline wins within `alpha`). Every arm
is a committed product, so a stopped search resumes without running one
again. The product holds `harness.yaml` (the winner, else the baseline),
each candidate under `candidates/` and `search.json` with every candidate's
decision.

```yaml
  - id: search
    uses: harness.search
    inputs: { tasks: $stages.train-tasks.tasks }
    with: { agent: sven, model: student, environment: environments/default.yaml,
            harness: harness/baseline.yaml, space: harness/space.yaml, candidates: 4,
            strategy: hill-climb, attempts: 4, seed: 1 }
```

## Structured extraction: `agent` and `foreach`

An `agent` stage asks a model for one value of a JSON Schema: the model is
given the instruction file, the schema and the stage's inputs as text. A reply
that breaks the schema goes back to the model with the validator's complaint
(`repairs`, two by default); an input the model never answers validly is
rejected, with the reason. A stage with `foreach` runs once per item of a
collection - here each file of a recipe input - at most `concurrency` (1 to
64) at once:

```yaml
schema: splinter.recipe/v2
name: facts
inputs:
  notes: { kind: files, path: notes }
models:
  writer: "hosted:deepseek/deepseek-v4.1-flash"
stages:
  - id: facts
    uses: agent
    foreach: $inputs.notes
    concurrency: 4
    inputs: { input: $item }
    with: { instruction: prompts/extract.md, output_schema: schemas/fact.json,
            model: writer, deadline_seconds: 120 }
```

Each item is committed on its own, under its content: a run stopped halfway
resumes without asking again for a finished item, and a fork after one note
changed asks only for that note. The stage's product lists every item in
`items.json` (`splinter.items/1`), the rejected ones with their reasons, and
holds each answered item's `output.json` and `call.json` (the model that
answered and the tokens it spent) under the item's id. The model is called
from Splinter's own process, with the key the connection names read from its
environment; a resume that recomputes nothing reaches no model.

A `harness` stage runs a sven harness over the stage's files for the value it
returns: a workflow of model calls (plan, draft, check, repair) where `agent` is
one call. Its agent holds no tools (`agent: { profile: none }`); the stage's
files reach it as a list of `{ name, text }` under its `files` input (or the one
`files_input` names), beside the literal `inputs`, and the product holds
`output.json` and `trajectory.json`. A harness that ends in its own `fail`
rejects the item. `splinter validate` refuses a harness that would run
commands (a shell, a command gate): one that must, works a task in its sandbox,
as an attempt stage's `harness:`.

```yaml
  - id: summaries
    uses: harness
    foreach: $inputs.notes
    inputs: { input: $item }
    with: { harness: harnesses/summary.yaml, model: writer }
```

## Knowledge from documents

`source.ingest` splits each document of its input into sections (at headings
for Markdown, at blank lines otherwise), one item per section holding its
`text.md` and `source.json` (the document, its title, the section's number and
byte range, the document's digest). An `agent` stage over the sections proposes
typed knowledge, each object with the quote it rests on
(`{ "objects": [{ "id", "kind", "document", "quote", "data" }] }`), and
`knowledge.admit` keeps only what the sections state: an object stands when its
quote is in its section verbatim, its data has its kind's keys, its numbers and
values are in its quote (a fact's key in the quote's own words, `attribute |
entity`); conflicts are flagged, amendments resolved, every refusal kept with its
reason, in the pack `pack.json`.

```yaml
  - { id: sections, uses: source.ingest, inputs: { documents: $inputs.docs } }
  - id: proposed
    uses: agent
    foreach: $stages.sections.sections
    inputs: { input: $item }
    with: { instruction: prompts/extract.md, output_schema: schemas/objects.json, model: teacher }
  - id: knowledge
    uses: knowledge.admit
    inputs: { sections: $stages.sections.sections, proposals: $stages.proposed.output }
    with: { pack: shop-policy }
```

Knowledge from several documents is merged by ingesting them together: one
admission over all their sections relates what each says, flagging conflicts
and resolving amendments in one pack.

`knowledge.query` gives each question of a file in the task repository (a YAML
list of strings) the sections that bear on it, best first, by lexical retrieval
(rare words weigh more; words are matched as written, not stemmed). Each
question is an item holding `question.md` and `passages.json` (each passage's
section, document, title, score and text), for a stage over the items to read:

```yaml
  - id: found
    uses: knowledge.query
    inputs: { sections: $stages.sections.sections }
    with: { questions: questions.yaml, passages: 3 }
```

## Tasks from documents

`task.generate` asks a generator model for tasks of the named kinds (`recall`,
`advise`, `debug`, ...) from each document, and code admits a task only when
its evidence quotes the document verbatim, it stands on its own, names its
subject and is new; a kind that runs code is offered its runtime, found on the
default search path. Each admitted task is an item: `task.json`, and
`sources/`, the document its evidence is in. `rejected.jsonl` keeps every
refusal with its reason. `answer.collect`, run once per task, has a model answer
it in Splinter's own process (closed-book, or in its kind's runtime) and grades
the answer with the verifiers the task's kind names, the same verdicts as
wherever an answer is graded: a quotation or grounding check reads the item's
`sources/`, a code check runs in the process sandbox. A judge grades only where
one is given; a kind solved in a workspace, or whose check's evidence or
runtime is missing, is refused by name. Each item holds `experience-N.json` and
`verdicts-N.json`.

```yaml
  - id: generated
    uses: task.generate
    inputs: { documents: [$inputs.docs] }
    with: { kinds: [recall, arithmetic], model: teacher, max_output_tokens: 16000 }
  - id: answered
    uses: answer.collect
    foreach: $stages.generated.tasks
    inputs: { task: $item }
    with: { model: student, attempts: 4 }
```

## Tasks from a repository's own tests

`task.mutate` makes a revision of a git repository into Harbor tasks by the
repository's own tests. The revision is named by its commit, so the stage is
the tree's. How the tests run is read by rule (or given as `test_command` and
`files`), and the baseline is run twice: a red or flaky one refuses the stage.
Small faults are seeded into the files under test, each kept only when the
tests catch it reproducibly, reversing it passes, and the tests that catch it
can be hidden from the solver. A family is the file a fault is in;
`eval_percent` of the families, drawn by the seed, go to the `eval` split and
the rest to `train`, so nothing trained on shares a file with what is
examined. Each task is a Harbor folder (`SPLIT/FAMILY-KEY`) for `tasks.pin`;
`mutate.json` records what was admitted, refused and held out.

```yaml
  - id: practised
    uses: task.mutate
    with: { repository: project, revision: 4f1c...e2, max_tasks: 24, eval_percent: 25, seed: 1 }
  - { id: train-tasks, uses: tasks.pin, inputs: { collection: $stages.practised.tasks }, with: { split: train } }
```

## Environments from an objective

`environment.build` compiles a learning objective (a file of the task
repository) into admitted, executable training environments, as `splinter env
compile` does: each buildable skill's generator plugin makes environments along
its difficulty knobs, and each passes the admission gates (its verifier
red-teamed, its base pass rate banded) before it is kept. The product holds the
instances as Harbor task folders by split, which `tasks.pin` pins like a
compiled collection, and `compile.json`, the report of every environment, the
rejected and quarantined ones with their reasons:

An environment whose gates all passed still waits for its band: a base model it
always solves, or never solves, gives a group-relative trainer nothing to learn
from. `environment.qualify` measures each environment's base pass rate from the
counted attempts an attempt stage made at its dev instances (one model's), and
re-judges its admission report (`admission/ENV.json` in the build's product)
against its band. Its product holds the instances of the environments admitted
now, by split, and `qualify.json`, every re-judged report; with none admitted,
the stage is rejected with each one's reasons:

```yaml
  - id: built
    uses: environment.build
    with: { objective: objectives/python.md, environments: 4, instances: 8, seed: 7 }
  - { id: dev, uses: tasks.pin, inputs: { collection: $stages.built.environments },
      with: { split: dev } }
  - id: base
    uses: attempt.collect
    inputs: { tasks: $stages.dev.tasks }
    with: { agent: sven, model: base, environment: environments/python.yaml,
            attempts: 8, seed: 11 }
  - id: qualified
    uses: environment.qualify
    inputs: { environments: $stages.built.environments, attempts: $stages.base.attempts }
  - { id: tasks, uses: tasks.pin, inputs: { collection: $stages.qualified.environments },
      with: { split: train } }
```

## The task repository's own operators

A task repository may add operations of its own: a program it ships, declared
in `operators/NAME.yaml` (`splinter.operator/v1`) and used by name in its
recipes like a built-in one.

```yaml
schema: splinter.operator/v1
name: text.upper
revision: 1
summary: a text file upper-cased
inputs: [{ name: text, kind: files, many: false, optional: false }]
output: { name: upper, kind: files, many: false, optional: false }
parameters: { type: object, properties: { suffix: { type: string } },
              additionalProperties: false }
reproducibility: deterministic
command: [./bin/upper.sh]
code: [bin/upper.sh]
```

Splinter runs the command with `--request FILE`, a `splinter.operator-request/1`
naming its `params`, its `inputs` (each port's paths), the directory `out` it
writes its product into, and the `repository`. It runs in the process sandbox:
it reads its inputs, the request and the repository, writes only `out`, and
reaches the network only with `network: true`, within `timeout_seconds`. It
exits 0 having written its product, 3 to reject its input (the reason on its
last line of output), anything else when it failed. Its parameters are checked
against `parameters` before anything runs, and `code` is pinned: a changed
program is a changed input, which a run refuses and a fork recomputes.

An operator file is the same on every machine; what a machine lets an operator
have beyond its inputs and the repository is the machine's to say, in
`~/.config/splinter/operators.yaml`. An operator that declares `network: true`
or needs variables (`env: [NAME]`) is refused before anything runs unless its
grant says so; a grant may also let it read paths outside the repository, and
write the paths where it keeps state between runs:

```yaml
schema: splinter.operator-grants/v1
grants:
  exam.sealed:
    read: [/srv/bench/sealed, /srv/models]
    write: [/srv/bench/private]
    network: true
    env: [BENCH_BRAIN_KEY]
```

Granted variables are passed from the controller's environment by name (a
grant never holds a value), credentials included, and the grant is part of the
operator's pinned identity.

A collection an operator compiles is pinned like a collection file: bind the
pin's `collection` to the operator's output, and its tasks (every directory
holding a `task.toml`, of the pin's split) are read when the pin runs:

```yaml
  - { id: compile, uses: course.compile, inputs: { course: $inputs.course } }
  - { id: tasks, uses: tasks.pin, inputs: { collection: $stages.compile.tasks },
      with: { split: train } }
```

An evaluation draws its tasks from its plan's collection, or, when the plan
names none, from a pin the stage is given: a development plan then states its
`split` and environment, and the stage binds `tasks` to the pin, so a run that
compiles its collection evaluates on the same compiled tasks:

```yaml
  - { id: held, uses: tasks.pin, inputs: { collection: $stages.compile.tasks },
      with: { split: eval } }
  - id: eval-r16
    uses: evaluation
    inputs: { candidate: $stages.lora-r16.model, tasks: $stages.held.tasks }
    with: { plan: eval/vim-dev.yaml, agent: sven, baseline: student }
```

## What each stage guarantees

- **Attempts** (`attempt.collect`). Every attempt is graded by the one outcome classifier: a reward-0
  trial is a graded failure and is kept; a sandbox, model-endpoint, verifier or
  agent failure is retried up to `retries` times and then kept as what it was,
  never as the agent's 0. A killed controller resumes without repeating an
  attempt that reported. The trials, every attempt's report and the native Harbor
  output are in the stage's directory (`hosted/`).
- **Dataset** (`dataset.project`, `projection: native-sft`). Every verified success becomes one record: the
  conversation exactly as the agent's model was sent it - its own system prompt
  and tools - with every action supervised. Nothing is ranked, capped or
  rewritten; a trial excluded is listed in `excluded.jsonl` with its exact reason,
  and `report.md` shows the waterfall from raw trials to records. Validation
  holds out whole tasks (or task families). The target is trained for no-think
  mode unless the dataset says `thinking: true`: the reasoning a teacher's replies
  carry is then removed from every turn (the report counts them), and the trainer
  is told the same mode. `projection: portable-sft` rebuilds
  each conversation from the trial's ATIF trajectory instead, for an agent whose
  model calls are not recorded (Claude Code).

  `objective: preference`, `objective: feedback` and `objective:
  policy-gradient` project the same
  attempts by splinter-data's own views, the ones that read the experience
  database, over the trials as experiences (each trial's grade its executable
  verdict, its recorded model calls its conversation). A preference record is
  a verified pass over a verified fail of one task by one policy, from where
  their conversations part: the shared prompt is never trained on
  (`preference: { pairs_per_task: N }` bounds a task's pairs, drawn by the
  seed;
  `max_block` leaves out the pairs whose longer continuation exceeds it, as the trainer
  sizes every row to the longest). A feedback record (KTO) is one verified attempt's continuation,
  desirable when it passed. `objective: distill` makes every attempt a self-distillation
  record: its conversation as the student saw it, and beside it what only the
  frozen teacher reads (the task's reference solution, `solution/solve.sh` of
  its folder, and the checks that failed); it trains by the policy-gradient
  job. `objective: retrieval` makes contrastive triples:
  a passed experience's question, the source passage its evidence names, and
  the passages other experiences rest on as negatives. These objectives also
  read `answers`: the experiences `answer.collect` made in process, each with
  its verdicts and the sources its evidence is in. An SFT dataset of answers
  (and no attempts) supervises each passed answer's final reply under its
  question. A policy-gradient record is an action of a graded attempt carrying
  the attempt's reward and its group; a group whose attempts all earned the
  same is left out. Hinted, continued and harnessed attempts are refused: they
  would teach what the target is never shown. `model.train` trains each
  dataset by its objective's Brain job: chat and text records by supervised
  fine-tuning, pairs by DPO against the base, judged completions by KTO (the
  smaller class weighed up), rollout records by the clipped policy gradient.
  A retrieval dataset trains an embedder and is refused as its input.

  ```yaml
  - id: preferred
    uses: dataset.project
    inputs: { attempts: [$stages.local.attempts] }
    with: { objective: preference, preference: { pairs_per_task: 4 }, target: student, seed: 13 }
  ```
- **Texts** (`dataset.text`). Continued pretraining on a directory of the task
  repository: each file's text is one record by splinter-data's CPT view (a
  text two files share is trained on once, a file that is not text or is
  empty is left out and counted), one supervised turn, so it trains like a
  chat dataset; `validation: { percent: N }` holds out whole documents, drawn
  by the seed.
- **RLVR tasks** (`dataset.rlvr`). An online trainer rolls a task out again and
  grades each rollout by the task's own verifier; a task the policy always or
  never solves gives it no gradient. `dataset.rlvr` keeps the pinned tasks whose
  counted attempts (at least `min_attempts`, 2) passed only sometimes (a pass
  rate strictly inside 0 to 1, or inside `min_pass_rate`..`max_pass_rate`),
  each task's folder as the run pinned it, for `tasks.pin`; `rlvr.json` lists
  every attempted task's counts, rate and whether it was kept.
- **Replay** (`model.train` with `replay`). A candidate trained on new chat records can rehearse
  earlier datasets' records beside them, `replay_share` of the draws (the plain union when
  unset), so what it learned before is kept: `inputs: { dataset: $stages.sft.dataset, replay:
  [$stages.earlier.dataset] }`. Replay rehearses chat records, so a view objective's dataset
  takes none and is none.
- **Training** (`model.train`). Each candidate is one brain job of its dataset's regime, resumable under a key derived from
  its identity; a candidate counts only once brain's server lists it. Changing a
  stage's parameters trains a new candidate and reuses the dataset and attempts.
- **Evaluation** (`evaluation`). A development plan runs a baseline arm (once,
  shared by every evaluation with the same plan and baseline) and a candidate arm under the same agent, arguments, environment,
  tasks, attempts and seeds, and decides by the exact one-sided paired sign test
  at the plan's alpha divided among the stage's `comparisons`. A plan's
  `rerun_failed: N` gives a task an arm failed up to N more attempts of that task
  alone (one attempt of every task first, nothing more for a task that passed); a
  task counts as solved when any of its attempts passed, both arms get the same
  chances, and the sign test runs over tasks. A protected plan names the
  task repository's examiner instead of tasks, whose sealed tasks never leave it:

  ```yaml
  schema: splinter.evalplan/v1
  attempts: 1
  seed: 7
  decision: { alpha: 0.05 }
  authority: { operator: exam.sealed, public_key: config/verdict.pub }
  ```

  The examiner is one of the repository's operators: it is given the candidate
  (`candidate/candidate.json`, the adapter and its digest), the trained
  candidate's files (`model`) and, against a champion, the predecessor, all
  read-only, and writes `verdict.json`; what it needs of the machine (the brain
  server, a key, the sealed tasks) is that machine's grant. An authority may
  instead name a `command`, to which Splinter appends `--candidate FILE --out
  FILE`. Either way the verdict counts only when its signature verifies under
  the public key and it is about this candidate against the base at the plan's
  level.
- **Release** (`promotion`). Records every decision and the gates each candidate met; with
  `promote: true` it promotes the best candidate that meets them all. Each evaluation states its
  `purpose`: `target` (the default: what it was trained for, must be `improved`), `transfer`
  (tasks it was not trained on, must be `improved`) and `retention` (what the baseline could
  already do: must not regress, neither a loss the plan's sign test finds significant nor a
  pass rate more than `retention_margin`, 0.05, below the baseline's). A promotion requires
  an evaluation of every purpose; `requires: [target]` narrows that in plain sight, where a
  recipe has no transfer or retention suite. Evaluations of one baseline share its arm only
  when they run the same plan on the same tasks. Each release records a `lineage`, one point per
  generation of the champion (its target and transfer pass rates), carried from the champion's
  own release; a candidate whose point shows the target climbing over the last
  `divergence_window` generations (3) while transfer stays flat trips the divergence alarm and is
  not promoted: it is learning to please its verifier, not the task. It names the champion the run leaves in use
  (below).

  ```yaml
  - id: eval-retention
    uses: evaluation
    inputs: { candidate: $stages.lora-r16.model, tasks: $stages.anchors.tasks }
    with: { plan: eval/anchors.yaml, agent: sven, baseline: student, purpose: retention }
  - id: release
    uses: promotion
    inputs: { evaluations: [$stages.eval-target.evaluation, $stages.eval-transfer.evaluation,
                            $stages.eval-retention.evaluation] }
    with: { promote: true }
  ```

## Changing an experiment

A run refuses to continue on inputs that changed since it started. Change the
recipe and fork the run: every product the change leaves alone is reused.

```sh
splinter fork runs/vim-mastery --recipe recipes/vim-mastery-r32.yaml --out runs/vim-mastery-r32
splinter eval runs/vim-mastery --plan eval/vim-sealed.yaml --out runs/vim-mastery-sealed
```

## Importing Harbor jobs

Trials an agent already ran elsewhere - a Harbor job of a benchmark, a teacher run from before the factory - join a
recipe through `attempt.import`, a stage that runs nothing. Its `jobs` are recipe inputs naming Harbor job
directories in the task repository; its product is attempts that `dataset.project`, `unsolved_by` and `splinter
index` read like any attempt stage's.

```yaml
inputs:
  course: { kind: files, path: collections/vim-headless-mastery.yaml }
  deepseek-job: { kind: files, path: jobs/20261122-unsolved-deepseek-v4.1-flash-k4 }
stages:
  - id: tasks
    uses: tasks.pin
    inputs: { collection: $inputs.course }
    with: { split: train }
  - id: deepseek-unsolved
    uses: attempt.import
    inputs: { tasks: $stages.tasks.tasks, jobs: [$inputs.deepseek-job] }
    with: { model_label: deepseek/deepseek-v4.1-flash }   # optional; tasks: [ids] narrows it
  - id: teacher
    uses: attempt.collect
    inputs: { tasks: $stages.tasks.tasks, unsolved_by: [$stages.deepseek-unsolved.attempts] }
    with: { agent: sven, model: teacher, environment: environments/vim/environment.yaml,
            attempts: 4, seed: 2 }
```

- **Matching.** The execution worker matches every trial to a task the stage's `tasks` pin by Harbor's own content
  digest of the task directory (the trial's `lock.json`, else its `task_checksum`). A trial of a task that changed
  since it ran, of a task the pin does not hold (another split, examination material included), or one Harbor never
  finished is refused and listed in the stage's `refused.jsonl` with its reason; `report.md` counts them.
- **Grading.** Each trial is reported as a trial the worker ran would be and graded by the one classifier, so a
  trial the infrastructure decided is no verdict, never a 0. A task's trials fill its slots in the order of the jobs,
  then by trial name; every trial of a job is kept.
- **Pinning.** Each job is part of the run's lock by the digest of its whole tree: a trial added to a job is a
  changed input, so fork the run (above). An import has no environment, agent or seed of its own.
- **The model** is recorded as `model_label` states it (not a role: nothing is served). Harbor records the agent's
  own model name, which may be a gateway's alias (`bench/sven-hosted-sft`); name the model that answered when you
  know it.

**Telling records apart.** Every record `dataset.project` writes carries `metadata.trial`: the stage that holds its
trial, the slot, the agent and model, `guided` when a person guided it, `withheld` (what the target was produced
with that the record leaves out: `hint:solution`, `hint:brief`, `continuation`), and for an imported trial `imported:
{harbor_job, trial}`. Records of one dataset can be filtered or weighed by their source from the JSONL alone.

## Hinted passes

A hinted pass rationalises what the model could not solve unaided: it attempts
the tasks earlier passes left unsolved with their reference solution in front
of it, and each verified success is learned under the task's own instruction,
the hint nowhere in the record:

```yaml
  - id: hinted
    uses: attempt.collect
    inputs: { tasks: $stages.tasks.tasks, unsolved_by: [$stages.local.attempts] }
    with: { agent: sven, model: student, hint: solution,
            environment: environments/vim/environment.yaml, attempts: 2, seed: 5 }
```

A trial whose own replies or reasoning repeat a line of the solution learned
to quote the hint, not to do the task, and is excluded (`hint_repeated`); so is
a record that still shows the hint (`hint_not_removed`) or never held it
(`hint_not_found`). `supervision.jsonl` marks every record of a hinted pass
`hinted`. A hinted stage is never interactive, and every task it runs needs a
reference solution.

## Corrections

Every repair of what the student could not solve is an attempt stage over the
tasks earlier stages left unsolved, differing only in what the actor is shown.
A retry is a stage with another seed or model. A hinted pass (above) shows the
reference solution. A briefed pass (`hint: brief`) shows each task a brief that
an earlier stage wrote for it - knowledge passages, a plan of subgoals, notes
on the mistake - and learns what it then solves under the task's own
instruction, excluding a trial that repeats its brief, as a hinted trial that
quotes its solution is excluded:

```yaml
  - id: failures
    uses: attempts.failures
    inputs: { tasks: $stages.tasks.tasks, attempts: [$stages.local.attempts] }
  - id: briefs
    uses: agent
    foreach: $stages.failures.failures
    inputs: { input: $item }
    with: { instruction: prompts/plan-subgoals.md, output_schema: schemas/brief.json,
            model: teacher }
  - id: corrected
    uses: attempt.collect
    inputs: { tasks: $stages.tasks.tasks, unsolved_by: [$stages.local.attempts],
              briefs: $stages.briefs.output }
    with: { agent: sven, model: student, hint: brief,
            environment: environments/vim/environment.yaml, attempts: 2, seed: 7 }
```

`attempts.failures` collects every task some given stage attempted and none
solved, each with its instruction and its last failed attempt condensed for a
reader (`attempt.md`); its items are named by task id with `/` written `--`. The
brief of a task is the `brief` text of that item's `output.json`, which an
`agent` stage asked for an object `{ "brief": ... }` writes. The teacher's
instruction decides what kind of correction it is: passages from reference
material (knowledge-backed), a numbered plan of checkable subgoals
(decomposition), or what went wrong.

A fork correction continues each failed attempt instead of starting the task
afresh. A teacher stage over `attempts.failures` names, per task, the last step
to keep (`keep_through`, a step id of `attempt.md`) and a sentence of
`guidance`; the attempt stage given `continue_from` runs, for each task that
stage attempted and never solved, its last failed attempt cut after that step:
the kept steps' tool calls run again in the fresh sandbox, so the workspace is
as they left it, and the model goes on after the guidance. What it then solves
is learned as one conversation, without the guidance turn, and a continuation
that repeats its guidance is excluded (`guidance_repeated`):

```yaml
  - id: located
    uses: agent
    foreach: $stages.failures.failures
    inputs: { input: $item }
    with: { instruction: prompts/locate.md, output_schema: schemas/fork-point.json,
            model: teacher }
  - id: forked
    uses: attempt.collect
    inputs: { tasks: $stages.tasks.tasks, continue_from: $stages.local.attempts,
              briefs: $stages.located.output }
    with: { agent: sven, model: teacher, environment: environments/vim/environment.yaml,
            attempts: 1, seed: 8 }
```

A solution hint is trusted only where the solution works: an attempt stage
with `agent: oracle` runs each task's reference solution against its own
verifier, and a hinted stage given `solved_by: [$stages.oracle.attempts]` runs
only the tasks the oracle solved.

## Generations

A campaign of generations is a chain of forked runs, each held to the champion
before it. A run's release names the champion it leaves in use: the candidate
it promoted, else the model it was measured against. A trained champion's
adapter is kept in the release product. The next generation's recipe is the
last one with a `champion` model role, one more attempt stage in which the
champion attempts what no earlier stage solved, and the champion as the
baseline to beat:

```yaml
models:
  student: "brain:Qwen/Qwen3-4B"
  teacher: "hosted:deepseek/deepseek-v4.1-flash"
  champion: "champion:runs/vim-g1"
stages:
  # ... every stage of the generation before, unchanged, so the fork reuses them ...
  - id: g2
    uses: attempt.collect
    inputs: { tasks: $stages.tasks.tasks,
              unsolved_by: [$stages.local.attempts, $stages.hosted.attempts] }
    with: { agent: sven, model: champion, environment: environments/vim/environment.yaml,
            attempts: 4, seed: 21 }
  - id: sft
    uses: dataset.project
    inputs: { attempts: [$stages.local.attempts, $stages.hosted.attempts, $stages.g2.attempts] }
    with: { target: student, seed: 3 }
  # ... model.train from base: student, evaluation with baseline: champion ...
  - id: release
    uses: promotion
    inputs: { evaluations: [$stages.eval-r16.evaluation] }
    with: { promote: true, requires: [target] }
```

A candidate may also *continue* the champion's adapter: `model.train` with
`start: champion` begins from the champion's weights rather than from the base,
which is how a preference objective (`loss`, `beta`, `nll_weight`) refines an
adapter that already works. The champion must be served from the candidate's
base model.

```sh
splinter fork runs/vim-g1 --recipe recipes/vim-g2.yaml --out runs/vim-g2
```

`champion:RUN` names the run directory, relative to the task repository or
absolute. The champion is served from its base and adapter wherever an agent's
model role goes, and it is never trained from (a `model.train` base or a
dataset target that is the champion is refused): every variant trains from the base
model on the data of every generation so far. A protected plan hands the
champion to its examiner as the predecessor (`--predecessor FILE`), and keeps
the verdict only when it is about that predecessor. A changed champion is a
changed input like any other.

`splinter index RUN` files a run's trials in the experience database, an
optional index across runs (each trial an experience with the factory's grade as
its verdict; indexing again files nothing new); the run's products stay its record.

`splinter cancel RUN` stops a run at its next stage or batch boundary;
`splinter resume RUN` continues it. One controller holds a run at a time, on a
lease it renews while it lives, however long a stage takes; a resume takes the
lease of a controller that crashed at once when it can prove it gone (the same
host and pid namespace, its process no longer running), and otherwise when the
lease lapses (ten minutes). `splinter export RUN --out run.tar.zst` writes
a bundle that `splinter verify` checks on another machine.

## Guided sessions

An attempt stage with `interactive: true` is never run unattended. `splinter play` starts
one of its tasks on this terminal; a person works with the agent, the finished
session is graded like any trial and marked guided, and the run continues, so the
session reaches the dataset and what is trained on it:

```sh
splinter play runs/vim-hosted-sft --stage guided --task train/019-move-backward-by-word
splinter play runs/vim-hosted-sft --stage guided --task train/019-move-backward-by-word --continue 0
```

`--continue N` continues session N's conversation as a new session; the earlier
one is left as it is.
