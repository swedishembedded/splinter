<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Learn from documents, directories and commands

Use this guide to teach the policy what a document, a directory of text or a
command-line tool says, then check that it learned it. Everything here works on
`splinter learn`; the sections after the first show the same pipeline one stage
at a time, which is how you debug or rerun a part of it.

The worked example with expected output is [your first learning
run](../getting-started/first-learning-run.md). Every option is in the [command
reference](../reference/cli.md).

## One command

```bash
splinter learn ./manual.md --goal "the console and power limits" --kinds denoise --budget 40m
```

`learn` takes files, directories, `cmd:<command>` (quoted as one argument) or ids of
sources already stored. `--dry-run` prints the plan and writes nothing under the state
root. `--goal` steers which tasks are written. `--kinds` names the task kinds (see
[tasks](../reference/tasks.md)); without it a planner model chooses them, as [Learn how
a person thinks](personas.md) describes. `--budget` bounds the run's wall-clock time.
`--no-release` stops at the trained candidate instead of putting it through the
[release gate](../concepts/release-policy.md).

`learn --adapter NAME` keeps the trained adapter in the model store as the
adapter `NAME` of the base (`--base CHECKPOINT`, a store name or a path; the
configured base by default) instead of putting it through the release gate, so
`local:<base>+NAME` selects it anywhere a model is named. Running the same name
again continues that adapter from its own factors; a new name starts one.

### Sources

| You point it at | What is captured |
|---|---|
| A file | A document, split into addressable parts and sections |
| A directory | A repository source: the files under it, at the revision when it is a git work tree |
| `cmd:COMMAND ...` | The command's real output, run without a shell, with the command, working directory, exit code and truncation recorded as the source's origin |
| A stored source id | The source as captured earlier (`splinter source list`) |

`source add` stores a source without learning from it. URLs are refused. The same bytes are
the same source, so capturing twice stores once. `--usage-policy` records the terms the
content is under (`redistributable`, `research_only`, `noncommercial`, `restricted_DUA` or
`unknown`); a release is restricted to local use unless every input's terms allow
distribution, as [Releases](releases.md) explains.

```bash
splinter source add ./manual.md --usage-policy research_only
splinter source add "cmd:brain --help"
splinter source list
```

## What a run does

### The stages

| Stage | Command | Reads | Writes |
|---|---|---|---|
| sources | `source add` | a file, a directory, a command's run | a source |
| sessions | `session add` | ATIF files of a person's sessions with an agent | session sources, secrets removed |
| claims | `claims extract` | session sources | a claim set: what a model proposes the person taught, with their exact words |
| gates | `claims gate` | a claim set | the ledger: every proposal admitted, refused (with the reason), or superseding earlier claims |
| absorb | `absorb` | a person's sessions with an agent | the above for sessions, then kits of records per claim, a candidate trained again from the base and a release the declared count gate opens |
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
every attempt is graded by its task kind's verifiers (the judged one
by the judge the command names: `verify --judge`, or, in a `learn`, its judge
role); each task no graded attempt solved is solved
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
four fifths, and training and the exam then run to the end. When a share of the
training tokens is the writer's own text (`--voice`), training has that much more to do
and nothing model-made to wait for, so the tail grows by it - half the share
doubles it, to two fifths - and the open-ended stages give up their share in
proportion, never below a fifth of the budget between them. The tasks stage
reports how many of the sources' text parts it generated from, how long it ran,
and, when it stopped before every part, what covering them all would take at that
pace, so a run over a few per cent of a corpus is never read as a run over the
corpus; `--goal` steers what tasks are asked for; `--dry-run`
prints the plan and writes nothing. A `learn` that stops before training
(nothing admitted, no task worth training on, nothing passed, the budget
spent) or whose candidate the gate blocks says why and exits 1.

## Learn a documented domain

```bash
make deps/lua LUA_DIR=./lua51     # once: the pinned Lua 5.1 interpreter
splinter --runtime-dir ./lua51/bin learn "Become an exceptional Lua 5.1 programmer" \
    --source https://www.lua.org/manual/5.1/manual.html --kinds construct-lua,predict-lua
```

With `--source` the words of the command are the objective and each source (a URL, a file or a
directory) is read into a graph of skills, with the constraints and requirements the text states
and the prerequisites between them. A URL is fetched once; an HTML page, fetched or a local file,
is decoded in the charset it declares (UTF-8, or Latin-1/windows-1252, the default for a page that
declares none) and read as Markdown; the record ties the URL (`file://` and the path for a local
page) to the digest of the bytes and to the source. Each skill has probes
written from its own sections, split into dev probes (the baseline, a round's gain and what a
round cost are measured on them) and final probes (only the closing assessment touches them); no
training task may repeat a probe. Each time the generator is asked again for a skill it is shown
the instructions already written for it (the newest 12, each cut to 240 characters) and told to
write different tasks, and a round's task request lists the skill's probes, so repeats are mostly
not written instead of refused afterwards. Rounds train the skill expected to benefit most,
prerequisites first; a candidate is accepted when its skill improved on the dev probes and no skill measured
before fell by more than `--retention-drop`; the competence graph is kept revision by revision.
The run ends when every skill is mastered (`--mastery`), no skill can be chosen (a skill has
`--max-rounds-per-skill`), `--max-rounds` or `--budget` is spent, and is accepted only if every
assessed skill reaches the target on the final probes. `--no-release` stops before the gate;
otherwise the final candidate is gated like any other and only the gate moves the alias.

Each round's report in the run record (`splinter --json learn ...`, stage `round-N`) says who
did what and what it cost: `incumbent` (the model that attempted the tasks) and `generator` (the
model that wrote them), then `proposed`, `written` (admitted), `attempts`, `graded`,
`verified_passes`, `worth_training`, `records`, and `solve` and `train` with `wall_seconds` (and
`gpu_seconds` where the attempts recorded device time; training reports wall-clock only). From
them the useful verified experiences per unit of exploration compute are computed per round.

### Training every weight, and a model that writes its own tasks

By default each round trains a LoRA adapter on the start policy. `--full-fine-tune` trains every
weight instead: each round starts from the incumbent checkpoint (the start base in round 1, the
last accepted round's checkpoint after) on the round's records beside a replay of the accepted
rounds' records (`--replay-fraction`, default 0.25 of each earlier dataset's trained-on records,
taking a quarter of the training draws). The candidate is a checkpoint directory, measured as
`local:<directory>`; an accepted one becomes the incumbent, and a rejected or superseded one is
deleted unless `--keep-candidates`. The learning rate is `--lr`, else 1e-5. `--train-precision
<SPEC>` (for example `state=int8,sr=on`) is handed to brain's parser unchanged; the default is
brain's. The release gate takes adapters, so `--full-fine-tune` needs `--no-release`, and `--rank`
does not apply. Training holds the card for itself: the resident policy is released first.

`--self-generate` lets the incumbent write the training tasks of every round: the policy the run
began with until a round is accepted, the accepted candidate after. The probes are written once,
before any training, by the configured generator in either case, so the exam does not depend on
the trained model. `--reference-teacher` teaches a task the policy never solves with the task's own reference answer -
for a code-output kind, the output the runtime computed - said as the teacher's reply and graded
like any answer, so no model teaches. `--structural-graph` reads the skill graph from the sources' structure, one
skill per section, without asking a planner: a small model rarely writes the planner's reply.

The Lua families and the `lua` runtime are Lua 5.1 and no other: `make deps/lua LUA_DIR=<dir>`
builds the pinned interpreter, `--runtime-dir <dir>/bin` puts it first, and the runtime refuses
an interpreter that reports another version.

## Ask the learned model

`ask` records every answer under `<state>/answers/` with the model that
gave it and, asked through `policy:<alias>`, the release the alias
resolved to; its report carries the answer's `id` and that `release`. It answers closed-book unless
you show it a source.

Passages are built, not found: a part's paragraphs are merged forward until a
passage holds about 120 words (a header goes with what it heads, a signature with
what precedes it), a paragraph over 260 words is split at sentence ends, and a
fragment too short to say anything is left out, so an embedding is of something
with a referent and none is cut off by the embedder.

`--open-book` shows the model the whole text of one source. `--retrieve`
shows it instead the `--passages` (default 6) passages of the named sources
that bear on the question: ranked by meaning (Qwen3-Embedding), with a
fixed share of the tail given to the passages only the question's exact
words find (a name, a date), each under its part and section. With `--reranker REF` a model
reads the six candidates found for each passage shown, beside the question,
and the ones it says bear on it come first: search finds passages about the
same things as the question, and only a reader of both says whether one
bears on it. The answer
records the sources and the passages shown (part, section and how each
begins; printed under the answer, and in `--json` as `shown`), and
`splinter lineage` links it to the sources. The passages'
vectors are made once and kept in the state as derived data (an artifact found
by a pointer named for the embedding model and the passages' text), so other
sources, changed text or another model is another index and a second question
over the same sources embeds nothing.

## Read a run's report

A `learn` report opens with an `outcome` block of five separate facts: whether the
run finished or stopped, whether a candidate was trained, what the release gate decided
(naming the checks that blocked it), whether a release was written, and whether
`policy:default` moved. A trained candidate the gate blocked is reported as exactly
that, and the policy as unchanged.

## Say it in a sentence

`splinter "learn the manual in ./docs"` asks the policy model, through
sven's typed method call, which command the sentence means; the reply is
a list of candidate intents (`learn`, `absorb`, `ask`, `status`, `add_source`,
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
by a guess. Sentences are read by the assistant when there is one, since a
larger model reads them better; `--front-door-model` names another.
