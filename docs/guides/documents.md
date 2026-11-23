<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Learn from documents, directories and commands

Use this guide to teach the policy what a document, a directory of text or a
command-line tool says, then check that it learned it. The run is a recipe of
the [learning factory](learning-factory.md); the stages it is made of are
also commands of their own, which is how you debug or rerun a part of it.

The worked example is [your first learning
run](../getting-started/first-learning-run.md). Every option is in the [command
reference](../reference/cli.md).

## Start from the template

```bash
splinter init --template knowledge-learning my-repo
cd my-repo
cp ./manual.md documents/
splinter validate recipes/knowledge-learning.yaml
splinter run recipes/knowledge-learning.yaml --out runs/knowledge
```

The `knowledge-learning` recipe has a teacher model propose closed-book tasks
from each document, admits only those whose evidence quotes the document
verbatim, has the student answer each one several times, grades every answer
by the task kind's verifiers and trains a LoRA on the passed answers. The
recipe is yours to change: replace `documents/guide.md` with your own
documents, and name the models in `models:`. The [learning
factory](learning-factory.md) says what a run needs on the machine and what each
operation does.

### Sources

The stages below that read a source take these:

| You point it at | What is captured |
|---|---|
| A file | A document, split into addressable parts and sections |
| A directory | A repository source: the files under it, at the revision when it is a git work tree |
| `cmd:COMMAND ...` | The command's real output, run without a shell, with the command, working directory, exit code and truncation recorded as the source's origin |
| A stored source id | The source as captured earlier (`splinter source list`) |

`source add` stores a source. URLs are refused. The same bytes are
the same source, so capturing twice stores once. `--usage-policy` records the terms the
content is under (`redistributable`, `research_only`, `noncommercial`, `restricted_DUA` or
`unknown`); a release is restricted to local use unless every input's terms allow
distribution, as [Releases](releases.md) explains.

```bash
splinter source add ./manual.md --usage-policy research_only
splinter source add "cmd:brain --help"
splinter source list
```

## The stages, one command each

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

Each stage reads what the one before it stored under the state root and
writes its own result, so any of them can be rerun or inspected alone. A task
set is made from the text parts of the sources, and every question answered
from a source must name its subject, which the source's identity or a cited
section must name; two tasks of a set that ask one question of one subject
with different answers are both left out. `solve --frontier` solves each task
k times (`--k`), grades every attempt by its task kind's verifiers (the judged
kind by the judge `verify --judge` names), has a teacher solve open-book each
task no attempt solved, and keeps the tasks worth training on: those the
policy fails at least sometimes and has a verified answer to. `tasks variants`
asks the same facts in other words, which the release gate measures and
nothing trains on. `dataset build` caps near duplicates and gives a dataset
whose records are the passing attempts, the verified teacher answers and the
revisions `critique` made; `train` trains a candidate on it from the champion
and `release` lets the gate decide.

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

## Say it in a sentence

`splinter "absorb the sessions in ./today"` asks the policy model, through
sven's typed method call, which command the sentence means; the reply is
a list of candidate intents (`absorb`, `ask`, `status`, `add_source`,
`list_sources`, `list_runs`, `cancel_run`), each with its arguments and a
confidence. Code decides: a single reading at confidence 0.7 or more, and
0.2 ahead of any other, runs as its command (printed on stderr first).
Anything less is asked back - in the REPL at a terminal you pick one; with
`--json` or a sentence argument the candidates are printed and the exit is
3. A remote model without the opt-in is refused; cancelling a run, running
a command (`cmd:`) and using a remote model are never done on a guess.
A path the model copied wrongly is replaced, by code, by the one path the
sentence names that exists, and never by a guess. Sentences are read by the assistant when there is one, since a
larger model reads them better; `--front-door-model` names another.
