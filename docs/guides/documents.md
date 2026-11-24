<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Learn from documents, directories and commands

Use this guide to teach the policy what a document, a directory of text or a
command-line tool says, then check that it learned it. The run is a recipe of
the [learning factory](learning-factory.md); `splinter inspect RUN --stage STAGE` shows
what one of its stages made, and `splinter resume` runs a stopped run again without repeating a finished stage.

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

## The stages of the recipe

| Stage | Command or operation | Reads | Writes |
|---|---|---|---|
| sources | `source add` | a file, a directory, a command's run | a source |
| sessions | `session add` | ATIF files of a person's sessions with an agent | session sources, secrets removed |
| claims | `claims extract` | session sources | a claim set: what a model proposes the person taught, with their exact words |
| gates | `claims gate` | a claim set | the ledger: every proposal admitted, refused (with the reason), or superseding earlier claims |
| absorb | `absorb` | a person's sessions with an agent | the above for sessions, then kits of records per claim, a candidate trained again from the base and a release the declared count gate opens |
| tasks | `task.generate` | documents | tasks that quote their evidence, and `rejected.jsonl` with the reason for each refusal |
| answers | `answer.collect` | a task | k graded answers per task, each graded by its task kind's verifiers |
| dataset | `dataset.project` | answers | a dataset and its manifest |
| candidate | `model.train` | a dataset | a candidate adapter (not released) |
| evaluation | `evaluation` | a candidate | the paired comparison with its baseline |
| release | `promotion` | evaluations | every decision and the gates met; the best candidate promoted when `promote: true` |

Other commands act on what the stages store under the state root: `tasks variants` makes a task set of the same
facts asked in other words, each recording the task it varies, which the release gate measures and nothing trains on;
`critique` makes an experience set of critiques and revisions. A task set is made from the text parts of the sources,
and every question answered from a source must name its subject, which the source's identity or a cited section must
name; two tasks of a set that ask one question of one subject with different answers are both left out. How a stage
selects, projects and trains is in the [learning factory](learning-factory.md#what-each-stage-guarantees).

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
