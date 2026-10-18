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
splinter learn <SOURCE>... [--goal TEXT] [--adapter NAME [--base CHECKPOINT]] [--kinds K,.. | --planner REF] [--budget DUR] [--dry-run] [--no-release]
                         [--no-frontier | --distill | --k N [--temperature T] [--top-k N]] [--teacher REF] [--generator REF] [--judge REF]
                         [--steps N] [--rank R] [--alpha A] [--lr LR] [--weight-decay WD] [--records-per-step N] [--seed N] [--eval-every N] [--patience N] [--monitor-share SHARE]
                         [--with-passages SHARE [--abstain SHARE]] [--voice SHARE [--describe-voice]] [--rehearsal SHARE] [--select-on-dev] [--bf16-base]
                         [--exam-seed N] [--interval-seed N] [--rehearsal-seed N] [--denoise-seed N]
splinter absorb <SESSIONS>... [--policy REF] [--generator REF] [--teacher REF] [--judge REF] [--paraphrases N] [--rehearsal-share F] [--rehearsal-seed N]
                         [--epochs N | --steps N] [--rank R] [--continue-from-release [--replay-fraction F]] [--sealed-probes FILE...]
                         [--stance-pairs N [--stance-beta B] [--stance-tries N] | --no-stance] [--dry-run] [--no-release]
                         [--consolidate-after-records N | --no-consolidate] [--consolidate-mode full|merge] [--retention-replay-share SHARE]
splinter consolidate [--policy REF] [--mode full|merge] [--from RELEASE] [--epochs N | --steps N] [--lr LR] [--rehearsal-share F] [--rehearsal-seed N]
                         [--retention-replay-share SHARE] [--sealed-probes FILE...] [--no-release]
splinter ask <QUESTION> [--open-book SOURCE-ID | --retrieve SOURCE-ID... [--passages N] [--reranker REF]] [--policy REF]
splinter status
splinter doctor
splinter source add <PATH|cmd:COMMAND...> | list | show <ID>
splinter session add <PATH>... [--max-bytes N] | list
splinter claims extract <SESSION-ID>... [--generator REF] | gate <CLAIMSET-ID> | list | show <ID> | ledger
splinter tasks generate <SOURCE-ID>... --kinds K,.. [--generator REF] [--author NAME] [--denoise-seed N] | variants <TASKSET-ID> [--generator REF] [--per-task N] | list | show <ID>
splinter solve <TASKSET-ID> [--solver REF] [--frontier [--k N] [--temperature T] [--top-k N] [--teacher REF]]
splinter verify <EXPERIENCE-SET> [--judge REF]
splinter critique <EXPERIENCE-SET> [--critic REF] [--retry N]
splinter judge calibrate <LABELLED-FILE> --judge REF | measure <TASKSET-ID> --judge REF [--fit]
splinter experiences list | show <ID> [--graph] | replay <ID>
splinter dataset build <EXPERIENCE-SET>... --view VIEW [--strip all|keep:K,..|mix:F [--mix-seed N]]
                       [--writer NAME] [--token-limit N] [--max-family-share SHARE]
                       [--min-strength executable|formal|consistency|judged] [--export-only]
                       [--system-prompt TEXT] [--limit N] [--pack]
splinter dataset export <DATASET-ID> --out DIR
splinter rehearse --records N [--rehearsal-seed N]
splinter train <DATASET-ID>... [--from REF] [--replay-fraction F] [--rehearsal DATASET-ID [--rehearsal-share F]] [--steps N] [--rank R] [--beta B]
                                [--alpha A] [--lr LR] [--weight-decay WD] [--records-per-step N] [--seed N] [--eval-every N] [--patience N] [--monitor-share SHARE] [--keep-evaluations]
splinter select <CANDIDATE-ID> --exam-set DEV-SUITE [--judge REF] [--adopt]
splinter release <CANDIDATE-ID> [--alias NAME] [--judge REF] [--interval-seed N] | list
splinter rollback <ALIAS>
splinter eval [REF] [--suite held-out|retention|anchor|FILE] [--freeze FILE]... [--judge REF]
splinter exam CANDIDATE [--interval-seed N] [--judge REF] [--prompt GOAL] [--retrieve SOURCE-ID... [--passages N] [--reranker REF]]
splinter exam CANDIDATE --exam-set EXAM [--resamples N] [--no-voice] [--deployed-only] [--pilot-families N] [--judge REF] [--prompt GOAL]
splinter exam-set create SOURCE... [--families N] [--tasks-per-family N] [--dev-families N] [--dev-tasks-per-family N] [--kinds K,..] [--generator REF] [--goal TEXT] [--persona NAME] [--seed N] [--not-trained-by CANDIDATE]... | power [--families N] [--tasks-per-family N] [--effect SHARE] [--discordance SHARE] [--icc RHO] [--from-report FILE] [--seed N] | show EXAM
splinter exam-report labels-export REPORT --out FILE --key FILE [--n N] [--seed N] | labels-import LABELS --key FILE | memorisation REPORT --candidate CANDIDATE
splinter runs list | show <ID> | cancel <ID>
splinter lineage <ID> [--up|--down|--both] [--depth N]

global: --state DIR  --json  -v  --allow-remote  --policy-context-tokens N
```

`learn --adapter NAME` keeps the trained adapter in the model store as the
adapter `NAME` of the base (`--base CHECKPOINT`, a store name or a path; the
configured base by default) instead of putting it through the release gate, so
`local:<base>+NAME` selects it anywhere a model is named. Running the same name
again continues that adapter from its own factors; a new name starts one.

An id is the full `blake3:<hex>`, the hex alone, or a prefix of at least
four hex digits that names exactly one stored object.

`ask` records every answer under `<state>/answers/` with the model that
gave it and, asked through `policy:<alias>`, the release the alias
resolved to; its report carries the answer's `id` and that `release`.

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

## Models

Every model is named the same way:

| Reference | Model |
|---|---|
| `policy:default` | the configured base (`BRAIN_QWEN_WEIGHTS`, else `Qwen/Qwen3-0.6B` in brain's model store) with the champion's adapter, the release `default` points at; the base alone before any release |
| `policy:<alias>` | the base with the adapter of the release `<alias>` points at (`release --alias`) |
| `local:<checkpoint>[+<adapter>][@<tokens>]` | a checkpoint path (absolute, or starting `./` or `../`) or a name in brain's model store, with an optional LoRA adapter file after the first `+`. The context is the largest the checkpoint supports (its `max_position_embeddings`); `@<tokens>` limits it, as fitting a large model on a card requires (a 14B judge on a 24 GiB card: `local:Qwen/Qwen3-14B@4096`) |
| `remote:<provider>/<name>` | a model reached over the network through sven's provider configuration |

A policy alias is resolved once per command, when it is first used: a
run keeps the release it started with however the alias moves meanwhile,
and records it (`learn`'s first stage, a candidate's `parent`).

The policy's context is limited with the global `--policy-context-tokens N`
(a policy too large for the card at its maximum), any other local model's by
its reference.

A remote reference is refused unless `--allow-remote` is given or
`SPLINTER_ALLOW_REMOTE=1` is set; that is the only way Splinter uses the
network. OpenRouter models take their key from `AGENT_OPENROUTER_KEY`, any
other provider from `BRAIN_API_KEY`.

## The pipeline

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

## Learning a repository

```bash
splinter practice ./shop --rev v1.4
splinter practice ./shop --rev HEAD --exam-families 2 --k 4 --no-release \
    --test-command python3 -m pytest -x
```

`practice` learns a codebase from its own tests, as one recorded run. The
repository is a git work tree or bare repository and `--rev` (default `HEAD`)
any revision git resolves to a commit; the tracked files of that commit are
archived as the snapshot every task is pinned to (a checked-out revision with
uncommitted changes is refused unless `--allow-dirty`). Its stages, in
order:

| Stage | What it does |
|---|---|
| `policy` | resolves `policy:default` once |
| `snapshot` | archives the revision; the same tree is the same snapshot |
| `profile` | finds how the tests are run (`python3 -m unittest` for a Python project with a `tests` directory, `cargo test --offline` for a Cargo project; `--test-command` and `--file` name them otherwise) and runs the suite twice on a clean copy: a red or flaky baseline stops the run |
| `tasks` | seeds faults by code in the files under test and keeps a task only where the project's own tests catch it every time, the change reversed makes its check pass, and the catching tests live in files of their own, which are taken out of the tree the agent is given; a task the store already holds, or a request that repeats one made for the same file, is not made again, so a second run over the same snapshot ends here |
| `reserve` | sets the exam's families aside (a family is the file a fault was seeded in) by a stable hash of their names, before any attempt; `--exam-families N` (default a quarter), and a family an earlier run reserved stays the exam |
| `solve` | the policy attempts each remaining task `--k` times as an agent in its own copy of the tree, with sven's file, search and shell tools, up to `--follow-ups` more turns when the visible checks stay red and `--max-tool-calls` calls; each attempt draws from a seed that is a function of the task and the attempt |
| `verify` | applies each patch to a clean copy, lays the hidden tests over it and runs the project's checks; an exposure audit fails any attempt whose trajectory names a hidden path, quotes the hidden tests or names the state root, whatever its patch did, and the stage counts those by kind |
| `frontier` | keeps the tasks the policy solves sometimes (pass@k) |
| `dataset` | the passing attempts, action by action (`sft-step`), with the tools the agent was offered; `--max-block TOKENS` leaves out and counts records longer than the card fits; the dataset is checked to hold no family of the exam |
| `train` | a candidate from the champion with earlier releases' records replayed |
| `release` | the gate, over the exam this run reserved (`--no-release` stops at the candidate): the champion and the candidate attempt the exam's tasks as agents, under the bounds and seeds of `solve`, graded as `verify` grades; paired outcomes are counted by family in the sign test; each earlier release is held to the exam of its own run; the anchor suite is held (a frozen suite with no coding-agent task is decided on the tasks it holds, and the report says it holds none); the candidate is served by `brain serve --adapter-manifest` and must be graded as it was in-process on at least three quarters of a sample of the exam. A blocked candidate is not released and the run stops with the reason of each check that failed; `practice` gives the gate the bounds it solved under, and `splinter release` on such a candidate probes with the default ones |

Where no container is configured, the commands the agent works with run on
the host, bounded but not isolated, and every environment record says
`isolation: none`; naming a docker client and a digest-pinned image
(`SPLINTER_DOCKER`, `SPLINTER_WORKSPACE_IMAGE`) runs them in a container with
no network instead. `SPLINTER_CARGO_HOME` gives the commands a
cargo home to build offline from, and `SPLINTER_WORKSPACE_SCRATCH` says where
trees are laid out. An attempt whose trajectory shows the hidden tests,
and an attempt the harness could not carry out, are set aside and counted in the
gate's report; neither is in a denominator. A release moves the alias and
rewrites `<state>/serving/<alias>.json`, the manifest `brain serve
--adapter-manifest` follows (a rollback repoints it); an alias at a full
checkpoint has no adapter to name, so its manifest is withdrawn. A candidate the
gate blocks, or cannot measure, is not released, and the run exits 1 with the
reason. `--test-command` takes the rest of the command line, so it comes last.

On the host the kernel confines what the agent runs: a command (the solver's
shell, the checks) reaches the system read only, its own tree and a private
directory, and nothing else of the filesystem (no home directory, no state
root with the hidden tests, no sibling attempt's tree); it cannot connect
anywhere, loopback included (Landlock denies TCP, a seccomp filter refuses
every internet socket, UDP too); and its environment holds no variable named
like a secret (key, token, password, auth socket), even when an allowlist
names one. Without a kernel that can do this (Landlock ABI 4) the command is
refused, never run open. `--allow-workspace-network` opens the network,
`--allow-unconfined-workspace` the filesystem on a kernel that cannot confine,
and `--workspace-max-processes N` bounds the processes and threads of a
command; each is part of the environment record, so a task generated without
it names another environment. The agent's own file and search tools are
confined to the attempt's tree by sven's project root, for its delegated
children too. The supervisor makes itself non-dumpable when it starts an
attempt, so a command cannot read its environment through `/proc`. The exposure
audit reports a command's mention of a path outside the tree by what came of it:
`outside_workspace` (something came back), `outside_workspace_denied` (the sandbox
refused) and `outside_workspace_absent` (no such path, or nothing returned); all
three still fail the audit, and the stage counts them by kind.

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
teach advice the sources do not hold, and a plan that names a person while the
sources hold advice or judgment must choose a kind that teaches how they judge
(`advise`, `converse`), facts alone being no way to think like someone. The
person becomes the policy's system prompt, in training and afterwards, and the
teacher's answers and the student's attempts are written under it too, so what
the records teach was said as the person they open with. A plan that breaks a rule is sent back
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
adds beyond telling the base what the goal is. With `--retrieve SOURCE-ID...` the candidate is
also asked with the passages of those sources that bear on each task shown
before it (as `ask --retrieve` does), compared with the candidate alone, and the
report says for how many tasks a retrieved passage overlaps the evidence the
task was written from: what retrieval found, apart from what the model made
of it. The sources hold the held-out letters too, so this measures access to
what the weights never saw, not recall of it. A judge compares each answer
with the reference, and code separately checks whether the answer states a
number, name or quotation the task's source does not hold. Before the judge
grades an arm it is calibrated on controls made from the tasks' own references
- each task's reference answering it, and another's answering it - which no
model wrote; a judge that cannot tell them apart grades nothing and the report
says it makes no claim. A task an arm gave no answer to counts as not done and
is reported as unanswered. The result is a paired sign test of the judged
results, in which tasks about one family of source text (two prints of a
letter, several questions about one document) count once, as the release gate's
improvement check counts them: a model that knows the letter gets all of its
questions right, so they are one piece of evidence; beside it, a bootstrap
interval of the candidate's gain per family (won 1, lost -1, tied 0), which is
what a result on a few families has to be read by. It never reaches a
training set. Each control the judge did not
judge as labelled is reported with the answer and the judge's reason, so a judge
that is not trusted can be seen failing.

**The powered exam.** An exam of a few families cannot show a gain: the paired
tests above rest on as many units of evidence as families, which is a handful.
`learn --exam-families N` (fifty by default for a run with a persona, none
otherwise; 0 reserves none; `--exam-tasks-per-family`, eight; and `--dev-families` and
`--dev-tasks-per-family`, fifty of four, a dev suite on its own families for choosing a
checkpoint) reserves the exam's families before anything is
generated or built, after the plan. A family is a group of text parts that
print the same text (the groups a split uses, named by the least content digest
in them); a stable hash of the name chooses N of the examinable ones - long
enough to write tasks from, not so long that reserving them costs the run its
corpus, in a directory source, and not text the policy's own lineage was trained
on - and the run is refused, saying why, when the sources cannot spare that many
or reserving them would leave less to learn from than is examined. The text of a
reserved family in every edition is left out of the sources the run reads from
then on (the source records the parts as skipped, reserved), so no task, dataset
or passage index can hold it, and the exam is written from a source of the
reserved parts alone, within `--exam-tasks` tasks spread over the families, and
frozen under `<state>/exams/<id>/` with a ledger that refuses a changed manifest;
the manifest carries the tasks and the text they were written from, so loading it in
another state root installs the same exam there.
`splinter exam-set create` makes such an exam from sources apart from any
`learn`, clear of what the candidates named by `--not-trained-by` (and the
releases they continue) were trained on; `splinter exam CANDIDATE --exam-set`
puts a candidate to it. Four arms answer each task - the base, the base under the
prompt the candidate was trained under (the goal when it has none), the
candidate under the default prompt and the candidate under its own prompt - the
first answer greedy and the rest sampled (`--resamples`, three by default); a
family the candidate or a release it continues was trained on is left out. A
judge calibrated on controls from the tasks' references, and on harder wrong
answers (each task answered with the most similar reference of another family),
decodes greedily and judges each answer alone, shown only the task, the reference
and the answer: it cannot know which arm gave it or what was judged before. The
report keeps every answer and verdict of every arm so the analysis can be redone,
and for each pair of arms gives the per-task counts, the exact sign test over
tasks and over families, the difference in the share right and in the pass rate
over the samples with a family-clustered bootstrap interval, the same test over
answers of like length, the answers' lengths, the invented specifics, the number
of discordant tasks and the smallest difference the exam's own spread over
families could have shown at 80% power. The voice of each arm is also scored with
no judge, by the likelihood it gives the writer's own text of the reserved
families (brain's held-out scoring path). The `exam` stage of a `learn` that
reserved its exam runs this exam.

`splinter exam-set power` sizes an exam before it is paid for: it simulates the primary
comparison's test (the family-clustered interval of the difference above zero) under an assumed
effect, discordance and intraclass correlation, or under the discordance and clustering the
report of a pilot (`exam --pilot-families N`) estimated, and says how often the test finds the
effect and how often it finds one that is not there.

The judge is measured against a person, who has to supply the labels:
`exam-report labels-export` writes a sample of the exam's answers (400 by default)
stratified by arm, by what the judge said and by length, shuffled and blind to the arm, each
with the task and its reference passage and an empty `label`, and a key that keeps the arm and
the judge's verdict; the person fills each `label` with `right` or `wrong`, and `labels-import`
reports the agreement, Cohen's kappa, the share of what the judge called right that the person
called wrong and the reverse, agreement on short and on long answers and the length bias
(the judge grades each answer alone, so there is no position to favour). Nothing here makes up
a label. `exam-report memorisation` measures how much of each arm's answers is the training
text again - the share of 8, 13 and 20 word runs found in what the candidate was trained to
produce, the longest run, the answers quoting twenty words or more - so voice can be told from
memory; nearest-passage similarity and an authorship embedding are not measured.

The `converse` kind teaches how the writer talks and reasons. Its task is the
opening message of someone speaking to the writer (admitted only in the first or
second person: a situation, a decision, a question put to them; "What does he say
about ..." is refused, and the generator told so), with a passage of the
writer's text as its reference. A teacher shown the passage answers as the
writer, and a model that is never shown it plays the other speaker, following
up for up to three exchanges; the training record is the whole conversation
with every reply of the writer supervised and no passage in it. Two code
checks refute a dialogue, whatever else is said of it: every number, name and
quotation in the replies must be in the writer's text or in what the other
speaker said, and no reply may talk about "the material", "the passage" or "the
speaker", which the student is never shown. Neither establishes that a
dialogue is good: a judge other than the teacher does, and a dialogue it does
not pass teaches nothing. The other speaker never restates what the writer said
or asks it to confirm it; it brings its own case or objection. One dialogue in
eight ends by asking for a specific the exchange has not given, so that the
student also sees the writer decline to invent one.

`--with-passages SHARE` trains a share of the records with retrieved passages of
the sources in the prompt, as `ask --retrieve` gives them, four to a record and
the passage the task was written from among them for four in five, the answer
unchanged. A policy trained only closed-book cannot tell the passage that holds
the answer from one that merely resembles the question: measured, passages helped
where retrieval found the evidence and hurt exactly as much where it did not.
Each passage is shown under its part and section, so an answer can name where it
stands. `--abstain SHARE` makes that share of *all* the dialogue records
abstentions (so at most `--with-passages`): the base model, as the writer, says
in two to four sentences that the writings before them do not establish an
answer - two thirds of them for a question whose evidence retrieval missed, a
third for a question that lies beyond the writings (about what came after the
writer's lifetime), asked of the passages retrieved for it - admitted only
when the reply is brief, in the first person and states no number the question
and the passages do not hold. An abstention is made from a dialogue record and
keeps its experience and family, so it is held out and trained with them; a
record no admissible abstention is written for keeps its answer. About a tenth
of the data is the working figure; abstentions are never the majority.

The `author` stage teaches the writer's own voice. For the kinds whose
reference is a passage the writer wrote (`advise`, `converse`), the task's
message was written for that passage, so the passage itself, word for word, is
the answer to train on - instruction backtranslation: it has the writer's
diction and reasoning, which a teacher's paraphrase does not. It is recorded as
the source's answer, not a model's, and a judge of fit (the judge role, another
model than the generator that wrote the messages, measured like any judge on
controls from the tasks' own passages) keeps only the pairs where the passage is
a natural reply to its message. For one message the writer's passage is kept and
a teacher's dialogue on it is the duplicate; where it does not fit, the
dialogue stands. The judge of fit admits, so its passes are what is measured: a
passage it fails is withheld. With no usable judge of fit the stage says why and
the run goes on.

The `voice` view is the writer's own text as training data, with no model in
the loop. A conversation costs a teacher's time for every answer, and a run over
a large body of writing reaches a few per cent of it before its budget is spent;
the writer's text costs nothing to turn into records and has the diction and
reasoning no paraphrase has. Every text part of the sources is cut into chunks
of 250 to 650 words (and at most 880 tokens, counted by the policy's own
tokenizer, so a record never outgrows the 1024-token row the conversations
need): whole sentences, closed where a paragraph ends, an address line or
heading always with the text it heads, and no chunk of a heading alone or of a
few words. Each chunk becomes one record whose supervised answer is the chunk
word for word and whose request is written by code from what the text says of
itself: `Write, as NAME, to RECIPIENT in YEAR about: <its opening>`. Half of
the records open with the persona system prompt and half with the line that
says only who the model is, so the voice is carried by the adapter and not
only by the prompt. Two prints of one text are one family and only one is
projected, and no family supplies more than a tenth of the tokens
(`--max-family-share` on `dataset build`; `--writer NAME` names the writer and
`--token-limit N` keeps at most N tokens, spread evenly over all the chunks).
`--describe-voice` (`dataset build --describe-with REF`) asks each chunk by what the
generator says it is about instead of by its opening: 20 to 60 words in the model's own
words, admitted only when no run of eight words, and no more than 35% of its words in runs of
four, are the passage's own; a chunk with no admitted description keeps the opening request.
Descriptions are kept per generator under the state's `work/describe`, so a second build
reuses them.
A record names the part it prints and no experience, so a dataset shows what is
the writer's and what a model wrote. A run with a persona trains on the
writer's text by default: `--voice SHARE` is the share of the training tokens it
makes (seven tenths; `--voice 0` turns it off), the chunks chosen as an even
spread over the whole of the sources in a stable order that does not follow
their names, and the training is sized by all its examples. Such a record is
held out with the family of the letter it prints: the families held out are
decided by the records that can be examined - those projected from a task or an
experience - and a record of the writer's text goes where its family goes,
however many there are and wherever they sit, so the exam never asks about a
letter the policy was trained on under another print's name. The held-out score
is measured on the examinable held-out records, what the policy is asked as; the
writer's text of the held-out families is written to a file of its own
(`held_out_text.jsonl`), trained on by nobody, and the writer's text of the
monitoring families is monitored with the dialogues, where most of a run's
tokens are.

A run with a persona also rehearses the base (the `rehearse` stage). A
fine-tune on one person's answers erodes the base's general behaviour before
it forgets a fact: under the plain assistant prompt it answers a sum at length
and gets it wrong, or answers in a paragraph where a word was asked, which is
what the release gate's anchor suite catches. What holds a model to its base
is training beside the new records on what the base itself answers, with no
corpus from outside: the base model - the policy's base checkpoint with no
adapter, under the default assistant prompt, decoding greedily as the gate
asks it - answers a set of general tasks, and its answers are rehearsed as it
gave them. Half the tasks are `arithmetic` and `format` tasks built by code
from a seed (fresh numbers and subjects, a sum asked worked through or for the
number alone, a list asked one item per line), each with its reference, so an
answer the base got wrong is graded wrong and never rehearsed; the rest are
general requests the base writes itself, one domain at a time over sixteen
everyday domains, graded by nothing and kept as given. No anchor task, nor a
near copy of one (the same text in other case and spacing, a close rewording,
its words inside a longer prompt), is ever in the set: every anchor
instruction is held against each task by the generator's own near-duplicate
rule before it is stored, since a candidate rehearsed on an anchor task would
pass the anchor check by having trained on it; the gate's leakage check reads
the rehearsed records too. `--rehearsal SHARE` is the share of the training
draws the rehearsed records take (a quarter by default for a run with a
persona; `--rehearsal 0` turns it off, and a run without a persona rehearses
nothing unless asked), the set sized so that a rehearsed record is drawn about
as often as a dialogue answer; a tenth of its records (the monitoring share)
join the monitoring set instead, so the step the run carries is the one that
keeps the base's answers as well as it fits the new ones. `rehearse --records
N` builds such a set by hand and `train --rehearsal DATASET-ID
[--rehearsal-share F]` mixes it, or any chat dataset of the base's own
answers, into a training run.

Task generation is bounded by the budget and spread over the sources: each
text part is shown through at most four evenly spaced windows of sections,
parts are visited in a stable order that does not follow their names, and the
budget stops generation inside a part. A held-out split never divides texts
that print the same passage (two editions of a letter, a letter one edition
prints inside its neighbour, a passage reused deep inside a longer text):
records name the group of overlapping source text they came from - texts that
share eight distinct eight-word runs anywhere in either, a run held by more
than six texts counting for nothing - and a group is held out or trained on
whole. Records about one group are one unit of evidence to the
gate's and the exam's paired tests, and a tenth of a set generated letter by
letter is two or three of them, so whole groups are held out, newest first,
until there are eight, as long as that stays within a quarter of the records.
A group is named by the least content digest of the texts in it, found over
every text part of the sources, so two datasets built from the same sources
name a family the same and are split together as one.

`--distill` skips the policy's own attempts: the teacher answers every task
open-book and the policy is trained on its verified answers, with no
frontier and no critique. It is for a policy that cannot answer a task
closed-book at all, where its attempts are the most expensive part of a run
and teach nothing. A plan can choose it too.

A run needs no flags. The configuration names what a machine knows:
`SPLINTER_ASSISTANT_MODEL` is the stronger model that plans, writes tasks and
teaches when the command names none, `SPLINTER_JUDGE_MODEL` judges what only a judge can decide, `SPLINTER_FRONT_DOOR_MODEL` reads the
sentence, and `SPLINTER_BUDGET` is how long a `learn` may take when it names no `--budget`
(a sentence names none), `SPLINTER_THINKING=1` lets local models reason before
they answer (off, a reasoning model is asked with its reasoning block closed, so
every reply is the answer and comes at once; training renders its answers the
same way, so both settings work), and `SPLINTER_BF16_BASE` holds a large policy's base at bf16 so it
trains on one card. `--remote-concurrency` is how many requests to a model
reached over an API may be in flight at once (4 by default); a model on the local
device is asked one at a time. `--temperature` is the sampling temperature (0 to 2)
asked of such a model, the provider's own default when not given. `--solve-deadline` is how many seconds one task's solve may take on the wall clock (300 by default); raise it when many tasks share one served model, so a task is not failed for waiting in the queue. Unless `--steps` is given a run's step budget
is three passes over what it learned, within bounds, at one learning rate
suited to a short LoRA run for `train` and `learn` alike (`--lr`, 2e-4),
a LoRA alpha of twice the rank (`--alpha`, so the update's scale does not
change with the rank) and no weight decay (`--weight-decay`), all recorded
in the candidate. Each pass takes every record once in a fresh order, a step
is a mean over its supervised tokens, the rate warms up over a twentieth of
the steps and decays over all of them, and a run that stops on its plateau
first cools the rate down to the floor. A step averages several records - one per
sixteen the data holds, up to eight, or `--records-per-step N` - because an
update on a single long, individual answer is noise the next record undoes.
The budget is a ceiling: the run is watched as it trains (see `train`) and
carries the step with the best monitoring loss, stopping early once that
loss stops improving.

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

Task kinds: `recall`, `taught` (something a person told an agent; graded by the terms of its reference sentence an answer carries and adds, see `absorb`), `explain`, `predict`, `construct`, `debug`,
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
agreement among answers, and - with `--judge REF` - a judge gated by
the calibration `judge calibrate` stored for it. A verifier either
establishes that an answer is right or only refutes it: grounding (every
number, name and quotation is in the source) is a constraint, whose fail
refutes whatever else says and whose pass decides nothing, so an answer that
invents nothing but says nothing right passes no check. A kind no other
verifier can pass - a conversation - is decided by a judge, and a `learn` on it
names one (`--judge`, else `SPLINTER_JUDGE_MODEL`, else the assistant). The
judge is another model than the teacher, the generator and the policy; it is
measured before any verdict counts, on controls made from the tasks' own
references (each task's reference is the right answer to it, and a task of
another family of sources lends the wrong one) unless a stored calibration
rests on at least `SPLINTER_MIN_CALIBRATION_CONTROLS` of them (16). A judge is
trusted for what its verdicts do. Where they admit answers to a training set
(`verify`, the `teach`, `critique` and `author` stages) its passes must be
precise (0.9): a fail only withholds an answer, so a strict judge whose fails
were measured less precise has them abstain and admits nothing wrong. Where two
models are compared on its verdicts (the release gate, the exam) its passes and
its fails must both be precise, or the comparison would be left with the ties.
A judge not precise enough for its use is refused with its numbers; `splinter
judge measure` says which uses a judge is fit for. A labelled file is JSON
Lines, `{"experience": "<id>", "label": "pass" | "fail"}`.

`dataset build` views: `sft-final`, `sft-step`, `critic`, `preference`,
`verifier`, `decision`, `retrieval`, `outcome`, `denoise`, `cpt`, `voice`
(the writer's own text as answers, from every text part of the sources the
experience sets cite; see above), `rehearsal` (a model's own answers as it
gave them, every experience with a final answer that no verifier decided
wrong, under the default prompt). `--limit N` keeps at most N records of any
view, evenly spread over the projection, the rest counted as `over_limit`. The
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
order and holds the newest records out for scoring (a record of the writer's
own text, which has no question to examine, is held out with its family and
scored by nobody): a supervised candidate
reports the held-out loss of base and candidate, a preference candidate
brain's preference score on the held-out pairs (how often, and by how
many nats, it prefers the chosen answer more than its reference does).

A supervised run is watched as it trains. A share of the training families
(`--monitor-share`, default 0.1, whole families) is set aside as the
monitoring records - carved from the training half, never from the held-out
half the gate and the exam decide on, so no step is chosen on the evidence
it is then judged by - and scored every `--eval-every` steps (default:
sixteen evaluations over the step budget, and at the last step). The
candidate carries the adapter of the evaluation with the lowest monitoring
loss, not the last step's, and the run stops once that loss has gone
`--patience` evaluations (default 4) without improving; `--patience 0` runs
the whole budget and still carries the best, `--eval-every 0` monitors
nothing and carries the last step. `--steps` is the budget, the most steps
the run may take: by default three passes over the examples. The candidate's
report shows the steps trained of the budget, the step carried and why, the
training and monitoring loss at it and the gap between them, the whole curve
(step, mean training loss since the previous evaluation, monitoring loss),
and warns when the gap at the carried step is more than a quarter of the
monitoring loss, when the monitoring loss rose after the step carried, or
when it was still falling at the end of the budget; the release gate and the
exam repeat those warnings beside their verdicts.
`--keep-evaluations` also keeps the adapter of every evaluation with the
candidate, and `splinter select CANDIDATE --exam-set DEV-SUITE` puts each of
them to the dev suite (the families `learn --dev-families` reserves apart from
the final test and from all training) and chooses one by a rule fixed in
advance: an evaluation whose answers state what the source does not hold in
more than 8% of the answers checked is out; of those left, the best share of
answers the judge calls right and every one within a standard error of it are
in the running; of those, the writer's own text of the dev families gets the
lowest loss. Correctness decides, voice breaks the tie it cannot, and an
evaluation that invents is never chosen for its voice. `--adopt` makes the
chosen evaluation a candidate of its own (its held-out score is not carried
over: it was measured of another adapter) to release; `learn --select-on-dev`
does both in the run, between training and the exam.
From `policy:<alias>` (the default) it continues the adapter of the
release the alias points at - never the base weights once a release
exists - and a supervised run replays `--replay-fraction` (default 0.25)
of every earlier release's trained-on chat records, a sample drawn with the
run's seed (so the same seed replays the same records) that never includes
what that release held out;
replayed records are never held out, and take a quarter of the training
draws however many there are, so a large replay cannot starve the new
records. `--rehearsal DATASET-ID` mixes a rehearsal dataset (`rehearse`; see
`learn`) in the same way at `--rehearsal-share` of the draws (default 0.25;
the replay's quarter and this share must leave the new records some draws),
never held out, and puts `--monitor-share` of its records into the
monitoring set beside the training families' records, so the step carried is
chosen on the mixture the run trains. The candidate records what it
rehearsed (dataset, share, records trained on and monitored), and the records
mixed in are kept as an artifact the gate's leakage check reads. `--seed N`
seeds the adapter's initialisation, the batch order and the replay draw;
without it a seed is drawn, printed (`no --seed given, using seed N`), written
to the run record and to the candidate (`seed`), and passing it again writes
the same adapter on the same data. Two runs that differ only in the seed show
how much of a measured difference is the draw. A preference run replays and rehearses nothing:
brain's preference trainer trains on its pairs alone. `--from
local:<checkpoint>+<adapter>` continues that adapter instead, with no
replay, and such a candidate cannot replace a champion. The candidate
records its regime; the release gate grades either the same way.

Every random choice a pipeline makes takes a seed the run states. The exam's
families (`--exam-seed`, also on `exam-set create --seed` and, for `practice`,
kept with the families so every later run reads it back and naming another is
refused), the rehearsal's form tasks (`--rehearsal-seed`), the denoise tasks
(`--denoise-seed`), the subjects of `--strip mix:F` (`--mix-seed`) and the
exam's bootstrap intervals (`--interval-seed`) are drawn when no seed is given,
printed (`no --exam-seed given, using seed N (pass --exam-seed N to
reproduce)`), written to the run record (`seeds`) and kept in the artifact they
decide: the reservation, the frozen exam, the rehearsal and task reports, the
dataset manifest and the exam report. The release gate's bootstrap intervals
(the tasks' own evidence, the anchor's drop) are resampled from a seed of their
own, recorded as `gate-interval` and kept in the gate report and so in the
release manifest; `release --interval-seed N` passes it again. A local model
that samples draws a seed for each request that names none; the run that made
the requests records them in order (`runs show` lists the count, the first
sixteen and a digest of all of them, and the whole list is kept in the run's
events), and a seed passed back as the request's replays its decoding. Greedy
decoding draws nothing. Passing the seed again reproduces the choice. Two runs
reserve different families unless they share a seed, so
results are comparable across runs only when each names a recorded one.

A `learn` report opens with an `outcome` block of five separate facts: whether the
run finished or stopped, whether a candidate was trained, what the release gate decided
(naming the checks that blocked it), whether a release was written, and whether
`policy:default` moved. A trained candidate the gate blocked is reported as exactly
that, and the policy as unchanged.

## Releases

`release <CANDIDATE-ID>` decides a candidate against the release its alias
(`--alias`, default `default`) points at - the champion, or the base before
any release - and refuses a candidate that was not trained from it. Both
are graded closed-book by each task's own verifiers (the judged kinds by
`--judge REF`, a calibrated judge) on the same suites, and the candidate is released only if all four checks pass; each
is printed with its numbers, and a check that could not be measured fails:

| Check | Passes when |
|---|---|
| improvement | on the new datasets' held-out tasks - the records training held out, and the variants of the tasks it trained on - a one-sided paired sign test over the tasks only one model got right is significant at alpha 0.05, tasks about one family of source text counting once (a family is won or lost by which model got more of its tasks right); ties and tasks without a verdict for both are excluded and counted |
| retention | on each earlier release's held-out tasks, the candidate's accuracy is at most 0.05 below the champion's (each release reported) |
| anchor | on the anchor suite in force, the candidate's accuracy is at most 0.02 below the champion's; both arms are asked it under the default prompt, so it measures what the weights kept and not the persona prompt; the report gives the drop's bootstrap interval over the paired items beside it |
| serve | `brain serve --adapter <candidate>` (the `brain` on `PATH`, or `SPLINTER_BRAIN_BIN`), with the base checkpoint the candidate was trained on as its `BRAIN_QWEN_WEIGHTS`, starts - brain binds the adapter only to the base whose digest training recorded on it - reports the candidate's adapter digest, and re-answers up to 8 held-out tasks through its OpenAI-compatible endpoint, both sides decoding greedily and without a reasoning block, with the same answers as in-process on at least three quarters of them (the same verdict, and either the same text over its first nine tenths, runs of whitespace aside, or - worded otherwise, as a sampling server or another summation order will - the same meaning: embedded by Qwen3-Embedding, an answer must be nearer its own in-process answer than to the in-process answer of any other task, so the comparison cannot call everything alike); each task answered differently is reported with both answers |

A gate that passes is not silent about the candidate's training curve: the
warnings `train` reported (a large generalisation gap at the step carried, a
monitoring loss that rose after it or was still falling at the end of the
budget) are printed again with the verdict and carried in the JSON report
(`warnings`), as the exam carries them (`training_warnings`).

The improvement check measures whether the candidate learned the facts it
was trained on, on questions it was not trained on: the held-out records (the
newest tenth of the new data, grown to eight families of source text where
the data has them, within a quarter of it, so that a sign test over families
can reach significance) are few, and questions about other facts cannot
improve for a campaign that teaches facts. So the suite is completed with the stored
variants of the tasks the candidate's trained-on records were projected
from (`learn` writes them; `tasks variants` writes them for any task
set). A variant of a task that was held out measures nothing the
candidate learned and is not used. A variant whose wording the candidate
trained on - the same question, a near duplicate, or its words inside a
longer training prompt - is left out like any leaked held-out task. The
gate prints how many variants were measured and how many were left out,
by reason; the manifest records them. The decision rests on the two together by the family-level sign test, which stays the rule even
though its p-value moves with one family when only a dozen are discordant; the report therefore
also gives, for the whole suite and for each part, the difference in the share right with an
interval that resamples families and the exact sign test over the tasks, to read it by.
The decision rests on the two together,
but the report keeps them apart: generalisation (the held-out records, text
the candidate was not trained on) and recall under paraphrase (the variants
of trained tasks) each print their own comparison and sign test, and the
report states the system prompt each arm was asked each suite under
(`prompts`). Retention, anchor and serve are measured as before.

A release is written once. Its manifest is kept in the experience database
(`<state>/expdb`) in canonical JSON - the base model and
its digest, the adapter digest, the parent release, the candidate, the new
datasets, the replay sample and the rehearsal with their digests, the training record
(brain's included), every gate number with the anchor suite's version and
digest, and `created_at` - and names the adapter file, which stays where the candidate
was kept, by content address under `<state>/artifacts/`. The manifest's
digest is the release id. An alias is a pointer in the same database that moves only
from the champion the gate measured against; `<state>/serving/<name>.json` is the manifest a
server follows for it, rewritten whenever it moves. `release
list` shows every release with its parent and aliases; `rollback <ALIAS>`
points the alias at the release its current one was trained from, and
refuses when there is none.

An adapter belongs to one base model. To serve a release without
Splinter, name that base - the manifest's `base_model` and `base_digest`
say which checkpoint it is - and the release's adapter, as the serve check
does:

```bash
BRAIN_QWEN_WEIGHTS=<base checkpoint> brain serve --openai --adapter-manifest <state>/serving/<alias>.json
```

brain serves the adapter on that one Qwen3 base, as the model
`brain/qwen3`, and refuses to start when the base's digest is not the one
the adapter records.

To keep a running server on whatever an alias points at, give it the
manifest the release store rewrites whenever the alias moves (a release, a
rollback) instead of an adapter file:

```bash
BRAIN_QWEN_WEIGHTS=<base checkpoint> brain serve --openai --adapter-manifest <state>/serving/default.json
```

The manifest names the adapter the alias points at and its digest, and is
replaced atomically; the server swaps onto the new adapter once the file
hashes to the digest and binds to its base, and keeps the one it has when it
does not. An alias at a full checkpoint names no adapter, so its manifest is
withdrawn.

A candidate trained by `practice` is gated on a different suite than a
candidate trained on documents: the champion and the candidate attempt, as
agents in copies of the tree, the tasks of the exam that candidate's run set
apart before any attempt (every earlier release is held to the exam of the run
that made it), and the paired outcomes are counted by family in the same sign
test. Attempts that read the hidden tests, and attempts the harness could not
carry out, are set aside and counted for each arm in the report; neither is
in a denominator. The serve check asks the served candidate the same tasks
through the same worker and requires its verdicts to agree with the
in-process ones on at least three quarters of a sample. The anchor suite is
the frozen one; it holds coding-agent tasks only if some were frozen, and the
report says how many it holds rather than scoring an absence.

`eval <REF>` grades one model - a candidate id, or a model reference -
closed-book on `--suite held-out` (the default: a candidate's new data, or
the release a policy alias points at), `retention` (every earlier release,
each reported), `anchor`, or a FILE of tasks, and names every task it got
wrong with the answer it gave, so two evaluations show what a candidate
lost against the base. The anchor suite is frozen
from one or more files with `eval --suite anchor --freeze FILE...` (the
files' tasks in order are one version; different tasks are the next version;
the same tasks are the same version) and shown with `eval --suite anchor`. An
anchor file is JSON Lines, `{"instruction", "reference", "kind"?}`, `kind`
(default `recall`) a closed-book kind graded against its reference by code:
`recall` (the answer states the reference), `arithmetic` (the last number the
answer states is the reference) or `format` (the answer has exactly the
non-empty lines the reference counts). The last two are what a persona
fine-tune of a reasoning model erodes and trivia cannot show.

## The front door

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
by a guess. `SPLINTER_FRONT_DOOR_MODEL` names a model for reading sentences
other than the policy, since a larger model reads them better.

## Learning from sessions

A person's own sessions with an agent are ATIF trajectories (`*.atif.json`).
`session add` takes files, or directories searched recursively, and for each:

- validates the ATIF, and refuses (naming the step and why, in `refused`) a
  trajectory the training projection cannot render - for instance a second
  user step beside a tool call - or one with no user step;
- removes secrets before anything is addressed: private key blocks, values
  assigned to secret names (`password=`, `api_key:`, `GITHUB_TOKEN=`, JSON keys
  of that kind), bearer credentials, and tokens known by their shape (`sk-`,
  `ghp_`, `AKIA`, JSON web tokens). Each removal leaves `[REDACTED:<kind>]`; the
  source's origin records the step and kind of every removal and never the
  secret. The filter recognises shapes; it is not proof that nothing sensitive
  remains;
- stores one part per step text (`step-000003-user`, `step-000004-agent`,
  `step-000004-calls`, `step-000004-observation`) beside the redacted trajectory,
  so a claim cites a step by its ATIF `step_id` and a span into its part.

The same bytes are the same source, so intake is repeatable. The command
exits 1 when any file was refused; the others are taken in.

`claims extract` shows each session to the generator model (a typed call; a
malformed or over-long reply is sent back for correction, and one that stays
unusable is reported against its session in `failed`) and stores the proposals
whole as a claim set. A proposal is a kind (`correction`, `fact`, `procedure`),
a self-contained statement, the question it answers, the person's quotes by
step, and, for a correction, what the agent said wrong (context only) or, for a
procedure, the tool-call steps and the observation quotes that proved it. Every
proposal also says whom it is about (`subject`: `self`, `world` or
`third_party`); a reply that leaves it out is sent back for correction.

`claims extract --passes N` (two by default; `absorb --passes N` likewise) reads
each session N times, in different orders, and keeps only the claims every pass
proposed, matched by the words they cite and what they state, not by their
wording. The proposals some pass lacked are stored in the claim set as
unconfirmed, listed in the report with their count, and never ruled on. A pass
whose reply stays unusable fails the session.

`claims gate` rules on every proposal by code alone, in order: shape; every
quote verbatim in the user step it cites (a quote of an agent step is refused
as `assistant_evidence`, whatever it says); a procedure has its calls and
observation; what the agent is said to have got wrong was said; every number,
name and quoted term of the statement occurs in the cited words (and, for a
procedure, in its cited tool calls); the statement does not give personal data
of someone other than the person (a phone number, a home address, health, a
relationship, finances), else it is refused as `third_party_personal`: when the
proposal's subject is `third_party`, or when it is not `self` and the statement
names someone; the words it recognises are a short list that leaves out words
also used of software, and what the person says about themselves is not
affected; the same claim proposed twice from the
same words is refused as `duplicate`. A claim is then paired with the live
claims about the same thing: those it shares a name with (a capitalised word
inside a sentence, an identifier, a quoted term, over the statement and the
question) and at least one more content word, so two questions the extractor
worded differently are still one subject; two claims that name nothing are
paired when their statements share most of their content words. A judge
(`claims gate --judge REF`; in `absorb`, the judge role) decides each pair:
`supersede` (the later contradicts the earlier, which leaves the live set),
`reinforce` (the same fact said again) or `separate` (two facts, both live).
Without a judge the statements decide: they agree when one contains the other
or state the same numbers, and an agreeing pair reinforces while a disagreeing
pair supersedes, the later winning. Without a judge, then, two different facts
about one named subject replace each other; the ledger keeps both rulings.
A fact said again is never refused: it is recorded as `reinforced`, the live
claim it restates stays live (it is not a second thing to teach), and
`claims ledger` shows how many times each live claim was said again, which is
a signal that the model is still wrong.

When a judge is configured it also checks entailment: shown only the cited
quotes and the statement (never the question, the session or the agent's words),
it must say that the person's words assert the statement, else the claim is
refused as `not_entailed`. This catches what the term rule cannot, such as a
denial ("it wasn't 12") restated as a fact. `--entail=false` (on `claims gate`
and `absorb`) turns it off; without a judge it does not run. "Later" is the order of the ledger,
and within one session the order of the first step a claim cites. Every
ruling is appended to the ledger, each refusal with its reason; ruling a claim
set again rules on nothing. `claims ledger` reads it back: the live claims
with their reinforcement counts, the superseded ones and by which claim, the
forgotten ones, and the refused proposals.

A recording of a session may name the fact the session was made to teach, as
`{"absorb_subject": "<fact id>"}` in the `extra` object at the root of its ATIF
trajectory. `claims gate --subjects FILE...` (and `absorb --subjects`) reads what
each fact looks like (JSON Lines of `{"id", "terms": [...]}`; a statement
carries a fact when it contains every term, case aside) and the gate report
lists, for each named session, whether the claims the extractor proposed and the
claims the gates admitted cover it, with the extraction's recall (sessions whose
admitted claims cover their subject) and precision (admitted claims of those
sessions that cover their subject). A subject with no description has no
verdict, and a ratio over nothing is absent.

`claims forget <CLAIM-ID>` appends a `forgotten` ruling for a live claim (an id
or unique prefix from `claims ledger`): the claim leaves the live set, and the
ledger keeps it and its evidence. Because `absorb` trains again from the base on
the live claims, the next night drops it from the weights; each release
records the claims it was trained on, so a night whose champion holds a claim
that is no longer live trains again even when no claim is new (not with
`--continue-from-release`, which cannot take a claim out of the weights).
Forgetting the only live claim leaves no record to train on, and the night
halts saying so.

The first word of a sentence is not taken for a name, and numbers written
in words are not recognised: both are terms the gate does not check, so on the
statement side they make it admit more, never refuse more, and a statement may
carry them without the person's words having said them. A number
that ends a sentence ("on port 9090.") is a number like any other.

### absorb

`absorb <SESSIONS>...` composes the stages above into one recorded run that
turns a day's sessions into the next release, or says why not. Its stages are
`policy`, `intake`, `extract`, `gate`, `rebuild`, `kits`, `dataset`, `rehearse`, `train`,
`stance`, `preference`, `release`, `ledger` and `consolidate` (`rebuild` and `consolidate` only when there is
something to do; see [Generations](#generations)); each is a function of its own in `splinter-pipelines`,
its JSON is in the run, and a failed run is resumed by running it again: the
sessions, the claim set of the same sessions and generator, the kits and the
datasets are kept by content, and a candidate trained on the dataset a failed
run built is taken as it is.

- **extract and gate** are `claims extract` and `claims gate`. `--dry-run` ends
  here: the claims are ruled on, every refusal listed, and nothing is trained.
- **kits.** A fact shown once is memorised in one wording, so every live claim
  without a kit gets one, kept for good: its question and eight differently
  worded questions (`--paraphrases` are written, a third kept out), answered by
  a teacher that is shown the claim and speaks as the policy (its release's
  system prompt), each answer kept only when the claim's verifiers pass it (the
  `taught` kind: the answer carries every number, name and quoted term of the
  statement and adds none); what the person first asked, answered with that
  verified corrected answer (the agent's wrong reply, and its "you are right"
  after the correction, are context and never trained on); four statements,
  two questions that ask for the subject given what is said of it, and two
  consequences, admitted by the same term rule. That is 18 records per claim. The
  four paraphrases left over (of twelve) are the claim's own stopping and gate
  set: no record contains them. A claim with no subject the question and
  statement both name has no paraphrase to write and is taught but cannot be
  measured. A claim of kind `procedure` is stored and never trained on; a
  session that mixes a later user step with tool calls is refused at intake.
- **dataset.** The kit records of every live claim, all fixed to the training
  side (`metadata.split`), so the holdout rule never holds a claim out and no
  family is set aside to monitor; the stopping paraphrases are the held-out
  records. `--sealed-probes FILE...` (JSON Lines of `{"question", "reference"?,
  "name"?}`) refuses a record containing a probe's question or sharing an
  8-word run with one beyond what its claim's statement says, naming the probe;
  the rehearsal is checked too. The gate never reads the probes. A policy deployed under a system prompt (a persona) is taught and graded under it: the SDK's
  `AbsorbRequest::system_prompt` names it for the first night, and each release records the prompt it was
  trained under, so a later night keeps it.
- **train.** The adapter is trained again from the base on every live claim and
  the rehearsal (`--rehearsal-share`, a quarter by default), for at most
  `--epochs` passes (6) over the records, so a superseded claim is simply not in
  the next set. `--continue-from-release` is the ablation: continue the current
  release on the claims not yet absorbed with `--replay-fraction` (a quarter) of
  each earlier release's records replayed. The run does not stop on the claims'
  own stopping paraphrases; they gate the release instead.
- **stance and preference.** A model under a persona gives in to a confident
  falsehood about a fact it knows, says "you are right" and invents a rationale
  for what it is told, and makes up the specifics of what nobody could know; the
  person still needs it to take a true correction of a fact it got wrong. After
  the supervised pass the night therefore trains a DPO pass from that candidate
  on pairs derived again each night from the live state, on the pairs alone and
  against the model it starts from (`--stance-beta`, brain's temperature by
  default). Three kinds, split evenly: **hold** - a fact the policy answers right
  (written by the generator, kept only when the policy's own answer carries every
  number and name the statement holds that its question does not give away) is
  pushed with a confident false value, one key of it replaced by code with another
  of its kind and granularity; the preferred reply is the teacher's, shown what is
  true and kept only when it holds the true key without the false value, the
  rejected reply is a warm sample of the supervised candidate that
  takes the false value up (`--stance-tries` draws; when none does, a written
  capitulation checked the same way, counted in the report as written). **Accept**
  - each live claim, stated by the person; the preferred reply acknowledges it
  and adds no number, name or quoted term of its own (the rule that admits a
  claim's statement) in at most fifty words, the rejected reply is a sample that
  does add one. **Decline** - a question that asks for a specific nobody could
  know (written by the generator; the existing abstention writer writes the
  reply, kept only when it says it does not know and invents nothing), the
  rejected reply a sample that invents a number, name or term the question and
  the live claims do not hold. A pair whose half cannot be found by code is not
  made, and the report counts why. No source is a sealed probe: a fact or
  question that is one, or an anchor task or a near copy of one, is left out and
  counted, and a pair that contains a sealed probe - its conversation or either
  reply - stops the night naming the probe. `--stance-pairs` (96 by default) is
  the number asked for, never more than half the supervised records of the night,
  so absorption is not drowned; `--no-stance` trains the supervised candidate
  alone. A fifth of each kind, chosen by a hash of the item, is held out: no pair
  is made of it, and it is what the gate measures. The candidate the gate judges
  is the preference one; it records the datasets of both passes.
- **release.** A declared rule on counts, no significance test. A claim is
  answered when every one of its stopping paraphrases passes, graded greedily
  under the prompt the model is deployed with. Improvement: the candidate
  answers a claim the champion did not (vacuous when the champion answered every
  claim, as on a night that only took a forgotten claim out). Retention: every claim the champion
  answered is still answered, with no tolerance. Stance, when a stance pass ran:
  the held-out facts held, statements accepted without inventing and questions
  declined are put to the candidate and the champion, greedily, and none of the
  three rates may be lower than the champion's (compared by counts); a true
  correction is not given up for them, since improvement and retention still
  decide on the claims' stopping paraphrases. A rate over no items is absent in
  the report (`stance_rates` beside `gate`, and `claims ledger` per release), never
  zero. Anchor, when a suite is frozen:
  the candidate may not get fewer items right. Serve: plain `brain serve` answers
  a sample of the stopping paraphrases as the candidate does in-process. A
  candidate the gate refuses stays on record (`candidate`, the run's `gate`
  numbers) and the current release stays in use; `--no-release` stops at the
  candidate. The release's parent is the champion it replaces.
- **ledger.** Each live claim the release answers records the release that first
  absorbed it; `claims ledger` shows it as `absorbed_by`. A claim the gate did
  not see answered stays new and is taught again the next night.

Not built: the stopping of the training on the stopping paraphrases. The stance
pass does not replay or continue earlier preference releases with
`--continue-from-release`: a preference release's pairs are not chat records and
are not replayed.

### Generations

A policy that goes on learning for good cannot retrain its adapter from the
base on every claim it ever learned: the records grow without bound and every
night costs more. So it learns in generations.

- **Within a generation** claims accumulate and every `absorb` trains the
  adapter again from the generation's base on the generation's live claims, as
  above. Generation 0 is the configured base.
- **`consolidate`** takes the generation into the weights. The checkpoint it
  makes is the base of the next generation, and that generation's claim set
  starts empty: the next `absorb` trains a new adapter on top of the
  checkpoint, on the new claims alone. `--mode full` (the default) trains every
  weight of the generation's base on the records of the claims of the release
  in use, beside a bounded sample of the records of the claims the base
  already holds (`--retention-replay-share`, a quarter of the draws, never more
  than there are) and the base's own answers (`--rehearsal-share`), at a
  learning rate far below an adapter's (`--lr`); brain keeps the optimizer's
  moments and master weights in system memory, so the cards hold the weights
  and their gradients, twice the model's fp32 size: about 4.4 GiB for
  Qwen3-0.6B, 61 GiB for Qwen3-8B, which fits no two 24 GiB cards. `--mode
  merge` folds the release's adapter into the weights it targets and trains
  nothing, on the host, at any size, and is the cheap alternative; what it keeps
  is exactly what the adapter did. `--from RELEASE` consolidates an earlier
  release of the policy instead of the one in use.
- **The gate** is the claim gate without the demand for a gain: every claim of
  the closing generation and every claim the base already held is asked by its
  stopping paraphrases, the checkpoint (no adapter) against the release in use
  (with its own); a claim the release answered must still be answered, the
  anchor suite may lose no item, and plain `brain serve` must serve the
  checkpoint. A claim neither answers is carried and listed as unanswered. A
  checkpoint the gate refuses is not kept and the release stays in use.
- **The release** is a full checkpoint, immutable and content addressed, whose
  parent is the release it consolidated, whose manifest records the claims it
  took in (`closing`), the ones its base already held (`retained`) and the mode,
  and which holds the claims it was trained on in the ledger. Every release
  manifest records its generation (`epoch`) and the consolidation it sits on;
  `status` shows the policy's generation, and `claims ledger` the generation
  and which live claims the base holds (`[in the base]`).
- **Retention.** The claims a base holds are never taught again. A night's
  adapter is trained on the generation's own claims beside the sample of the
  retained claims' records, and the gate asks every retained claim as well, so
  a candidate that loses one is refused with no tolerance.
- **Forgetting a claim the weights hold.** The weights have absorbed it and
  cannot give it back, so the only honest removal is to train again from the
  last base that did not hold it. The next `absorb` (or `consolidate`) after
  `claims forget` does that, in the `rebuild` stage: it finds the earliest
  consolidation that took the claim in, trains every weight again from the
  base that consolidation was trained from, on the claims consolidated since
  that are still live, in one checkpoint, gates it against the release in use
  and releases it in its place; the generation's own claims are then taught to
  a new adapter on top. A rebuild trains every weight whatever mode made the
  consolidations, so it needs `--mode full` (a merge refuses) and a model whose
  weights and gradients fit the cards. Forgetting a claim of the generation
  itself only retrains the adapter, as before.
- **The stance pass** (hold, accept without inventing, decline) is trained on the
  nightly adapter, so a `merge` carries it into the base and a `full`
  consolidation, trained on the claims' records alone, does not: the next night's
  stance pass teaches it again on top of the checkpoint, and the consolidation
  gate does not ask the stance rates.
- **When.** `absorb --consolidate-after-records N` consolidates the generation
  the night's release closes once its claims hold more than `N` trained records
  (not at `N`); `--no-consolidate` never does; `--consolidate-mode` picks `full`
  or `merge`. The default is a count at which retraining the adapter from the
  base each night has become the dearer step; the run reports the consolidation
  beside the release it follows.

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

`session add`: `{"sessions": [{"path", "source", "new", "steps", "redactions"}],
"refused": [{"path", "reason"}], "new", "redactions", "run"}`. `claims extract`:
`{"claim_set", "sessions", "proposals", "by_kind", "failed": [{"session", "reason"}],
"stopped", "run"}`. `claims gate`: `{"claim_set", "proposals", "ruled", "already_ruled",
"admitted", "superseded", "refused" (by reason), "rulings": [{"index", "session",
"kind", "statement", "outcome", "claim", "supersedes", "reason_code", "reason"}],
"live", "extraction_failed", "run"}`. `claims ledger`: `{"generation": {"epoch", "base", "retained", "current"}, "live": [claim],
"superseded": [{"claim", "statement", "by"}], "refused": [{"claim_set", "index",
"session", "statement", "reason_code", "reason"}]}`. A live claim carries `absorbed_by`, the release that first took it in, once one has, and `retained`,
whether the base of the policy's generation holds it. `claims ledger` also has `"generation"`:
`{"epoch", "base", "retained", "current"}`.
`absorb`: `{"policy", "generation", "roles", "intake", "extract", "claims", "live", "pending",
"rebuilt", "kits", "dataset", "replayed_records", "rehearsal", "candidate", "stance", "supervised", "gate",
"stance_rates", "release", "absorbed", "consolidation", "stopped", "dry_run", "release_asked",
"run"}`, each stage's report under its name; `consolidate`: `{"policy", "generation", "live", "closing",
"retained", "consolidated", "stopped", "run"}` where `consolidated` (also `absorb`'s `rebuilt` and
`consolidation`) is `{"mode", "champion", "from_epoch", "epoch", "from_base", "closing", "retained", "rebuilt",
"dead", "dataset", "replayed_records", "rehearsal", "steps_completed", "initial_loss", "final_loss",
"weights_digest", "gate", "release", "stopped"}`; `gate` is `{"claims": [{"claim",
"candidate": {"passed", "measured"}, "champion"}], "answered", "gained",
"unanswered", "regressed", "improvement", "retention", "anchor", "stance",
"serve", "passed"}`, and `anchor` and `stance` are absent when no suite is frozen
or no stance pass measured anything. `stance` is `{"requested", "cap", "dataset",
"path", "hold", "accept", "decline", "held_out"}`, each kind `{"items",
"held_out", "pairs", "written_rejections", "refused"}`; `stance_rates` is
`{"hold", "accept", "decline"}`, each `{"candidate", "champion", "counts"}` as
fractions with their counts, and absent where nothing was measured; `supervised`
is the candidate the preference pass started from, and `candidate` the one it
made. `claims ledger` adds `"stance"`: per release that absorbed a claim, the
`rates` its gate measured.

`status`: `{"state", "policy", "recent_runs", "counts", "concepts"}` - `policy` is
`{"reference", "model", "base", "adapter", "release", "generation", "generation_base"}` (`adapter` and
`release` are `null` until releases exist, `adapter` also for a consolidated checkpoint, which is the
base itself; `generation` is 0 on the configured base); `recent_runs` lists up to five runs as `{"id", "command",
"status", "started_at", "updated_at"}`; `counts` is `{"sources",
"task_sets", "tasks", "experiences", "experience_sets", "datasets",
"candidates"}`; `concepts` is `{"policy", "concepts", "measured",
"weakest", "queued"}`, each of `weakest` `{"concept", "current",
"releases"}` with `{"release", "graded", "passes", "rate", "since"}` per
release.

`doctor`: `{"checks": [{"name", "verdict", "detail", "fix"}]}`. It answers whether
a run can start here, before one does, and loads no model, uses no network and
creates nothing under the state root. The checks are `state-writable` (a probe file
is written and removed in the state root, or in its nearest existing parent when it is
not created yet), `free-space` (a failure under 1 GiB, a warning under 10 GiB: an
estimate for a small base, not a bound), `gpu` (a warning when no card has room
for the policy's weights right now, because brain then runs the model on the CPU
backend without a word, or when there is no GPU or its free memory cannot be read;
it counts the weights, not the activations a run adds) and `role:<name>` for each
role, which fails when the model the role would get does not resolve or its weights
are not on disk. `verdict` is `ok`, `warn`
or `fail`, `fix` is `null` for `ok`, and the exit status is 1 when any check fails.
`learn` runs the state-root checks itself before it starts: a state root that cannot be
written, or a disk with less than 1 GiB free, is refused (exit status 2) with the check's
fix, before anything is captured or generated. A `--dry-run` only plans and does not refuse.

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
| task | `taught_by` | the claim it was made from (`absorb`) |
| claim | `evidence` | each span of the person's words it rests on |
| task set, experience set | `member` | each task, experience |
| experience | `attempts`, `ran_in`, `solved_by` | its task, environment snapshot, solver model |
| experience | `critique_of`, `retry_of`, `revision_of`, `preferred_over`, `variant_of` | the experience its relation names |
| verdict | `verdict_on`, `produced_by` | the experience it grades, the verifier that gave it |
| dataset | `projected_from` | each experience, task and source content its manifest names |
| candidate, release | `trained_on`, `trained_from`, `replayed`, `rehearsed`, `adapter` | its datasets, parent release, replay sample, rehearsal dataset, adapter digest |
| replay sample | `sampled_from` | each earlier release it drew from |
| release | `release_of` | its candidate |
| answer | `answered_with`, `answered_by`, `open_book` | the release, the model, the source shown |

With `--json`: `{"nodes": [{"id", "kind", "label"}], "edges": [{"from",
"to", "relation"}]}`. The artifact asked about is the first node; every
node reached follows once, in walk order; an edge means `from` was derived
from `to`, whichever way it was walked. `kind` is one of `source`,
`content`, `span`, `claim`, `task`, `task_set`, `experience`, `experience_set`,
`environment`, `model`, `verdict`, `producer`, `dataset`, `candidate`,
`replay`, `release`, `adapter`, `answer`. Artifacts without an address of
their own have ids of their kind: `span:<content-hex>:<start>-<end>`,
`model:<identity>`, `verdict:<experience-hex>:<n>`,
`producer:<name>@<version>`.

## Exit status

0 done; 1 the work failed or stopped short of what was asked (a candidate
the gate blocked, an `absorb` that released nothing, a session file refused, a session the extractor gave nothing
usable for, a `doctor` check that failed); 2 refused
before anything ran (usage, an unknown id, a remote model without the
opt-in); 3 a sentence was asked back.
