<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- The README says what the claim gate does with numbers written in words and sentence-initial names in a statement: it
  does not check them, so it admits more; a spec pins it.
- A run that monitors nothing (`train --eval-every 0`) no longer hands brain the default patience, which brain refuses
  when there is no evaluation to be patient about; it names none.
- A number that ends a sentence ("on port 9090.") is a standalone number to the traceability rule that
  admits claims and task answers; its full stop was read as a decimal point, so a claim whose quote ended
  in the number was refused as stating an unsupported number.
- The release gate's serve check no longer loses every served verdict: the
  served answers are decided (the judge, a model of its own, loads) only after
  the `brain serve` process has exited, instead of while it holds the device,
  where the judge ran out of memory and left each served answer ungraded and so
  "different". A task the server gave no answer to, or a server that stops answering,
  is now "not measured" (with what the server last wrote), never a
  disagreement; an answer the judge abstains on is compared by its text and
  meaning.
### Added
- Claims are paired by the names they share, not by how the extractor worded its question, and a judge
  (`claims gate --judge`, the judge role in `absorb`) decides supersede, reinforce or separate for each pair; without
  one, agreeing statements reinforce and disagreeing ones supersede. A fact said again is recorded as a `reinforced`
  ruling (the live claim stays live; `claims ledger` shows how often it was said again) and is never refused as a
  duplicate; `duplicate` now means the same claim proposed twice from the same words. The ledger's `Ruling` gains
  `reinforced`.
- `absorb <SESSIONS>...`: a day's sessions with an agent become the next release in one recorded, resumable
  run - intake, extraction and the claim gates, a kit of 18 distinct records per live claim (the question and
  eight paraphrases answered by a teacher shown the claim, the hindsight dialogue, restatements, reverse
  questions and consequences) with four paraphrases kept out of training as the claim's own stopping set,
  the base's own answers rehearsed, an adapter trained again from the base on every live claim (so a
  superseded claim simply drops out; `--continue-from-release` continues the release instead), and a gate
  that is a declared rule on counts of claims answered with no regression and no tolerance. A refused
  candidate stays on record and the current release stays in use. `--sealed-probes` refuses any record
  that contains a probe's question or shares an 8-word run with one beyond its claim's statement; the gate
  never reads them. `--dry-run` stops after the rulings. `claims ledger` shows the release that first
  absorbed each live claim. The front door reads a sentence about sessions as `absorb`. Procedure claims
  are stored and not trained on.
- The `taught` task kind: a question a person's claim answers, closed-book, whose reference is the claim's
  statement. It is graded by the new `terms` verifier, which passes an answer that carries every number,
  name and quoted term of the statement and adds none the task did not give (the rule that admitted the
  claim, applied to the answer), and leaves a statement with no such term to a judge. `claim_task` makes the
  task from a live claim with the spans of the person's words as evidence, each checked against its quote.
- `splinter-data`: a dataset record may fix its side of the training split (`metadata.split` of `train` or
  `held_out`), which the holdout rule never moves and does not count among the units it chooses from;
  `session_dialogue` renders a recorded session as the conversation it was with only the agent's replies after a
  given step supervised.
- `session add`, `claims extract`, `claims gate`, `claims list|show|ledger`: the first three stages of
  learning from a person's own sessions with an agent, each recorded as a run and repeatable. Sessions
  (ATIF files, or directories of `*.atif.json`) are validated, refused with the step and reason when the
  training projection cannot render them, stripped of secrets before they are addressed, and stored once
  with one part per step text. A model proposes claims (correction, fact, procedure) with the person's
  exact quotes; code admits a claim only when every quote is verbatim in a user step, every number, name and
  quoted term of its statement is in the cited words, and no agent sentence is evidence. Repeats collapse, a
  later claim on the same question supersedes the earlier, and every ruling, refusal included, stays in a
  ledger with its reason. `splinter-data` names why the projection refuses a trajectory
  (`projection_refusal`), and the release fixtures' `brain` stand-in script moved to its own file to keep
  the fixture module under the file-size limit.
- `exam CANDIDATE --exam-set EXAM --deployed-only` asks only the candidate under the prompt it is deployed
  with, for a candidate to be paired offline with the base and prompted arms of another report of the
  same exam instead of asking them again.
- `train --keep-evaluations` keeps the adapter of every evaluation with the candidate; `splinter select`
  puts each to the dev suite and chooses one by a fixed rule (no more than 8% invented specifics,
  then the best answers within a standard error, then the writer's text of the dev families by
  loss); `--adopt` makes it a candidate of its own and `learn --select-on-dev` does both in the run.
  Needs the brain commit that writes an adapter at each evaluation.
- `learn --abstain SHARE` (with `--with-passages`): that share of the dialogue records become the
  writer's abstentions, for a retrieval miss or a question beyond the writings, written by the
  base as the writer, admitted by code, and kept in the family of the record they were made from.
- `learn --describe-voice` and `dataset build --describe-with REF`: the writer's chunks are asked
  for by a model-written description of their content (20 to 60 words, refused when it carries
  the passage's phrasing; the chunk keeps the heading-and-opening request otherwise), kept per
  generator and reused by later builds.
- The powered exam. `learn --exam-families N --exam-tasks-per-family N --dev-families N
  --dev-tasks-per-family N --exam-resamples N` (fifty final-test families of eight tasks and
  fifty dev families of four by default for a persona run) reserves both up front and writes it from the
  reserved text alone; `splinter exam-set create|show` makes and shows a frozen exam apart
  from any run; `splinter exam CANDIDATE --exam-set EXAM` puts a candidate to it: base,
  prompted base, candidate and candidate under its prompt, every answer and verdict kept,
  per-task pairing with family-clustered intervals, the sign test over tasks and families,
  a like-length test, invented specifics, answer lengths, a power report with the exam's own discordance and intraclass correlation, the primary
  comparison fixed with the exam and the secondary ones Holm-corrected, a judge-free voice
  score, and the judge's false passes on harder wrong answers.
- `splinter exam-report labels-export|labels-import|memorisation`: a blind stratified sample of an
  exam's answers for a person to label, what the labels say of the judge (agreement, kappa, false
  right and false wrong rates, length bias), and each arm's overlap with its training text.
- `splinter exam-set power` simulates the exam's planned paired test to size it, and
  `exam --pilot-families N` puts a candidate to some of the families to estimate the exam's
  discordance and clustering first.
- `reserve`: the exam's families are reserved before any task is generated or dataset
  built. A family is the group of text parts that print the same text (named as a split
  names it); a stable hash of the name picks them, and the text of each, every edition of
  it, is left out of the source everything after reads (the source records the parts as
  skipped for being reserved) while the exam is written from a source of those parts alone.
  It refuses, naming why, when too few families can be examined or reserving them would
  leave less to learn from than is examined.

### Changed
- The gate's improvement report adds what the tasks themselves say beside the family-level sign
  test it decides on - a difference in the share right with an interval that resamples families
  and an exact sign test over the tasks - for the whole suite and for each of its two parts.
  The decision rule is unchanged: the family-level sign test stays the rule, because a report is
  not a change of rule, and its p-value moves with a single family when only a dozen are
  discordant (so the task-level numbers are what to read it by).
- The `stated` verifier also reads a whole number from 21 to 99 written as tens and units
  (`forty-two`, `forty two`) as the same number in digits.
- A judge decodes greedily (the judge's version is now 4, so an earlier calibration
  is measured again): one answer gets one verdict, whichever time it is judged.
- The anchor check reports the bootstrap interval of the drop over the paired
  items beside the point estimate, and how many paired items a suite would need to see a
  drop of two points at the discordance it found; the 0.02 bound is unchanged.
- The release gate's improvement check still decides over the held-out
  records and the variants of trained tasks together, but its report keeps
  them apart: generalisation to held-out records and recall under paraphrase
  each carry their own comparison and sign test. The gate report states the
  system prompt each arm was asked each suite under, and the anchor suite is
  asked of both arms under the default prompt, so the anchor no longer
  compares a persona-prompted candidate with an unprompted base.
- The `stated` verifier no longer fails an answer that states the reference
  but runs past its length bound: it abstains, so a judge adjudicates it. It
  also reads sub- and superscript digits as plain digits and a whole number
  written in words and in digits as one, so a subscripted H2O and "8 legs" state `H2O` and
  `eight`.

### Added
- The timeline pipeline (`splinter_sdk::timeline`): import and split stages, a train stage for `timeline-v1` datasets (immutable candidates packed into one deterministic file), an evaluation stage scoring a candidate against a champion on the same held-out units with participant-clustered bootstrap differences, a release stage through the predictive gate that records a failing candidate as rejected, and the lineage of a release down to source file lines; `samples/health` runs it on a synthetic cohort.

### Fixed
- The powered exam stores its answered and graded report (as an `exam-report` artifact, announced as the
  `exam-answered` stage with its path) before the voice stage starts, and the voice stage no longer
  takes the run down: each arm is scored with the device given back first, one arm at a time, a
  device failure while scoring is returned as the `voice_error` of a complete report instead of a
  panic, and the arms scored before a failure keep their scores.
- A stored document of measured floating-point numbers was refused on read as altered: serde_json now parses floats exactly.

### Added (earlier)
- Rehearsal of the base model's own answers, against the forgetting a persona
  fine-tune shows on the anchor suite: the `rehearse` stage of `learn` (and
  `splinter rehearse --records N`) builds a dataset of general tasks the base
  answers itself - `arithmetic` and `format` tasks built by code from a seed
  with their references, and general requests the base writes one domain at a
  time - answered by the base with no adapter under the default prompt,
  decoding greedily, graded by the kinds' code verifiers alone; the new
  `rehearsal` view keeps every answer not decided wrong. No anchor task, nor a
  near copy of one, is ever in the set, and the gate's leakage check reads the
  rehearsed records. `learn --rehearsal SHARE` (default a quarter of the
  training draws for a run with a persona; `0` turns it off) and `train
  --rehearsal DATASET-ID [--rehearsal-share F]` mix the records in at their own
  share, never held out, and put a monitoring share of them into the
  monitoring set, so the step carried balances the new records against the
  base's answers. Candidates, releases and the lineage record the rehearsal.
- `train --seed N` and `learn --seed N`: the seed of the adapter's
  initialisation and the batch order, so two runs that differ only in it
  show how much of a measured difference is the draw.
- `eval` names the tasks a model got wrong on each suite, with the answer it
  gave (`missed`), so what a candidate lost against the base can be read off
  two evaluations.
- A supervised training run is watched as it trains: a share of its training
  families (`--monitor-share`, default a tenth, whole families, never the
  held-out ones the gate and the exam decide on) is scored every
  `--eval-every` steps and at the last step, the candidate carries the adapter
  of the evaluation with the lowest monitoring loss instead of the last
  step's, and the run stops once that loss has gone `--patience` evaluations
  (default four) without improving. The candidate's record and the `train` and
  `learn` reports carry the curve (step, mean training loss of the interval,
  monitoring loss), the step carried and why, and the generalisation gap at
  it, and warn when the gap is more than a quarter of the monitoring loss,
  when the monitoring loss rose after the step carried, or when it was still
  falling when the budget ran out; the release gate (`warnings`) and the exam
  (`training_warnings`) repeat the warnings beside their verdicts.
- The `voice` view: the writer's own text as training data, with no model in
  the loop - every text part of the sources cut into stretches of whole
  paragraphs, each a record in the shape the policy is asked in (the persona
  prompt, one user turn, the writer's words word for word as the answer), naming the part it prints
  and no experience. A `learn` with a persona trains on it beside the dialogues
  by default, `--voice SHARE` of the examples (half; `0` turns it off), chosen
  as an even spread over the parts; `dataset build --view voice [--limit N]`
  builds it by hand, and `--limit N` thins any view to an even spread.
- The tasks stage reports how many of the sources' text parts it generated
  from, how long it ran and, when the budget stopped it, what covering every
  part would take at that pace.
- The anchor suite freezes several files as one version (`eval --suite anchor
  --freeze A --freeze B`), and takes two more kinds graded by code alone:
  `arithmetic` (the last number the answer states is the reference) and
  `format` (the answer has exactly the lines the reference counts), what a
  persona fine-tune erodes before it forgets facts. Both are task kinds in the
  catalogue, graded by `splinter-eval`'s new form verifiers.
- **`splinter-core`.** Splinter's vocabulary - digests, the clock, sources,
  tasks, experiences, annotations, the chat wire shapes, the system prompt and
  the self-containment rule - in a crate that performs no I/O and names no
  store, model or agent runtime.
- **`splinter-expdb`.** A new standalone crate for a versioned experience graph
  database: immutable files, manifests as the transaction layer, snapshots,
  modality-neutral episodes of streams, queries and training views, with its
  storage protocol model-checked in TLA+.
- `splinter state status` and `splinter state maintain [--collect]`: what the
  experience database holds as files, and the merging, indexing and collection
  that keep it small and quick to open.
- `splinter state archive`, `restore`, `verify` and `repair`: the database and
  the files it tracks pack into one deterministic archive (incremental with
  `--since`), restore verifies before it puts anything in place, and what has gone
  missing is found and filled from any copy with the right digest. What no copy has
  is written off in a ledger only when asked.
- Every experience is also an attempt in the database's graph, a verdict is
  ranked evidence about that attempt, and a dataset, training run and release
  record where they came from, so a release traces back to the experience it
  learned from.

### Changed
- The voice view is the whole corpus of the writer's text as chunks of 250 to
  650 words: whole sentences, closed at paragraph ends, never a heading alone,
  each asked for by a request built by code from its recipient, year and
  opening, half of them under the identity line instead of the persona prompt,
  one print of each text, no family over a tenth of the tokens. `--voice` is
  now a share of the training tokens (default 0.7) and `dataset build` takes
  `--writer`, `--token-limit` and `--max-family-share`.
- A handful of answers of another strength, kind or concept no longer switches
  on a quota cap for the rest of a pool: a group under one candidate in twenty
  counts for none of a dimension's groups, so two formal answers beside 163
  judged ones no longer discard 39 of them.
- The training recipe: `train` and `learn` share one learning rate (2e-4;
  `train` used brain's 3e-4), the LoRA alpha is twice the rank instead of 16
  at every rank, the weight decay on the adapter is 0 instead of 0.1, and
  `--lr`, `--alpha` and `--weight-decay` name them on both commands; the
  resolved values are in the candidate's plan. brain now takes a step as a
  mean over its supervised tokens, draws every record once per epoch, warms
  up over a twentieth of the steps, and cools the rate down to its floor when
  a run stops on its patience.
- No record of a dataset goes unused by the splits. The writer's text of a
  held-out family is written to its own file (`held_out_text.jsonl`) and the
  writer's text of a monitoring family is monitored (`monitor_text.jsonl`)
  instead of being dropped, so the held-out, monitoring and training files
  partition the records. The monitoring set spans at least three families
  (within twice the monitoring share) where the data has them, and its
  loss, a mean per supervised token, now includes the writer's text, which is
  most of the tokens of a persona run.
- The step count of a training run is a budget, not a target: `--steps` (now
  optional on `train` too) is the most steps a run may take, by default three
  passes over the examples instead of two, and the monitoring decides where
  the run stops and which step it carries. What a step averages follows the
  size of the dataset on `train` as it did on `learn` when the steps are not
  named.
- A held-out split chooses its families among the records that can be
  examined (those projected from a task or an experience); a record of the
  writer's own text follows its family, held out with it or trained on, and the
  held-out score is measured on the examinable records. A family is named by
  the least content digest of its texts over every text part of the sources,
  the same in every dataset built from them.
- The budget's stage shares adapt to the share of the examples the writer's
  own text carries: the tail (training, the exam, the gate) grows by it and the
  open-ended stages give up theirs in proportion.
- A supervised candidate's replayed records take a fixed quarter of the
  training draws instead of joining a plain union, so a large replay set no
  longer starves the new records.
- The exam reports, beside its paired sign test, a bootstrap interval of the
  candidate's gain per family of sources.
- Texts are grouped as one print by passages shared anywhere in either, not
  only in their first seven hundred words: a letter one edition prints inside
  its neighbour, or a passage reused deep inside a longer text, now keeps the
  two on one side of every held-out split, the samples' included.
- `learn` is a pipeline of stages run by an engine, which checks for a cancel and
  the budget before each stage, skips a stage that does not apply, and records every
  stage with how long it took and why it failed when it did (`runs show`). Models play
  named roles - policy, teacher, generator, planner, judge, critic, router - decided by
  one rule, and a run records who played each. Each sentence of a REPL resolves
  `policy:<alias>` afresh.
- **`splinter-store` keeps** sources, tasks, experiences, annotations and sets
  in the experience database instead of one file per object. Annotations are ranked evaluations; a relation is also an
  edge. Splinter's own content addresses are blake3 (`blake3:<hex>`); digests a
  tool reports, such as an adapter's, stay `sha256:`. State written by earlier
  builds is not read: regenerate it.
- Datasets, candidates, releases, aliases, answers, the anchor suite, judge
  calibrations, the curriculum queue and frontier measurements are kept in the
  experience database too. Adapters and dataset files stay plain files under
  `<state>/artifacts`, content-addressed and tracked by the database. An alias is
  a pointer whose history is kept, claimed by compare-and-set, so there is no lock
  file. Releasing a candidate again completes an interrupted release instead of
  storing a second one.
- Run records are events in the experience database and a cancel request is a
  signal, instead of files in a run directory.
- A bulk stage (solve, verify, critique) commits its writes in groups instead of
  once per record.
- A judge is trusted for what its verdicts do: its passes must be precise where
  they admit answers to a training set (verify, teach, critique, author), and
  both its passes and its fails where two models are compared on them (the
  gate, the exam). `judge measure` reports both fitnesses.
- The held-out split runs to eight groups of overlapping source text, newest
  first, within a quarter of the records, so the gate's and the exam's paired
  tests over families can reach significance; a tenth and whole groups as
  before.
- A learn that trains a person's policy tells the generator the person wrote
  the sources (`tasks generate --author NAME` by hand), so a question about
  the writer names a subject the source gives; the teacher's answers and the
  student's attempts run under the person's prompt, the one the records open
  with.
- An instruction quotes its source at a run of eight words, not twenty-four
  characters, so a question may name a matter in the source's own words; a
  writer's passage is its own evidence, cited by section and written once.

## [0.1.0] - 2026-10-01

First release. Splinter is a learning agent with its own model: it is told
in plain language what to learn, learns it, measures that it did, and
releases a better version of itself. It is built on sven 3.0.0 (the agent)
and brain 2.0.0 (the model), pinned in `Cargo.lock`.

### Added
- **Command line.** `splinter "<sentence>"` and a REPL, plus one verb per
  stage: `source`, `tasks`, `solve`, `verify`, `critique`, `dataset`,
  `train`, `eval`, `release`, `runs`, `experiences`, `lineage`, `ask` and
  `status`. `learn` runs every stage as one recorded run. Every command
  takes `--json`, and a model is named the same way everywhere
  (`policy:`, `release:`, `local:`; `remote:` only with `--allow-remote`).
- **Sources.** Documents, repositories and the captured output of a
  command, split into sections that tasks cite by position and byte range.
- **Tasks that name their subject.** A generator model writes tasks from
  the sections, shown what the source is called. A question must name the
  product, tool or version it is about, so it has one answer without the
  source in front of the student; two questions about one subject with
  different answers are both left out. Several task kinds ship (recall,
  explain, predict, construct, debug, and more), and `denoise` needs no
  model.
- **Learning where it counts.** Each task is attempted closed-book several
  times. Tasks the policy always solves are dropped, and a task it never
  solves is solved by a teacher shown the source passage. The teacher is
  the policy itself unless `--teacher` names another model. The training
  record keeps the question and the verified answer, never the passage.
- **Verifiers by strength.** Executable checks (code seen to run to its
  end, with generated tests validated against mutants), a recall check that
  accepts an answer stating the reference, a consistency check, and a
  calibrated judge. A judge alone never makes a training record.
- **Critique and retry.** A failed attempt becomes a critique and a
  revision, kept as relations between experiences, so preference data comes
  from real attempts.
- **Reworded questions.** The generator rewrites each kept question in
  other words, keeping its subject. These are never trained on and
  measure whether a fact was learned rather than a sentence memorised.
- **Training.** A candidate continues the current release as a LoRA
  adapter, by supervised fine-tuning with a replay of what earlier
  releases learned, or by preference training on verified against failed
  answers. Experiences project into ten views, with the passage and other
  privileged material stripped where a student must not see it.
- **Release gate.** A candidate is released only when all four checks
  pass: it improved on reworded held-out questions (sign test, after
  leakage exclusion), it kept what earlier releases learned, it held a
  frozen anchor suite of general questions, and it gives the same answers
  on plain `brain serve --adapter`.
- **Releases.** Immutable: an adapter plus a manifest recording the base
  model and its digest, the parent release, the datasets and every number
  the gate measured. `release:` and `policy:default` name them, and
  `splinter rollback default` returns to the previous one.
- **Lineage.** `splinter lineage <ID>` walks from an answer, release or
  dataset up to the source, part and byte range it came from, and down to
  everything that came from it.
- **Curriculum and integrity.** Concept mastery is tracked over releases
  and forgotten concepts are queued for new tasks. Held-out questions
  that near-duplicate a training record are excluded, and an experience's
  code calls can be replayed in their recorded environment.
- **One resident model.** The policy, a candidate and the champion are one
  base with different adapters, so one copy of the base is loaded.
- **Default policy.** Qwen3-0.6B from brain's model store;
  `BRAIN_QWEN_WEIGHTS` names another checkpoint.
- **Building.** Builds from the pinned `Cargo.lock` with `make build`;
  `make local` builds against local sven and brain checkouts instead.
