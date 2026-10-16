<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Persona program: the study, the records and the real authors

Work items PP-10 to PP-27 of the persona program (index and order in
`persona-program.md`, findings in `persona-findings.md`, seeds in
`persona-seeds-and-health.md`). Items PP-10 to PP-15 change what the next runs
produce and come before any new training run; PP-16 to PP-23 are the runs and
analyses; PP-24 to PP-26 carry the method to real authors; PP-27 is the paper.

## PP-10 Training records list the options the case offered

- **Status.** Done. The case file carries `open`; the reader extracts it
  (`MinedCase`, `ReadCase`, `Verified`); the reading gate refuses a reading
  whose open set differs from the plan (`MisstatedFact { fact: "open" }`) and
  `verify` refuses a case whose open ways omit what was done or disagree with
  the offer; a record lists exactly the ways read, in an order shuffled per
  decision; `audit` reports the open-set agreement beside the decision
  agreement. A property test over every planned episode of seeds 101, 202 and
  303 finds no label the oracle would not give on the facts and options listed.
  What remains: no corpus written before this change can be used (its
  decisions carry no open set and fail to load); new corpora are written under
  PP-16, and PP-21 measures the gate's higher rejection rate.
- **Problem.** A record of a case with an outside offer lists four options
  and labels the decision made among three, contradicting the oracle on the
  prompt it shows (PF-01, PU-02).
- **Evidence.** `samples/synthetic-author/src/dataset.rs` (`records_for`
  lists the ways the reader found open) and `src/scenario.rs` (`options`
  removes one of three ways when help is offered). The case file the writer
  is shown (`src/writer.rs`, `writer_input`) now carries them, so the essay
  and the miner can recover the option set.
- **Acceptance.** The case file the writer gets names the ways that were open;
  the reader extracts them like the other facts; the reading gate
  (`src/gates.rs`, `check_reading`) checks them against the plan; a record lists exactly
  the options read; a property test over every planned episode of three seeds
  finds no record whose label differs from `Persona::decide` on the facts and
  options it lists; the extraction audit reports the option-set agreement.
- **Design.** An `open` field in `Episode::case_file` (`src/episode.rs`),
  `open` in `MinedCase` and `ReadCase`, a `MisstatedFact { fact: "open" }`
  rejection. The same shape is what a real author's decision record needs
  (PP-24): the alternatives a person had are part of the situation.
- **Depends on.** Nothing; new corpora are written under PP-16.
- **Risk.** Essays must now state the alternatives; the gate's rejection rate
  rises, which PP-21 measures.
- **Size.** M.

## PP-11 Every exam outcome is bound to its item and its model

- **Status.** Done for `synthetic-author`. Each outcome carries the digest of
  its item (rendered prompt, tool definition, expected label) and of its model
  (base and adapter digests, decoding, decoding seed); `exam/manifest.json`
  records, per arm, the items digest, `--articles`, `--per-cell`, the reply
  limit, the arm's system turn and the model, and a rerun under any other
  manifest, adapter, or an adapter retrained at the same path is refused with
  both digests named. `report` and `cross` fail on a mismatch, an empty cell
  prints `not measured`, and a requested arm without outcomes is named. What
  remains: the paper's `collect.py` and `confusion.py` do not yet join on the
  digests.
- **Problem.** Outcome files carry an item id that names different items under
  different flags and nothing about the model (PF-02); empty cells print as
  zero and missing arms vanish (PF-12).
- **Evidence.** `samples/synthetic-author/src/arms.rs` (`Outcome`,
  `run_items_with`), `src/binding.rs` (the digests and `bind`), `src/lab.rs`
  (`run_arm`), `src/report.rs` (`render`) and `src/exam_cmd.rs` (`report`,
  `cross`).
- **Acceptance.** Each outcome records the item's digest (rendered prompt,
  tool definition, expected label) and the model's identity (base digest,
  adapter digest, decoding and seed); `run_items_with` refuses an outcome file
  whose manifest or item digests differ; `report`, `cross` and the paper's
  `collect.py`/`confusion.py` join on the digest and fail on a mismatch; an
  empty cell prints as not measured; a requested arm with no file is named.
  Tests: a rerun with another `--articles`, another adapter and a retrained
  adapter at the same path are each refused.
- **Design.** The exam manifest of PP-07 plus a digest per outcome line.
- **Depends on.** PP-07 (shared manifest).
- **Risk.** Old outcome files lack digests: readable for the paper's
  reproduction only through an explicit adoption that recomputes the digests
  from the flags and records that it did.
- **Size.** S.

## PP-12 Failures accounted for, not mistaken for content

- **Status.** Done for `synthetic-author` and the job log. A writer or reader
  request lost in transit (a rate limit, a 5xx, a timeout, a dropped
  connection, a router's fallback refusing after the first choice failed; not
  another 4xx) is repeated up to `--transport-retries` times (default eight)
  with a pause that starts at `--transport-backoff-secs` (default five),
  doubles up to a minute or follows a `retry-after` figure the error carries
  (`src/transit.rs`), and uses no attempt; the miner is repeated the same way; each loss
  is a `rejected.jsonl` line of kind `writer-unavailable` or
  `reader-unavailable`, and only `content` lines reach `previous_refusal`.
  Past the bound the run stops and says which model could not be reached, and
  the article is counted under `unavailable` apart from `abandoned`. A draft
  left unread by a spent budget is logged as `budget`. A reader answer in no
  readable shape is a `content` rejection of its own (`Unreadable`), no
  longer a decision with an invented case name. `mine` fails on an append
  error and reads an article again for the cases it lacks. `recheck` fails
  naming the line of a row without an article index (`recheck.rs`).
  `jsonl::read` leaves a cut-short last line out and says so, `read_with_tail`
  returns it and `Log::open` cuts it off. An unreadable `usage.json` fails
  `corpus` before any request is made. The transport failures are scripted
  through a model double and the pause is injected, so no test sleeps. Not
  tested: the failure of an append to `decisions.jsonl` itself (the test
  covers the resumption of an article that is partly logged). Not done: rows
  written before `kind` existed carry none, and the counts of rejections by
  reason behind the paper's `corpora.csv` have not been redone with the kinds.
- **Problem.** Writer and reader transport failures consume attempts and are
  logged as content rejections (PF-06); the miner swallows an append failure
  and never completes a half-logged article (PF-07, PU-01); the job log cannot
  recover a torn tail (PF-08); an unreadable cost file resets to zero
  (PF-13); `recheck` defaults a missing index to article 0 (PF-20).
- **Evidence.** `samples/synthetic-author/src/writer.rs` (`Run::ask`,
  `Run::write_article`, `Run::judge`), `src/lab.rs` (`mine_all`),
  `src/cache.rs` (`Kind`, `note_rejected`, `usage`), `src/recheck.rs`,
  `src/corpus_cmd.rs` (the flags and their defaults, named constants in
  `src/writer.rs`) and `crates/store/src/jsonl.rs` (`read_with_tail`,
  `Log::open`).
- **Acceptance.** A transport failure is retried with a bounded backoff and
  does not use an attempt; after the bound the run stops and says so;
  `rejected.jsonl` gains a `kind` (`content`, `writer-unavailable`,
  `reader-unavailable`, `budget`) and only `content` reaches
  `previous_refusal`; an article given up for infrastructure is counted apart
  in the summary; `mine` fails on an append error and resumes an article from
  the first case it lacks; `jsonl::read` reports a torn final line as such
  (and `Log::open` truncates it after saying so) so the next append starts a
  new line; `usage.json` that cannot be parsed fails the command. One test per
  behaviour, the transport failures injected through a scripted model.
- **Design.** In the store's `jsonl` module for the log (it serves every
  sample); the rest in the sample.
- **Depends on.** Nothing.
- **Risk.** Low; the paper's `corpora.csv` counts `rejected.jsonl` lines by
  reason and must be recounted with the kinds when a corpus is rewritten.
- **Size.** M.

## PP-13 Examination v2: balanced positions, trivial baselines, a trained-on axis

- **Status.** Done for `synthetic-author`, except the training-prior baseline
  and the freeze. `exam --exam v2` (the default) draws the same situations as
  `--exam v1` (the examination as reported, which keeps its digests) and lists
  each item's options in an order drawn from the exam seed, with the right
  answer at each position equally often within every cell to within one item.
  Version 2 is a different examination, recorded in the arm manifest, so
  outcomes of version 1 are refused under it. Each version-2 item carries
  `combination_trained`, read from the kept decisions of the corpus; the
  examination refuses to be drawn before `mine` has read them, and `report`
  does not score the baselines when the items drawn again no longer match the
  digest the arms were asked. `report` scores two constant policies on the
  arms' own items, the option listed first and the option the beliefs
  determine most often, beside the arms and as sign-test comparisons with the
  tuned arm. `items` shows the listed order and the right answer's position.
  Not done: the training-prior baseline; a gate that freezes the exam file
  before the training it examines (the manifest records its digest and
  refuses a different one, but nothing forbids drawing it after training);
  the paper's figures are unchanged.
- **Problem.** The expected answer sits first in about half the items, so a
  constant policy beats the prompted arms and the reported chance level is the
  wrong reference (PF-05); `combo_seen` is documented as "seen" but means "not
  withheld" (PF-14).
- **Evidence.** `samples/synthetic-author/src/scenario.rs` (`options` sorts
  them), `src/exam.rs` (`items`, `draw`, `Version`, `Trained`),
  `src/report.rs` (`baselines`), `src/lab.rs` (`exam_items_of`,
  `asked_items`) and `papers/persona-training/sections/06-results.tex:8`.
- **Acceptance.** A new exam version (the reported v1 stays reproducible
  under its name): option order drawn per item from the exam seed, the
  expected position balanced within each cell; letter labels follow the
  listed order; the report prints the first-listed and majority-option
  baselines beside the arms; each item carries `combination_trained` (some
  kept, mined decision has its facts and options) beside `combo_seen`, which
  means only that the combination was not withheld. Open: the training-prior
  baseline (what a model that knew only the corpus's decision frequencies
  would score), and the exam file hashed and frozen before the training it
  examines.
- **Design.** `draw` gives each cell its right-answer positions in seeded
  permutations, one per block of as many items as there are options, and
  lists the other options after it in a seeded order; baselines are computed
  in `report` from the items alone.
- **Depends on.** PP-07 (exam seed), PP-11.
- **Risk.** Results on v2 are not comparable item for item with v1; the
  paper reports v2 for new runs and keeps v1 for the published ones.
- **Size.** M.

## PP-14 Records that teach stopping after the tool answers

- **Problem.** A training record ends at the assistant's tool call; nothing
  shows the model what to do once the tool has answered, so in a multi-round
  chat a tuned model calls `decide` again after `Recorded.` unless the tool is
  marked final. Persona-chat's tools file defaults `final` to false.
- **Evidence.** `samples/synthetic-author/src/records.rs` (`chat_record`:
  system, user, assistant call; no tool turn); `samples/persona-chat/src/tools.rs:37-38`
  (`final` defaults to false), `src/session.rs:174` (the exchange continues
  while calls are made and no final tool answered). The repeat is the reported behaviour of the tuned
  adapters in persona-chat; no recorded trace of it is in the repository.
- **Acceptance.** Each record continues with the tool's result and a short
  supervised closing turn with no further call; persona-chat batch over the
  exam's prompts with a non-final `decide`: the share of exchanges that call a
  tool again after a valid call is reported per arm, and the tuned arm's share
  is no higher than the base's; the decision accuracy of PP-16 does not fall.
- **Design.** In `data` (the chat record projection) and the sample's
  `chat_record`; brain must supervise an assistant turn that follows a tool
  result in the same record, which the agent-loop roadmap lists as open in
  brain (loss masks for multi-step tool conversations): that brain change
  comes first.
- **Depends on.** The brain loss-mask change.
- **Risk.** If brain trains the closing turn under a mask that also covers
  the call, the call's supervision changes; test the mask on the record shape
  before training.
- **Size.** M.

## PP-15 Motivations worth learning, and a measure of them

- **Status.** Done for `synthetic-author`; the corpora in use predate it.
  `dataset` makes no record for a decision without a usable reason (its
  summary counts them in `without_reason`), and a reason that holds a six-word
  run of a brief phrase in either voice is not used. The brief-copy gate refuses
  an essay that repeats the first-person form of the phrase for what was done.
  `report` lists, per arm, the reasons given, those that name the fact the
  deciding belief turns on (`Tenet::fact` and `Fact::terms`, checked against
  the verdict table by a test), those that repeat a brief phrase, and the
  number of distinct reasons.

- **Problem.** Some records teach an empty motivation, and the adapters'
  reasons repeat the brief's private phrase verbatim (PF-15). Nothing measures
  a reason.
- **Evidence.** `samples/synthetic-author/src/dataset.rs:33` (a decision with
  no quoted reason gets `""`); `src/episode.rs:47-54` (`did_phrase`);
  `src/gates.rs:253-262` (the brief-copy check in `check_form` runs on the
  third-person `did_phrase` only).
- **Acceptance.** No record has an empty reason (a decision without a quoted
  reason is not a record); the brief-copy gate also refuses the first-person
  forms of `did_phrase`, and records refuse a reason holding a 6-word run of
  any of them; the exam reports per arm: the share of reasons that name the
  fact of the deciding belief (a fixed term list per tenet, code-graded), the
  share that copy a brief phrase, and the number of distinct reasons.
- **Design.** The tenet-to-fact map is the same table `Belief::verdict`
  reads (`src/persona.rs:139-168`); the measure is code, no judge.
- **Depends on.** PP-10 (new corpora anyway).
- **Risk.** Fewer records per corpus; PP-21 reports it.
- **Size.** M.

## PP-16 Replication: training seeds and author worlds

- **Status.** Built for `synthetic-author`; no stage has been run. `study
  freeze` (`src/study/protocol.rs`) writes `protocol.json` once: the author
  seeds and each author's training seeds drawn, printed and recorded, the digest
  of its canonical JSON, the three model references checked to be three
  families, the arms with the PP-19 ablations, the training settings, the
  primary comparison, the confirmatory family of ten comparisons with Holm's
  step-down over it and the exploratory ones beside, greedy as the primary
  decoding and a sampled one beside it (PP-20); freezing again with the same
  settings reports the frozen protocol and with others is refused, and a family
  the authors cannot confirm (an exact sign test over eight authors reaches
  1/256) is flagged. `study corpus` (`src/study/corpus.rs`) writes an author's
  corpus, mines it, makes the records and freezes and hashes the examination into
  `study.json` before anything is trained; `study train` and `study exam`
  (`src/study/train.rs`, `src/study/examine.rs`) train every adapter at its
  training seeds, one `run.json` each, and put the frozen examination to every
  arm, one directory of outcomes per adapter; each resumes, refuses a changed
  protocol, base, settings, records, adapter file or examination, and is tested
  through scripted models and a scripted trainer. `study report`
  (`src/study/analysis.rs`, `collect.rs`, `render.rs`) reads the manifests and
  outcomes and writes the per-author, per-seed paired sign tests, the spread over
  seeds, the exact sign test across authors with Holm's correction, first-listed
  and majority baselines, item-clustered bootstrap intervals (seed drawn, printed
  and recorded), survivorship (PP-21) and reason measures (PP-15) as CSV and JSON,
  and `papers/persona-training/scripts/study_data.py` maps them to the columns the
  paper's build reads. `study freeze --workers` records how many articles are
  written at a time and `study corpus` stops with the cause when it wrote no
  article or read no decision. The corpus stage has run for one author on
  throwaway studies of six articles (three attempts each, one worker, one author
  seed for every writer) with the reader and miner of the protocol example
  (`google/gemma-4-31b-it`, `meta-llama/llama-3.3-70b-instruct`). The writer
  that keeps its drafts is `deepseek/deepseek-v4.1-flash`: after the prompt and
  parsing fixes below it kept 6 of 6 articles in one run (6 of the 8 drafts it
  wrote) and, with the last prompt, 3 of 3 (every draft it wrote; a reasoning
  model's thinking cost it the output cap in one request of that run and two of
  the earlier one), and the stage ran through to the frozen examination of 480
  items, the records and the survivors report. The others:
  `minimax/minimax-m3` kept 3 of 6 (it leaves facts out, mostly how long the
  matter binds, and sometimes returns an empty body), `inclusionai/ling-3.0-flash`
  kept none of about 23 drafts before the fixes and was not tried after them,
  `nvidia/nemotron-3.5-lightning` spent every call's output cap thinking and
  wrote nothing, and `mistralai/mistral-large-2512` answered 429 from its
  shared upstream pool throughout. Four defects the runs showed were fixed in
  the prompts and parsing, none in a gate: the writer was handed the exact
  phrase for what it did that the brief-copy gate refuses, and copied it with
  the pronoun changed (12 of 18 drafts of one writer); the reader's schema
  listed the facts and the open ways as optional, and it left them out (6 of 11
  readings empty, none of 31 after); the writer named a case in one
  paragraph and told its decision and reasons in the next, whose quotations are
  lost because only paragraphs that name a case are its own (10 of 42 reasons
  of one writer dropped, 0 of 23 after); and the miner wrote briefings under
  the floor in about one case in ten, which turned the case away. The gate still refuses
  ordinary narration that happens to hold six words of a phrase of the brief
  (for example "put it before the assembly and ..."); the writer is shown those
  phrases instead of the gate being relaxed, and one draft of one writer after
  that still tripped it. Not done: no adapter trained under the protocol, and
  the paper's build is still laid out for three authors.
- **Problem.** One training run per author and three authors (paper Sec. 7);
  all three runs used the same default seed (PF-04); no sign test across
  authors is possible with three.
- **Evidence.** `papers/persona-training/sections/07-discussion.tex` (scope of
  validity); `data/training.csv`; the adapters' cards.
- **Acceptance.** A protocol frozen in the repository before any run: at least
  eight authors drawn from fresh seeds, each with its own corpus written under
  PP-10 to PP-15, and three training seeds per author; primary endpoint and
  comparisons as in the paper, plus per-seed paired sign tests, the spread of
  accuracy over seeds (min, max, standard deviation) per author, and a sign
  test over authors of the per-author mean gain (eight of eight gives
  p = 0.0039, seven of eight p = 0.035); the exam (PP-13) frozen and hashed
  before the first training run; every seed recorded (PP-05 to PP-07).
- **Design.** A `study` subcommand that runs the stages from a protocol file
  and writes one manifest per author and seed; the paper's scripts read the
  manifests.
- **Depends on.** PP-05, PP-06, PP-07, PP-10, PP-11, PP-12, PP-13, PP-14,
  PP-15, PP-23.
- **Risk.** Hosted writing cost and GPU time grow with authors times seeds;
  budget the corpus runs with the existing request and token caps.
- **Size.** L.

## PP-17 The opposite author, trained on its own corpus

- **Status.** The corpus side is done for `synthetic-author`; no model has been
  trained on an opposite corpus. `corpus --opposite-of SEED` writes the
  opposite's corpus in the world of SEED; the manifest's `persona_source` says
  so and names SEED, `Persona::opposite_of` applies the checkability rule and
  refuses with the reason, and every command reads the opposite back from the
  manifest. `cross` lists a trained opposite (`opposite-of-<seed>`) against the
  author it opposes beside the existing rows. The paired test of the two
  adapters on identical situations is not built: each adapter is examined on
  the situations of its own author, so the pairing needs an examination drawn
  for both.

- **Problem.** The divergence count compares a trained adapter with the
  opposite author's oracle only; no model is trained on the opposite author.
- **Evidence.** `samples/synthetic-author/src/persona.rs:292-301`
  (`opposite()`); `src/exam_cmd.rs:253` (`cross` adds it as a rival oracle);
  `corpus` draws an author from `--seed` with `Persona::generate`
  (`src/corpus_cmd.rs:98`), and `generate(seed ^ OPPOSITE_SALT)` is not
  `opposite()`, so the opposite cannot be written today.
- **Acceptance.** `corpus --opposite-of SEED` writes the opposite's corpus
  (its manifest says so); the opposite passes the same checkability rule
  (`src/persona.rs:239-249`, `is_checkable`) or the run refuses it; on items where the two
  authors are determined and differ, each adapter takes its own author's
  decision more often than the other's, by a paired test between the two
  adapters on the same items.
- **Design.** A persona source in the manifest: drawn or opposite of a seed.
- **Depends on.** PP-07, PP-16 protocol.
- **Risk.** The opposite of a checkable author may decide too few situations;
  the refusal makes that visible.
- **Size.** M.

## PP-18 The long-context arm

- **Problem.** The base with the whole corpus in its context is not run.
- **Evidence.** `samples/synthetic-author/src/exam_cmd.rs:149` loads every arm
  with a 16384-token context; the kept corpora run to 56,107, 57,362 and
  38,916 words (`papers/persona-training/data/corpora.csv`), several times that
  context.
- **Acceptance.** An arm `context` whose system turn holds every kept article
  in order, run with a context that holds it whole (the manifest records the
  context and the prompt length of every item); an item that does not fit is
  an error, never truncated.
- **Design.** Same `run_items_with`; the context size is a parameter of the
  answerer.
- **Depends on.** brain serving the policy at that context on the available
  hardware (open prerequisite in `persona-program.md`); PP-11.
- **Risk.** Hardware: if the context does not fit, the arm waits; it is not
  replaced by a shorter one.
- **Size.** M.

## PP-19 Mechanism ablations

- **Status.** The switches are done for `synthetic-author`; no adapter is
  trained and no arm is asked of a model. `dataset --briefings` and
  `train --briefings` make and train on records with both briefings, the
  reader's alone or the fixed-wording one alone. `exam --arm` takes
  `retrieval-2`, `retrieval-12`, `retrieval-records-2`, `retrieval-records-6`,
  `retrieval-records-12` beside `retrieval` and `principles-verdicts`; the arm's
  record in the exam manifest names the source and the number of passages, and
  the corpus they come from is part of its digest. The paired tests of each
  against the main arms and the correction for multiplicity belong to the
  protocol of PP-16, and `report` lists the pairs. The protocol of `study
  freeze` names the ablation adapters and the retrieval settings, trains and asks
  them for every author (`study train`, `study exam`), and `study report` tests
  tuned against each (the ablation adapters and the records retrieval under
  Holm, the sweeps of the retrieval arm as exploratory); none is run.

- **Problem.** The strongest alternative explanation (a lookup from fact
  patterns, helped by records that restate facts in the exam's own sentences)
  is bounded but not tested; the retrieval arm is one configuration; the
  principles arm lacks the verdict rules (review record, gate "mechanism
  ablation: FAIL").
- **Evidence.** `samples/synthetic-author/src/dataset.rs:36-43` (the
  fixed-wording briefing); `src/retrieval.rs:21`, `:52-61`;
  `src/lab.rs:284-292` (the principles arm); `.agents/research/persona-training-review.md` sec. 4.
- **Acceptance.** Pre-registered in the PP-16 protocol: adapters trained
  without the fixed-wording briefings and with them only; retrieval at 2, 6
  and 12 paragraphs, over whole articles, and over the training records
  themselves (briefing and decision, the fair retrieval comparator); a
  principles arm that also states each belief's verdicts. Each reported
  against the main arms with paired tests.
- **Design.** Dataset flags; retrieval parameters in the exam manifest.
- **Depends on.** PP-16.
- **Risk.** Multiplicity grows; the protocol names the correction.
- **Size.** M.

## PP-20 Sampled decoding beside greedy

- **Status.** The code is done for `synthetic-author` and checked with a
  scripted model; no model has been asked. `Decoding::Sampled` and
  `SampledThinking` in the model adapter carry an optional seed, and
  `Answerer::provider_with` decodes under another seed without loading the
  weights again. `exam --decoding sample[:T] --draws K` asks each item K times,
  each draw with a seed derived from a run seed that is drawn, printed and
  recorded in the arm's `<arm>-sampled` record beside the temperature and the
  number of draws. `report` lists the sampled share right per arm beside
  greedy, with an item-clustered bootstrap interval and, per pair of arms, the
  interval of the mean per-item difference; its seed is drawn, printed and
  written into the report. `study exam --sampled` asks the protocol's sampled
  decoding for every arm beside the greedy one and `study report` lists its
  share right per arm and seed; the study's sampled runs are not made.

- **Problem.** Every arm is decoded greedily; the robustness of the ranking
  under sampling is unmeasured (PU-06).
- **Evidence.** `samples/synthetic-author/src/exam_cmd.rs:134`;
  `crates/model/src/answer.rs:49-59` (`Sampled` exists, with no seed).
- **Acceptance.** Each arm asked K times per item at a stated temperature,
  each draw with a recorded seed; per-item share right; arms compared with an
  item-clustered bootstrap interval; greedy reported beside it.
- **Design.** A seeded `Decoding::Sampled { temperature, seed }` in the model
  adapter (PP-05).
- **Depends on.** PP-05, PP-11.
- **Risk.** Cost scales with K.
- **Size.** S.

## PP-21 Survivorship analysis

- **Status.** The report is done for `synthetic-author`: `survivors --cache DIR`
  prints the planned and kept decisions by option, by deciding belief and by
  combination of facts and options, and the planned articles that were not kept
  by why (`content`, `writer-unavailable`, `reader-unavailable`, `budget`,
  `unattempted`). It reads the plan, `articles/` and `rejected.jsonl`. The
  per-option accuracy of each arm set beside the share kept needs the study's
  exam outcomes and is not produced by the command; `study report` joins them
  (`survivorship.csv`), over examination outcomes that are not yet made.

- **Problem.** The corpus keeps only essays whose reading agrees with the
  oracle; author 303 gave up 24 of 60 articles. What was lost is not
  reported, and infrastructure failures are mixed into it (PF-06, PF-22).
- **Evidence.** `papers/persona-training/data/corpora.csv`; 303 lost 30 of its
  61 planned assembly decisions; 2 of 101's 4 given-up articles failed only
  on writer errors.
- **Acceptance.** A `survivors` report per corpus: planned against kept
  decisions by option, by deciding belief and by combination, and given-up
  articles by kind (PP-12); in the study, the per-option accuracy of each arm
  set beside the share of that option the corpus kept.
- **Design.** Reads the plan, `articles/` and `rejected.jsonl`; no model.
- **Depends on.** PP-12.
- **Risk.** None.
- **Size.** S.

## PP-22 Enough items to settle tuned against retrieval

- **Status.** Done in the protocol; the settling examination is not run. `study
  freeze` sizes `per_cell` by an exact computation (`src/study/power.rs`) from
  the observed discordance and win share, records the derivation in the
  protocol, and a test draws an examination of that size for a drawn author. For
  37 of 160 items discordant and 23 of 37 for the adapter, 80% power at
  one-sided 5% needs 120 items per kind, 480 per author (about 111 discordant
  pairs): the estimate below by normal approximation is a little low because the
  exact test is conservative at these counts and the size must hold at the next
  three sizes too.
- **Problem.** For author 101 the adapter and retrieval split 23 to 14 on the
  items they differ on (p = 0.094): the comparison is underpowered.
- **Evidence.** `papers/persona-training/data/paired.csv`.
- **Acceptance.** The PP-16 protocol sizes the exam for this comparison: at
  the observed discordance (37 of 160) and win share (23 of 37), 80% power at
  one-sided 5% needs about 106 discordant pairs, about 460 items per author;
  pooled over training seeds as the protocol states.
- **Design.** Raise `per_cell` in the protocol; the power computation is the
  existing powered-exam simulation.
- **Depends on.** PP-13, PP-16.
- **Risk.** Exam size raises every arm's cost.
- **Size.** S.

## PP-23 A reader other than the writer

- **Status.** The protocol check and the audit are done for
  `synthetic-author`; no corpus has been written with three families. `corpus`
  refuses a reader of the writer's family and `mine` a miner of the family of
  either, unless `--allow-same-family` is passed, which `manifest.json` or
  `mine.json` records. The gate reader's readings are kept in `readings.jsonl`;
  `audit` reports the gate reader's and the miner's agreement with the beliefs
  and their agreement with each other per case. The choice of the three models
  belongs to the PP-16 protocol.

- **Problem.** One hosted model writes the essays, reads them for the gate and
  extracts the decisions; a consistent error of that model passes every gate.
- **Evidence.** `papers/persona-training/sections/07-discussion.tex` (scope of
  validity); `samples/synthetic-author/src/corpus_cmd.rs:47-53` (`--writer`, `--reader`) and `src/main.rs:91`
  (`mine --miner`) take any reference.
- **Acceptance.** Writer, gate reader and miner are three different model
  families in the protocol; the audit reports agreement between gate reader
  and miner per case and the oracle agreement of each.
- **Depends on.** Nothing; the reader is fixed in the PP-16 protocol before
  any corpus is written.
- **Risk.** A weaker reader rejects more essays (PP-21 measures it).
- **Size.** M.

## PP-24 Decisions and tool calls in `learn`

- **Problem.** The sentence `splinter "Learn to think and act like X from
  ./dir"` cannot teach acting: `learn`'s kind catalogue has no decision,
  tool-decision, principle, polarity, rebuttal, anachronism, attribution or
  interlocutor kind (the persona task menu), and the decision mining and
  tool-call records exist only inside the synthetic-author sample.
- **Evidence.** `crates/knowledge/src/tasks/kind.rs:424-643` (the built-in
  kinds: recall, arithmetic, format, explain, counterexample, transform,
  classify, retrieve, multi-turn, combine, advise, converse, predict,
  construct, debug); `samples/synthetic-author/src/mine.rs`,
  `src/records.rs`, `src/tools.rs`; `crates/pipelines/src/learn/stages.rs`
  (`system_prompt`, `persona`: a persona run adds the voice share and the
  persona prompt, no decisions).
- **Acceptance.** A `decision` kind (a documented situation with the decision
  withheld, its options and its quoted reasons, verified by quotation) and a
  `tool-decision` kind (the same decision through a caller-defined tool,
  graded by the captured argument) in the knowledge catalogue; the miner
  (situation, options, choice, quoted reasons) in knowledge behind the typed
  call the agent crate owns; the tool-call record projection in `data`,
  ending with the tool's answer (PP-14); a scripted integration spec of
  `learn` on a fixture corpus that produces decision records and an exam of
  unseen tools. The synthetic-author sample then calls these instead of its
  own copies.
- **Design.** Move, do not copy: the sample's verified pieces become the
  crates' (rule in `persona-platform.md`, "Generalizing sample code into the
  SDK").
- **Depends on.** PP-10, PP-14, PP-15, PP-36, PP-37, PP-38 (the principle
  verification, the scenario planner with its giveaway gate, and the judge
  calibration and document bundling the new kinds and the miner use).
- **Risk.** Real documents state options implicitly; the miner's `open`
  field may be empty, and a decision without alternatives is not a
  `tool-decision` task.
- **Size.** L.

## PP-25 Documented-decision exams for Jefferson and Adams

- **Problem.** The real-author runs must be scored on held-out documented
  decisions; none is frozen or implemented.
- **Evidence.** `.agents/research/persona-decision-scenarios-draft.md`: 7
  Jefferson and 6 Adams scenarios, quotes checked verbatim, EXCLUDE ranges
  listed, most marked base-guessable. `samples/jefferson/src/exam.rs` and
  `samples/adams/src/main.rs` exams grade recall, attribution and
  reconstruction, not a decision through a tool. Neither `materials`
  command removes the draft's EXCLUDE ranges
  (`samples/jefferson/src/main.rs:400` and `samples/adams/src/main.rs:325`,
  `materials_command`; `samples/jefferson/src/corpus.rs:375` clears only the letters of
  its recall exam).
- **Acceptance.** Per author at least 30 documented decisions (13 cannot
  support a paired test; seven of seven wins is p = 0.0078 only when every
  item is discordant), each with a neutral briefing, options, the documented
  choice and verbatim reasons; screened against the prompted base and the
  named base, with base-guessable ones reported apart; frozen and hashed;
  `materials` removes every EXCLUDE range and a test proves no quoted reason
  of an exam decision is left in the materials by Splinter's overlap rule; an
  `exam --decisions` command asks through a `decide` tool and through an
  unseen tool and grades the captured argument. Beside it, a programmatic
  puzzle exam over the person's facts, names and reasoning, each item built
  by code with a solver that proves it solvable from the materials; retention
  and anchor suites; the leak check. A sample of every exam is shown to the
  user before it is frozen.
- **Design.** In each sample's apparatus (the independent check); the tool
  turn is `agent::tool_turn`.
- **Depends on.** PP-24 for the tool shape; PP-34 (shared-run check) for the
  leak test.
- **Risk.** Documented decisions of famous people are in pretraining; the
  screen against the prompted base is what makes a win mean something.
- **Size.** L.

## PP-26 Real-author runs through the one sentence

- **Problem.** Jefferson, Adams, their opposites and one non-historical
  profession have not been run through the identical command.
- **Evidence.** `samples/jefferson/README.md:39` and
  `samples/adams/README.md:45` run `learn` from the sentence without
  `--adapter` and examine recall; the router maps "as the adapter NAME" to
  `--adapter` (`crates/pipelines/src/router.rs:74-77`, `:229-233`).
- **Acceptance.** For each person: `splinter "Learn to think and act like X
  from ./materials as the adapter x"`, its learn report, the PP-25 exam on the
  published adapter against the base, named, principles and retrieval arms,
  retention and anchor suites unchanged within the gate's rule; opposing
  materials where the person had documented opponents, else generated
  opposed claims, measured as in PP-17; one non-historical profession with a
  body of materials and its own documented decisions.
- **Depends on.** PP-24, PP-25, PP-28 (publish safety), PP-08 (seeds).
- **Risk.** Hardware time; the corpus size is set by the saturation curve
  (decision in `persona-program.md`), not by a budget.
- **Size.** L.

## PP-27 Paper: REVIEW-READY to SUBMISSION-READY

- **Problem.** The review record leaves the paper REVIEW-READY, and new
  findings bear on its method.
- **Evidence.** `.agents/research/persona-training-review.md` sec. 6 and 7:
  the human author has not confirmed the language-model use paragraph,
  authorship and affiliation, and that the retrieval arm was added after the
  first three arms (C14); no venue's rules were checked; mechanism ablations
  and further seeds were not run; cost figures were carried over (C16, PU-05).
  New: PF-01 (records contradict the oracle on their own prompt), PF-05 (the
  reported chance level is below a constant policy), PF-04 (all three runs
  share one default seed), PF-06 (infrastructure failures among the given-up
  articles), PF-15 (the adapters echo a brief phrase as their reason).
- **Acceptance.** Immediately, as AGENTS.md requires of a found defect in a
  stated method: the paper's limitations section says, in a sentence each,
  the defects that bear on a claim it makes (PF-01, PF-04, PF-05 and PF-15),
  and no more: no catalogue of defects, no log of failures. PF-05's
  first-option baseline also enters the results table, since it is the
  reference a result is read against. `build_data.py` checks the new
  numbers. For submission: the human confirmations recorded in the
  review file; a venue chosen and its rules checked; the replication (PP-16)
  and ablations (PP-19) reported, superseding the single-run results; cost
  re-derived (PU-05); `make` clean.
- **Items a human author must confirm.** The "Use of language models"
  paragraph; that the retrieval arm was added after the first three arms
  had been run; authorship and affiliation as printed; the venue.
- **Depends on.** PP-16, PP-19 for submission; nothing for the disclosure.
- **Risk.** The replication may not reproduce the margins; the paper then
  reports what holds.
- **Size.** M.
