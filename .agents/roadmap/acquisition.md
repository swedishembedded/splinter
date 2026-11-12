# Autonomous acquisition, intervention selection and generations - roadmap

Scope: autonomous acquisition of a capability from documentation (`learn "<objective>" --source`), the choice of an intervention by measured gain and cost (`improve`), and gains kept across generations (`generations`).

Gate C (a capability the policy lacks is acquired from documentation with no manual edits) and
Gate D (gains that survive three generations). What is built, what each measures, and what a
GPU run must still show. Update this file in the same commit that finishes or changes an item.

## Built, verified on scripted agents (CPU, no weights)

- **Teacher for repository tasks** (`practice --teacher`): see `repository-practice.md`, item 11.
- **Acquisition** (`splinter learn "<objective>" --source <url|path>`, `splinter-pipelines::acquire`):
  sources (URLs fetched once, HTML read as Markdown) -> skill graph (planner proposal admitted by
  code, else read by rule; versioned) -> probes per skill, dev and final halves, checked to
  exercise the skill and to repeat no other task -> baseline competence on the dev probes ->
  rounds on the skill of greatest expected benefit (prerequisites first) -> accepted only if the
  skill improved and nothing measured before fell past the bound -> assessment on the final
  probes against the policy the run began with. No second scheduler, task store or trainer: it
  composes the task generator, solve, verify, teach, frontier, dataset, train and the gate.
- **Lua 5.1 pin**: the `lua` runtime is Lua 5.1 and refuses another version
  (`splinter-sandbox`); `make deps/lua LUA_DIR=<dir>` builds the interpreter from its
  checksummed release tarball; `--runtime-dir <dir>/bin` puts it first for the runtime and for
  the workspace commands; `scripts/tasks/families.py check` refuses a `lua` that is not 5.1.
- **Intervention selection** (`splinter improve`, `splinter-pipelines::improve`,
  `splinter-eval::intervention`): per intervention, the paired gain per family against the
  incumbent on the same tasks, its lower confidence bound, its one-time and per-use cost, net of
  what a success is worth; the arm with the greatest net value above zero, else no change;
  unavailable mechanisms reported as not measured; the choices against every fixed strategy.
  Controlled test: three synthetic failure clusters (bad harness, missing knowledge, search beats
  training) and one where nothing helps.
- **Generations on the coding path** (`splinter generation run|matrix`,
  `splinter-pipelines::generations`): the protected panel (every earlier exam), the champion
  measured on all of it before training, replay policies (`none`, `uniform`, `balanced`,
  failure-targeted, rebuild from the base), candidate and champion measured at equal settings,
  rejection on a family retention limit, a mean regression limit and no gain on the new exam, the
  gate as the only thing that moves an alias, a ledger of everything each decision rested on,
  and the full retention matrix (the base as generation 0, every cell measured) with ACC, BWT,
  forgetting, forward transfer and learning gain.
- **Qualification harness** (`make qualify/list|test|run`, `scripts/qualify/`): the roadmap's
  matrix and the six mandatory ablations as one recorded command; the plan is registered (its
  digest written before the first check); BLOCKED is never a pass; an unmeasured assertion
  fails; an ablation states its measured effect.

## Goal-directed selection of raw material (built 2026-11-12)

Scope: a participant with a goal (role, objective, probes) given a raw corpus that is mostly
irrelevant, outdated, contradicted or misleading finds the sections worth learning, the way a person
with a goal finds what to read. `splinter_knowledge::goal` (pure code, no model) and
`splinter_pipelines::goal` (the loop, the comparison, the counterpart persona);
`samples/goal-selection` runs it. `acquire` chooses a skill to train from documentation it is
told to use; this chooses which documentation sections to use at all, and composes with it.

- **Gap analysis** (`goal::gaps`): probe the participant, cluster the failed probes into gaps by
  corpus-rarity-weighted cosine of their words (a named `skill` is the cluster; numbers and words only
  one probe uses never link). A gap stays open after its probes pass, until no new material relevant
  to it is left: a probe samples a capability, it does not bound it.
- **Selection** (`goal::select`): one decision with a reason per section, in order: relevance to the
  open gaps (rarity-squared coverage of the gap's words, weighted by the share of its probes that use
  them) with the goal's objective naming the domain; conflicts over fact keys across the whole corpus
  (authority, then date, then version: Contradicted / Superseded / Contested); novelty (the
  participant is asked the question each section fact answers: Known); duplicates; a word budget by
  value per word. Facts are read by a `FactExtractor` (pattern default; a model can replace it).
- **Loop** (`pipelines::goal::learn_goal`): probe, gaps, select, learn, re-probe; stops on goal met,
  nothing worth learning, budget or rounds. The participant is a trait (`Participant`); `InContext` learns
  by remembering sections and answers by showing a `Reader` the top BM25 sections (learning without weights,
  what a machine with no free card can do); a participant that trains an adapter on the compiled material
  implements the same trait. Counts every question put to it.
- **Counterpart roles** (`goal::behaviour`): the same loop with a counterpart goal ("behave as a realistic
  hesitant tenant"), material that describes behaviour, and the artifact is a persona and a `SeatSpec`
  (parsed by sven's own type); fidelity is the share of held-out situations of that type a reader acting from
  the persona alone gets right.
- **Labelled corpora** (`goal::synthetic`): relevant / known / outdated / misleading / irrelevant sections,
  labels apart from documents, two forms (equipment properties, client behaviour), dev probes and held-out
  questions on disjoint facts, document order shuffled so no order tells which is current.
- **Comparison** (`goal::compare`): selective vs nothing vs everything vs the same words at random vs
  the relevant-only oracle, and the loop with the conflict check, the novelty check and the objective each off.

Scripted reader (best-ranked passage wins, no date/authority reasoning), 10 seeds each, deterministic:

| arm | property held-out | behaviour held-out (one client type, persona fidelity) |
|---|---|---|
| none | 0.30 | 0.24 |
| selective | 1.00 (precision 1.00, recall 1.00, 1357 words) | 1.00 (388 words) |
| random, same words | 0.32 | 0.28 |
| all (8561 / 5277 words) | 0.43 | 0.43 |
| all, each section shown with its authority, date and version | 1.00 (8561 words) | 1.00 (5277 words) |
| oracle-relevant | 1.00 | 1.00 |
| selective without conflict check | 0.39 (harmful share 0.61) | 0.42 |
| selective without novelty check | 1.00 (precision 0.70, 1939 words) | 1.00 (517 words) |
| selective without the objective | 1.00 | 0.63 |

Real model (Qwen3.8-27B-FP8 on a brain server, one request at a time, reader answers from the 3 best
BM25 sections it is shown, reasoning closed, `goal-selection --reader model`, evidence in
`~/project/shared/goal-selection/` outside the repository; 3 seeds each, 2 goal topics x 4 units for properties,
one client type x 6 units for behaviour, 12-20 held-out questions per corpus, no confidence interval claimed):

| arm | property held-out (3 corpora) | behaviour held-out = persona fidelity (3 corpora) |
|---|---|---|
| none (closed book, invented corpus) | 0.00 | 0.00 |
| selective | 1.00 on all 3 (about 515 words, 96 questions put to the model to decide) | 1.00 on all 3 (about 385 words, 48 questions) |
| random, same words | 0.15 (0.30, 0.00, 0.15) | 0.25 (0.25, 0.33, 0.17) |
| all | 0.00 (the model answers unknown when the passages disagree) | 0.00 |
| all, with provenance headers | 1.00 on all 3 (about 2590 words) | 0.92 (0.83, 1.00, 0.92; about 2560 words) |
| oracle-relevant | 1.00 | 1.00 |
| selective without conflict check (1 corpus) | 0.15, harmful share 0.75 | |
| selective without novelty check (1 corpus) | 1.00 | |
| selective without the objective (1 corpus) | 1.00, precision 0.67, 780 words | |

Read plainly: selection beats nothing, everything-as-is and random on questions it never saw, and equals the
oracle. A reader that is shown each section's provenance and told how to weigh it matches selection on
accuracy and pays 5x the words (and prompt tokens) for it, so the gain of selection over a metadata-aware
reader is cost, not accuracy, and it grows with corpus size (retrieval depth was 3 in every arm); for training
the participant on the material, which cannot weigh headers at answer time, the accuracy gap is the one that
stands. The conflict check decides accuracy, the novelty check decides words and precision (scripted corpora
where the participant knows part of the topic; a model knows none of an invented corpus, so its novelty check
had nothing to remove), the objective decides precision where gap words alone are ambiguous.

The world is saturated for lexical selection (selective equals the oracle), so these numbers are a
regression spec and an ablation of what each check contributes (conflicts decide accuracy, novelty
decides words and precision, the objective decides recall where gap words alone are ambiguous), not a
claim about difficulty. Defects found on the way, each now a test: the gap vocabulary was
dominated by attribute and entity-number words (fixed by corpus rarity and dropping numbers);
relevance summed over gaps capped a section at its gap's share (floor now on the best single gap, rank on
the weighted sum); stopping when dev probes passed learned only probed entities (gaps stay open); a
corpus ordered current-first let a first-wins reader look good (documents shuffled).

Interface with the typed knowledge objects: documents and their headers are `objects::Document` and
`Provenance` (one definition, the same authority, date, version order). Not yet reused: an admitted pack's
resolutions and `superseded_ids` could give the Superseded verdict directly, and its fact and behaviour
objects could be the `FactExtractor`, instead of the two sentence patterns.

### Second round (2026-11-12): knowledge objects as the source, a hard world, meaning, weights, the scrubber

- **One implementation.** Facts and the rulings between disagreeing statements are the material compiler's
  (`goal::source::PackSource` over an admitted `objects::Pack`): my own conflict code is gone. Fixed there: a
  tie among documents that agree is no conflict. A section no object can be made of is `Empty`. Limit found:
  admission requires a fact key's words to be in its quote, so two statements that paraphrase each other are
  not seen to disagree; semantic key normalisation belongs in the compiler.
- **Hard world** (`goal::hard`, `--hard`): paraphrased current documents, header-less near-copies, partial staleness (the
  older manual is the only, right source for half the units), decoys. Not generated: multi-hop questions
  (answer in two documents) and gap expansion from learned material; both are open.
- **Meaning** (`goal::semantic`, `Signal`, `InContext::retrieving`): a section stands out for a gap by its cosine to the
  nearest failed probe, read against the corpus (0 at the median section, 1 at the nearest) with its own floor
  (0.2, tuned on one served-embedder corpus, checked on two others); the reader's memory is searched by BM25 and
  the embedder fused by reciprocal rank. Scripted runs use a synonym-table embedder (knows exactly the synonyms
  it is given); real runs use Qwen3-Embedding-0.6B served by brain.
- **Intervals** (`goal::report`): means over corpora with 95% percentile bootstrap, differences paired by corpus.

Scripted reader, 12 corpora per cell, held-out pass (95% interval); selective / selective with provenance headers /
everything with provenance headers / oracle; precision and recall of the chosen sections against the labels:

| cell | selective | sel + provenance | all + provenance | oracle | precision / recall of selective |
|---|---|---|---|---|---|
| property, hard, lexical | 0.73 [0.67,0.78] | 0.98 [0.97,1.00] | 0.87 [0.83,0.90] | 1.00 | 0.68 / 0.98 |
| property, hard, semantic | 0.70 [0.63,0.76] | 0.79 [0.73,0.84] | 0.90 [0.88,0.93] | 1.00 | 0.42 / 0.72 |
| property, hard, fused | 0.74 [0.68,0.79] | 0.94 [0.90,0.98] | 0.89 [0.86,0.92] | 1.00 | 0.55 / 0.93 |
| behaviour (one client type), hard, lexical | 0.76 [0.66,0.87] | 0.76 [0.66,0.87] | 0.75 [0.65,0.84] | 0.89 | 0.70 / 0.75 |
| behaviour, hard, semantic | 0.89 [0.83,0.95] | 1.00 [1.00,1.00] | 0.91 [0.83,0.98] | 1.00 | 0.73 / 1.00 |
| behaviour, hard, fused | 0.79 [0.70,0.88] | 0.99 [0.98,1.00] | 0.84 [0.79,0.90] | 1.00 | 0.73 / 1.00 |

Where each wins: words find the sections of an invented topic name (property recall 0.98 lexical, 0.72 semantic: an
embedder cannot know that `albalt` and `albalt pump 3` belong together); meaning finds the paraphrased ones (behaviour
recall 0.75 lexical, 1.00 semantic and fused); fused keeps both recalls and pays precision (decoy and stale sections the
embedder also finds).

Real Qwen3.8-27B-FP8 reader, Qwen3-Embedding-0.6B (served, 8801/8802), 5 corpora per cell, concurrency 1, evidence in
`~/project/shared/goal-selection/hard/` (seeds 3000-3004; a first set on seed 2000 with an uncalibrated floor is in
`hard-v1-seed2000-uncalibrated`): held-out pass (95% interval)

| cell | none | selective | selective + provenance | all | all + provenance | random, same words | oracle |
|---|---|---|---|---|---|---|---|
| property, lexical | 0.00 | 0.54 [0.49,0.60] | 0.70 [0.64,0.77] | 0.12 | 0.68 [0.62,0.76] | 0.20 | 0.93 |
| property, fused | 0.00 | 0.52 [0.47,0.55] | 0.67 [0.61,0.72] | 0.15 | 0.64 [0.53,0.73] | 0.20 | 0.95 |
| behaviour, lexical | 0.00 | 0.80 [0.62,0.98] | 0.80 [0.62,0.98] | 0.21 | 0.49 [0.42,0.58] | 0.16 | 0.80 |
| behaviour, fused | 0.00 | 0.78 [0.60,0.96] | 0.80 [0.62,0.98] | 0.20 | 0.56 [0.40,0.71] | 0.18 | 0.80 |

Read plainly: against reading everything as it comes, selection wins by a wide margin; against reading everything
with provenance, selection with provenance ties (property 0.70 vs 0.68, intervals overlap) at a fifth of the words and
wins in the behaviour form (0.80 vs 0.49-0.56, paired interval excludes zero). The served embedder raised behaviour
recall (0.80 to 0.95) without moving held-out accuracy at n = 5. On property the loop is bound by its budget (four
rounds of 300 words): it spends part on stale sections whose paraphrased current counterpart no key links, which is
why its recall is 0.71 against 0.98 scripted. Five corpora per cell: intervals are wide and no claim is made beyond them.

**Weights, toy model** (`make test/methods`, `tests/methods/selection.rs`, CPU, 161 s): brain's toy checkpoint trained
(supervised, 500 steps) on cipher tables chosen by the loop, held-out exact match on prompts in no table: none 0.00,
selective 1.00 (18 of 25 current sections; the 7 left were already answered, so `Known`), random same words 0.06,
everything 0.00 (current, outdated, forum and another cipher's tables conflict), oracle-relevant 0.99. One corpus, one
seed of draws; falsifiable bars asserted. 27B adapter comparison queued, not started: see the Log.

**Live data**: `goal::intake` admits a live document only with consent, the current detectors' scrub marker, and no residue
when the detectors run again; a marker alone does not vouch for the text. Not yet wired into a CLI command.

### Third round (2026-11-12): aligned keys, early rulings, multi-hop, live data behind a command

- **Semantic conflict detection** (`goal::align`, `objects::resolve_aligned`): an embedder proposes which attribute
  phrases and entity names are one (cosine >= 0.6, mutual best by a margin of 0.05, no document uses both phrases, no
  digit in an entity name); the rulings stay the documents' headers and the stated values, never the embedder's. The
  compiler's `resolve` is reused unchanged (only the keys it groups by are rewritten). Written down as findings 123
  and 124 in `.agents/knowledge/`.
- **Budget waste.** Rulings were already decided before novelty and before the budget; the waste was rulings that
  could not be reached across a paraphrase (finding 123). With keys aligned, real 27B reader, served
  Qwen3-Embedding-0.6B, property form, fresh seeds 6000-6004 (5 corpora, concurrency 1), held-out pass (95%
  interval), precision / recall of the chosen sections:

| property, hard, fused | selective | precision / recall | selective + provenance | all + provenance | oracle |
|---|---|---|---|---|---|
| keys aligned | 0.97 [0.95,1.00] | 0.99 / 0.99 | 0.97 [0.93,1.00] | 0.65 [0.58,0.72] | 0.98 |
| keys as written | 0.70 [0.54,0.89] | 0.84 / 0.84 | 0.82 [0.70,0.94] | 0.65 [0.59,0.72] | 0.99 |

  The earlier recall of 0.71 (seeds 3000-3004) is now 0.99 and the budget no longer goes to stale sections.
- **Alignment is not safe everywhere.** Client-behaviour form, same protocol: aligned selective 0.40 [0.33,0.47]
  (precision 0.46, recall 0.60), unaligned 0.52 [0.43,0.61] (precision 0.39, recall 0.78), oracle 0.52. Phrase
  embeddings cannot separate a paraphrase from a neighbouring situation, the structural check (no document uses both)
  refuses nothing when situations and types never co-occur, and a false merge turns two facts into a ruled conflict
  (finding 124). A guard that drops unsettled flags across an alignment was tried and was worse (0.28). Open: a
  structural check for keys that never co-occur, or leaving those keys unaligned and relying on provenance headers.
- **Multi-hop and gap expansion** (`Form::Link`, `Hardness::hops`, `gaps::dependent_gaps`, `InContext::looking_again`):
  a register says which unit is installed at a site; a question names the site; the unit's facts are in other
  documents. A still-failing probe whose rare words occur in a learned section opens a dependent gap with the rare words
  of those sentences (the unit named), so the next round asks for that unit's facts. Scripted, 8 corpora: expansion on
  raises recall and held-out by more than 0.2 over expansion off (test
  `a_learned_link_opens_the_gap_for_the_facts_it_names`). Real 27B reader, served embedder, 5 corpora (seeds
  6500-6504, 2 topics x 12 units, six rounds of 900 words): expansion off: recall 0.60 [0.51,0.70], precision 0.73,
  selective + provenance 0.20 [0.05,0.37], 1568 words; expansion on: recall 0.99 [0.97,1.00], precision 0.49,
  selective + provenance 0.88 (all + provenance 0.88, oracle 1.00), 5229 words (oracle 2049). Expansion opens the
  gap, at the price of reading everything the dependent gaps match (precision falls); without provenance headers the
  same chosen material scores 0.16 against 0.88 with them, cause not found (open, not a finding).
- **Live data** is behind `splinter goal admit --open FILE.. --live FILE.. --consent --scrub --out DIR`: a live file is
  admitted only with consent and after the current scrub, and a refusal exits with code 2.
- **27B adapter comparison** is queued and not started (the training card belongs to the RL lane); `queue-27b.sh`
  is verified up to model loading (`splinter learn` reaches `could not be loaded` with a nonexistent models directory).

Not done: no model-judged relevance; novelty is accuracy,
not loss (brain log-probabilities of the section's items would sharpen it); facts come from two sentence
patterns through the compiler's admission (a model proposer is untested here); the participant that trains on the chosen sections
(waits for a card, and the material compiler's task families); sections that are relevant but share
no word with the goal or probes are missed; live-user behaviour data goes through the scrubber first.

## What GPU runs must show (hand-off to the run owner)

1. **Teacher**: `practice --teacher reference` on a repository whose policy solves little
   (the 8B base on the sqlglot family): the number of never-solved tasks that become verified
   demonstrations, the training records they add, and CE4 with and without them. Report n and
   the paired comparison; demonstrations are oracle-localised.
2. **Acquisition**: the Lua 5.1 manual (`https://www.lua.org/manual/5.1/manual.html`) with the
   pinned interpreter (`--runtime-dir`), kinds `construct-lua,predict-lua,debug-lua`, the 8B policy
   and the 27B as teacher: skills admitted, baseline, rounds, final assessment against the
   baseline; a skill it did not improve is reported.
3. **Intervention selection** on real failure clusters: one experiment file per cluster, the
   arms measured by solving (samples, protocol, a candidate trained with teacher demonstrations),
   `splinter improve --compare-fixed`.
4. **Three generations** on the E3 plan with `generation run`, then `generation matrix`; the
   replay arms (`--replay none|uniform|balanced`, `--rebuild`) on the same plan.
5. The qualification run: `make qualify/run OUT=<dir> VARS="--var ..."` once the registered
   results exist.

## Not done

- Task kinds for Lua exist as data only after this change (`construct-lua`, `predict-lua`,
  `debug-lua`); the generator's admission of Lua programs has been run only against scripted
  replies.
- `improve` measures search, retrieval and a decision model as unavailable until the mechanisms
  land (search: sven's facility); a bandit over interventions is later work.
- The retention matrix is measured in full each time it is asked for (an attempt at every
  protected task per generation); a cache keyed by model and task is not built.
- A generation's candidate is measured on the panel before the gate, and the gate then measures
  the new exam again: the attempts are not shared between them.
