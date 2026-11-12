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

Not done: selection is lexical (no embedding signal, no model-judged relevance); novelty is accuracy,
not loss (brain log-probabilities of the section's items would sharpen it); facts are two sentence
patterns until a model extractor is admitted by code; the participant that trains on the chosen sections
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
