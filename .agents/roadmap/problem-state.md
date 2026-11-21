<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Problem state: which experiment to run next, and what each verdict taught

Scope: the campaign's problem state (`splinter campaign problems`, `splinter campaign next`,
`splinter_eval::campaign::problems`, `splinter_pipelines::campaign::problems`): diagnosed capability gaps with
evidence, the candidate interventions against each with a belief, best-first choice of the next single-variable
hypothesis, belief updates from the promotion gate's and `experiences compare-arms`' verdicts, the single-variable and
engineering-firewall checks on campaign configurations, and how a coordinator inspects and overrides all of it.

Design of 2026-11-21 (lane PRO2), adapted from "Pro2 Agent: Scientific Problem Probing for Autonomous AI R&D"
(arXiv 2611.25357). Status and open items are at the end.

## Why

Experiments in this repository are chosen by reading the ledgers. Items 19-25 of `experience-to-competence.md` ran
about twenty arms in two days; the reasons each was chosen, what it was expected to fix and what its verdict changed
live in prose spread over a 1,300-line ledger. The paper's ablation found that keeping a *problem state* (what limits
performance, with evidence, and the candidate fixes with a belief each) instead of a codebase state was the largest
single gain of its agent, and that grounding each entry in evidence and revising it after every experiment added
more. Splinter already has every other piece: diagnosis, interventions, the knowledge base, campaigns with one
promotion gate and rollback, and the validated arm statistics. What it lacks is the record that ties them together and
a rule for what to try next.

## The paper, and the mapping

| Paper | Here |
|---|---|
| problem `P_i` with evidence `E^p_i` | a diagnosed capability gap or failure class, with evidence links (knowledge entries, ledger items, results, failure diagnoses, commits) |
| candidate solution `C_ij`, belief `b_ij`, evidence | an intervention or training recipe (teacher distillation, concise traces, STaR, expert iteration, knowledge mount, RL round, curation option) with a belief in `[0, 1]` and evidence links |
| hypothesis: a single-variable change from the current best, checked by a reviewer | a campaign arm whose configuration differs from the champion's in exactly one declared variable, checked by code (`campaign problems check`) |
| supported if `dh >= delta_target`: new best, `b <- min(1, b + db_up)` | the promotion gate promotes, or the arm's interval clears the target (below): `b <- min(1, b + db_up)`, the candidate is adopted |
| falsified or crashed: roll back, `b <- max(0, b - db_down)` | falsified: the interval rules the target out; the champion stays (the gate keeps it; a promoted arm is undone by `campaign rollback`); `b <- max(0, b - db_down)`. A crash is *invalid*, not falsified (below) |
| `Pri = b_km / (N(P_k) + 1)` | the same, `N` the verdicts recorded against the problem plus the attempts recorded before the state existed |
| pending hypotheses re-scored with distilled insights | a coordinator's belief revision, recorded with its reason as an event (`campaign problems revise`) |
| engineering phase, firewalled from model, algorithm and data, reward `dh - gamma ln(T_exec)` | throughput and serving changes: an engineering candidate's arm may differ only in configuration paths classified as engineering; the firewall check refuses anything else |
| experience bank of error signatures, root causes, fixes | the existing knowledge base (`.agents/knowledge`) and the failure diagnoses (`splinter-eval::diagnosis`); evidence links point at them, nothing is copied |

### Where our setting differs from the paper's

- **Measurements are expensive and noisy.** The paper's `dh` is one number per run. An arm here costs GPU-hours and its
  effect comes with an interval that is often wider than the effect (item 21 S1b: +0.029 [-0.050, +0.105]). A raw
  delta would move beliefs on noise. Beliefs move on the *verdict* of the validated statistics instead: supported,
  falsified, inconclusive or invalid. **Inconclusive never counts as falsified**: it leaves the belief where it was
  and only counts an attempt, so the priority halves without the idea being called wrong.
- **The baseline moves.** Knowledge 130 and 196 measured a base exam moving by as much as the effects under study
  between time slots and loads. A verdict therefore always comes from a paired, same-slot comparison (the gate's or
  `compare-arms`'), never from two numbers measured apart.
- **Crashes and load-confounded runs are not evidence.** The paper lowers the belief of a crashed hypothesis. Here a
  crash or a measurement the time limit decided (item 25 E1: 71-96% of trials timed out under load) says nothing
  about the intervention, so it is recorded as *invalid*: an attempt is counted (so a hypothesis that keeps crashing
  sinks in priority) and the belief does not move.
- **"Single variable" includes fresh data.** An arm is a campaign generation: it rolls out fresh, trains a fresh
  candidate and is gated against the champion. Its configuration differs in one variable; its rollouts are a new
  draw of the same distribution under the campaign's seed. The gate's paired, family-clustered interval is what makes
  that draw part of the noise rather than a second variable.
- **Repeated gates spend error.** Knowledge 047 and 154: ten gates at 5% release a null candidate four times in ten, and
  the sealed exam found 2 of 12 greedy dev promotions were not improvements. A support is therefore weaker evidence
  than its interval suggests, which is why `db_up < db_down` below.
- **The engineering reward is not implemented.** The campaign records GPU-hours per training stage but no execution
  time per measured arm, so `dh - gamma ln(T_exec)` cannot be computed from the ledger yet. Only the firewall, the
  part that protects the science, is built (open item PS-3).

## Schema (version `splinter.problem-state/1`)

A problem-state *document* (TOML or JSON) seeds or extends the state. The state itself is never edited: it is the
fold of an append-only list of events. Field names are the serialised names.

- **parameters** (optional; the defaults below when absent): `belief_up`, `belief_down`, `alpha`,
  `targets` (metric name to `delta_target`), `engineering` (configuration paths an engineering change may touch).
- **problems[]**: `id` (letters, digits, `-`, `_`, `.`), `statement`, `metric` (must have a target),
  `status` (`open` or `resolved`), `attempts` (experiments run against it before the state existed, each named in the
  evidence), `evidence[]`, `candidates[]`.
- **candidates[]**: `id` (unique across problems), `intervention` (a short kind: `teacher-distillation`,
  `concise-traces`, `star`, `expert-iteration`, `knowledge-mount`, `rl-round`, `curation`, `engineering`, ...),
  `statement`, `variable` (the one configuration path its arm changes, e.g. `practice.profile`, `train.learning_rate`),
  `value` (optional: the value at `variable`, when the campaign file can express it; `campaign next` then writes the
  arm), `phase` (`hypothesis` or `engineering`), `belief` in `[0, 1]`, `prior` (why the starting belief is what it
  is), `status` (`pending` or `adopted`), `evidence[]`, `history[]` (derived, never written in a document).
- **evidence[]**: `ref` as `<kind>:<id>` with kind one of `knowledge`, `ledger`, `result`, `diagnosis`, `commit`,
  `experience`, `dataset`, `campaign`, `paper`; and a `note`. A link names where the evidence lives; it never copies
  it.
- **history[]** (one per verdict): `experiment`, `verdict`, `belief_before`, `belief_after`, `effect` (difference,
  interval, p, when measured), `evidence[]`, `reason`.

**Events** (`splinter.problem-event/1`, one per ledger commit):

| Event | What it does |
|---|---|
| `add` | a document: the first seeds the state; later ones add problems and candidates (an existing problem id may be repeated, with the same statement and metric, to add candidates to it; an existing candidate id is refused) and may replace the parameters (from that event on) |
| `verdict` | one experiment's verdict on one pending candidate (of an open or a resolved problem): belief updated, the problem's `attempts` + 1, history appended, a supported candidate becomes `adopted`; a second verdict for the same experiment and candidate is refused |
| `belief` | a coordinator sets a candidate's belief, with a reason (re-scoring with an insight, or an override) |
| `status` | a coordinator resolves or reopens a problem, with a reason |

**Where it lives and how it is hashed.** `<state>/campaigns/<name>/problems/ledger.jsonl`: the stage ledger every
campaign stage uses (`splinter_store::stage_ledger`), in its own directory beside the campaign's, so it can be read and
revised while `campaign run` holds the campaign's own ledger. Each event is one unit named `event/<key>`, its key the
digest of the event's canonical JSON. Recording an event already recorded finds its commit and records nothing
(idempotent delivery); recording a conflicting one (another verdict for the same experiment) is refused before
anything is written. The state's identity is the blake3 digest of its canonical JSON (`state` in `show`), so two
replays of the same history agree on one digest.

## Rules and parameters

Every parameter is in the document's `parameters`; the defaults and why:

| Parameter | Default | Why |
|---|---|---|
| prior belief | stated per candidate with a `prior` reason; 0.5 when nothing is known | the mean of a uniform Beta(1, 1); a candidate grounded in a result starts elsewhere only with that result linked |
| `belief_up` | 0.15 | a support is weaker than it looks under repeated gating (knowledge 047, 154; 105: a development gain that did not replicate); three supports take a 0.5 prior to 0.95 |
| `belief_down` | 0.20 | three falsifications take a 0.5 prior to 0.0 and retire it from `next`: no single measurement retires an idea, since a baseline can move by the effect size between slots (knowledge 196) |
| `alpha` | 0.05 | the level of every interval and test this repository decides at (the promotion gate's `ALPHA`, the pre-registered rules of items 19-25) |
| `targets.pass@1` | 0.05 | the programme's acceptance: "held-out gain >= 0.05 with CI above 0" (items 1 and 19, item 21's programme rule) |
| `targets.open-book-pass@1` | 0.06 | item 21 S1's pre-registered go margin |
| `targets.train-task-coverage` | 0.10 | a tenth of the train tasks newly solved: about what item 24b's first generation added (15 new of 168) |
| `engineering` | `practice.jobs`, `practice.harbor` | the only campaign settings that change how fast a generation runs and nothing it learns or is measured on; everything else is refused until reviewed into the list |

A problem whose `metric` has no target is refused when added: a missing target is never replaced by a default.

**Verdict of a measured effect** (`experiences compare-arms` contrast, or any interval at level `1 - alpha`) against
the problem's target `d`: *supported* when the interval's lower bound is above 0, the difference is at least `d` and the
Holm-adjusted two-sided sign-test p is below `alpha`; *falsified* when the upper bound is below `d` (a gain the size the
hypothesis claimed is ruled out, even if a smaller one is real); *invalid* when there is no difference or no
interval; otherwise *inconclusive*. The family-clustered interval is used when the report has one, else the task
interval.

**Verdict of the promotion gate** (a campaign generation's `gate` record): *supported* when it promoted (the champion
moved, so the paper's "new best" holds by construction); *invalid* when any of its reasons is a verdict the gate did not
accept as evidence; *falsified* when the dev interval's upper bound is below `d`, or the retention interval lies wholly
below `-RETENTION_MARGIN` (harm shown, not just non-inferiority unshown); otherwise *inconclusive*.

**Belief update.** supported: `b <- min(1, b + belief_up)` and the candidate is adopted; falsified:
`b <- max(0, b - belief_down)`; inconclusive and invalid: `b` unchanged. Every verdict adds 1 to the problem's `N`.

**Priority and the next experiment.** `Pri = b / (N(P) + 1)` over the pending candidates of open problems with
`b > 0` and the requested phase (`hypothesis` by default). Ties break by higher belief, then fewer attempts on the
problem, then the problem id and candidate id, so the order is a total, reproducible order. `campaign next` proposes
the first; `show` lists every candidate with its priority.

## Hypothesis to campaign arm

`splinter campaign next <campaign.toml> [--write ARM.toml]` proposes the top candidate. When it declares a `value`, the
arm is the campaign file with:

- `name` = `<campaign>-<candidate id>` (lower case, unsafe characters as `-`);
- `champion` = the campaign's current champion (its ledger's, after promotions and rollbacks; the file's when the
  campaign has not begun), so the arm continues from the current best;
- `seed` = the campaign's seed (the ledger's when drawn), so the arm draws what the champion's generation drew;
- the one change `variable = value`.

The written arm is checked (below) before it is written, and runs as its own campaign (`campaign run ARM.toml`). Its
generation's gate is read back with `campaign problems <campaign.toml> update --candidate ID --experiment ID
--arm ARM.toml --generation N`, which re-runs the single-variable check against the parent before it accepts the
verdict. A candidate without a `value` (most of today's: a sven profile, a teacher dataset, a bench mount) is proposed
with its variable named; its arm is built by hand, checked with `campaign problems check`, and measured by whatever
the pre-registration names, its verdict entered from a `compare-arms` report (`--compare REPORT --arm NAME`) or stated
with a reason (`--verdict ... --reason`).

## The single-variable check

Both campaign files are read as campaign specifications and serialised to JSON, without the fields that are not a
variable of the experiment: `name`, `generations`, `budget` (how far it may go), `seed` (the draw) and `champion` (the
current best by construction). The rest is flattened to dotted leaf paths (an array is one leaf; a missing leaf equals
`null`). The arm passes when at least one leaf changed and every changed leaf is the declared variable or lies under
it (`train` covers `train.rank`). The variable must be at least two segments deep, or `base`: naming a whole section
(`train`, `curate`, ...) would let a multi-variable arm through as one. Refused: no change ("not a hypothesis"), and
any changed leaf outside the variable, each named.

## The firewall

An engineering candidate's arm, or any arm checked with `--engineering`, may change only leaves under the configured
`engineering` paths. Any other changed leaf is a breach, named and refused. The configured list itself is refused when
it reaches into model, algorithm, data or measurement: `base`, `rollout`, `curate`, `train`, `practice.dir`,
`practice.profile`, `practice.sven_build`, `practice.rung`, `practice.families` and `examine` are protected whatever a
document says, so no configuration can widen the firewall into what it guards.

## Inspect and override

- `splinter campaign problems <campaign.toml> show [--json]`: every problem with its evidence, attempts and candidates
  with belief, priority, status and history; the parameters; the state digest and event count.
- `revise --candidate ID --belief B --reason TEXT` and `revise --problem ID --status resolved|open --reason TEXT`: a
  coordinator's override, recorded as an event with its reason; nothing is edited in place.
- `update ... --verdict supported|falsified|inconclusive|invalid --reason TEXT`: a verdict stated by hand, for an
  experiment measured outside the campaign; the reason is required and kept.
- `add FILE`: new problems and candidates, or new parameters from that point on.

## Seed: the state on 2026-11-21

`examples/problem-state/problems.toml` holds today's six problems with their candidates, priors and evidence;
`examples/problem-state/verdicts.toml` holds the verdicts already recorded in `experience-to-competence.md`
(items 19, 21, 24, 25), each with the numbers it rests on. Replaying both gives the beliefs; the eval crate's
`problem_state` spec replays the fixture and holds the resulting order.

| Problem | Metric | Status |
|---|---|---|
| P-overthink: students over-think under a wall-clock budget | pass@1 | open |
| P-kb-unused: the 4B does not consult a mounted knowledge base | open-book-pass@1 | open |
| P-ratio-spikes: RL ratio spikes from proxy tokens | pass@1 | resolved (knowledge 188, `--sampled-only`) |
| P-distill-transfer: exam transfer of distillation is unproven | pass@1 | open |
| P-4b-data: the 4B-only data volume is small | train-task-coverage | open |
| P-trajectory-credit: teacher-free RL with trajectory-level credit sharpens practised tasks only | pass@1 | open (added 2026-11-21 with E0-E5) |

The 2026-11-21 training-methods review's experiments E0-E5 (`generic-learning.md`) are candidates of these problems,
one variable each: E0 `overthink.token-turn-budget`, E1 `distill-transfer.leave-families-out`, E2
`distill-transfer.min-cost-selection`, E3 `distill-transfer.family-breadth`, E4 `4b-data.path-guided-resampling`, E5
`trajectory-credit.branch-value-pairs`. They change the replay's top six (E0 second, E5 fifth).

## Open

| # | Item | Next action | Acceptance |
|---|---|---|---|
| PS-1 | Schema, storage, rules, checks and the two commands | landed with this ledger (see Landed) | specs green; validation rows with their mutants killed |
| PS-2 | E2a and E2b verdicts | enter them with `update --compare` from their `compare-arms` reports when read | the concise-traces and expert-iteration candidates carry a measured verdict |
| PS-3 | Engineering reward `dh - gamma ln(T_exec)` | record wall time per examined arm in the campaign's evaluate stage; then score engineering arms by it | an engineering arm ranked by a recorded reward, its outcome non-inferior |
| PS-4 | Insight re-scoring is manual | after a verdict, propose belief revisions for sibling candidates sharing its variable (proposal only, the coordinator records them) | a proposal list in `update`'s output |
| PS-5 | Campaign stages do not record verdicts themselves | let `campaign run` take `--hypothesis ID` and record its generation's verdict on its own | one arm run end to end with no manual `update` |

## Landed

- 2026-11-21: this design; the seed fixture; `splinter_eval::campaign::problems` (schema, fold, priority, verdict
  classification, belief update, single-variable and firewall checks); `splinter_pipelines::campaign::problems`
  (storage on the stage ledger, campaign-file configuration, protected paths, the arm `next` writes);
  `splinter campaign problems show|add|update|revise|check` and `splinter campaign next`.
