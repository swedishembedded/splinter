<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Campaigns and the promotion gate

Scope: `splinter campaign` (resumable multi-generation learning over the stage ledger, `splinter_pipelines::campaign`) and the one promotion gate on the examiner's signed aggregates (`splinter_eval::promotion`).

## Done (2026-11-18, lane C of the experience-to-competence campaign)

- One promotion gate, `splinter_eval::promotion::decide`: dev gain (two-sided 95% interval above 0), retention
  non-inferior (one-sided 95% bound above -0.05) and a fresh sealed slice, each on a `bench.examination/1` verdict
  verified in-process (HMAC-SHA256 over bench's canonical bytes, knowledge 174; schema, validity, integrity flags,
  candidate and predecessor identities, `alpha = 0.05`, the slice's burn record). Typed reasons. Bench's verdict carries
  the burn record since bench 63a1b127 (docs/examination.md "Burn").
- `splinter campaign run|status|rollback` on `CampaignController`: rollout (practice material, frontier by
  `PassRates`), ingest, curate (dataset id = manifest digest), train (`train::train_identified`, brain `resume_key`,
  knowledge 175), evaluate (`bench examine` x3), gate, promote; every stage a stage-ledger unit keyed by content;
  budgets (rollouts, GPU-hours, tokens); Ctrl-C through the run's cancel token. Champion history append-only in the
  ledger; rollback with reason.
- The earlier in-memory arm loop of `CampaignController` (calibrate/collect/train/dev gate, CAS state) was retired: no
  production consumer; its served collectors, validators and trainers (`campaign::{served, endpoint, brain, factory}`)
  stay for samples and the episode executors. campaign-lua's dead `campaign_config` and the recipe flags only it read
  were removed.
- Tests: eval `promotion_gate` (5, real bench-signed fixture), canonical/mac units (4), model
  `served_train` resume test, pipelines `campaign::generations` (7: two-generation E2E, kill mid-rollout and
  mid-training at seed-drawn points, duplicate delivery, forged and mismatched verdicts, rollback, budget, identity).

## Done (2026-11-18, lane C2)

- The gate verifies bench's Ed25519 signatures with bench's public key (`promotion::PublicKey`, ed25519-dalek =3.0.0,
  strict verification); Splinter holds no signing key (`[examine] public_key`, the `.pub` of `bench keygen`). Clean break:
  a bench `hmac-sha256` verdict is `unsigned`. Schemas: `bench.examination/2` and the agent-protocol `bench.verdict/3`
  (bench 80ab3135: candidate and predecessor identities, `sealed_exams`, one burn record per exam in `burns`,
  `one_sided_lower` per comparison, the `served_model` check). On a verdict/3 the gate reads `retention` for the
  retention rule and `overall` otherwise; a burn of an exam the verdict is not over, an exam without a burn, an
  `Invalid` verdict or a control arm are typed rejections.
- `promotion::decide_unexamined`: the reviewed rule where no sealed slice applies (dev gain plus retention on the
  caller's own measurements), recorded as `rule: not_exam_verified {why}`, `exam_verified: false`.
- `splinter campaign`'s evaluate stage examines on bench's agent protocol (`[examine] protocol = "agent"`,
  `[examine.agent]`): per collection one `bench run` per model under a profile derived as practice derives one
  (endpoint, session `<name>-gNNNN`, the declared serving build), jobs in bench's private directory, one `bench
  verdict` per pair with the candidate files, the retention suite as anchor, `--alpha 0.05`. Spec
  `campaign::agentic` (scripted bench double).
- The rollout stage's tokens are summed from each trial's score-record usage (prompt plus completion), so
  `[budget].tokens` binds (knowledge 176; spec `bench_practice::as_a_campaigns_rollouts_...`).
- `bench::examine::Bench` (the minimind sample's examiner) verifies verdicts in-process with the public key and names
  each sitting's checkpoint in a candidate file (the workspace-tools exam is sealed, so bench requires the pair).

## Done (2026-11-19, lane K: capability tracking and the capability curriculum)

- Bench's capability ids reach splinter as data: `practice.json` tasks (`capabilities`, `track`, `taxonomy`,
  `selection`), export index rows, manifests and `skipped.jsonl` lines (bench, same day). `curriculum::capability`
  counts competence per capability (graded passes and fails, 95% Wilson interval, ungraded infrastructure stops kept
  out of every denominator, failure classes from the diagnosis, the campaign's bands) and the weight
  `(1 - r)(1/2 + f/2)(1/2 + m/2)`. `splinter experiences capabilities DIR... [--labels PRACTICE_DIR]` reports it per
  policy and keeps it in the store (`capability_competence`) with the index and practice digests it came from;
  material staged before bench named capabilities is labelled only by matching `source_digest`.
- `[rollout] curriculum = "capability"` (opt-in; `all` keeps keys and records unchanged): a generation's tasks are
  drawn by weight from a seed of the campaign's, retired tasks and tasks the infrastructure stopped every time never;
  `tasks` caps them; `transfer = { capabilities, count }` has bench draw fresh exercises for the heaviest measured
  capabilities (`bench task practice --count --seed --capability`) beside them. Every trial records its class and
  capability ids; the rollout record keeps the selection. Specs: `capability_curriculum` (real practice manifest
  fixture, interval, frontier preference, infrastructure exclusion, seed, sealed and out-of-pool negatives) and
  `campaign::capabilities` (E2E: generation 2 practises more on the capability generation 1 failed).
- Real data (knowledge 180): A2s1's G=8 practice, 1280 trials over 160 tasks of four collections, labelled by
  re-staged draws byte-identical to lane R's: design-review and professional-procedure 0.40 [0.33, 0.47],
  exact-arithmetic 0.44, abstention 0.89, stateful-service 0.87; the curriculum would weigh design-review (0.49),
  professional-procedure (0.47) and exact-arithmetic (0.44) first.

## Open

- No campaign has run with `curriculum = "capability"` on real models (no GPU in lane K); the transfer draw's
  bench call is covered by the argv it builds and bench's own tests, not by a live rollout.
- A `--count` draw selects after admission (N drawn, fewer kept): a variant's labels exist only once its tasks do.
- Three capabilities (apply-rules, locate-in-workspace, structured-answer) label nearly every task of the four
  knowledge collections, so their weight follows the overall pass rate; the discriminating ids are the narrow ones.

- Done 2026-11-18 (splinter baa35b8fe, lane G landed it): `release` and `harness optimise` decide on the one
  promotion rule, `promotion::decide` on the examiner's verdicts when the caller hands them in, else
  `decide_unexamined`, recorded not exam-verified with why; the sign test and per-suite drop are reported only.
  Small-sample fix (24709ee31, knowledge 177): a dev gain counts only over at least `MIN_GAIN_FAMILIES = 6` families
  (`too_few_families`), on bench's verdicts too (the section's `families`). Open: a release from `learn` still has no
  sealed slice (bench has no exam of the learner's material); retention's bound has no family minimum.
- The agent protocol's evaluate stage has not run against a real bench and brain (no GPUs in lane C2); its contract is
  covered by the scripted double only. Bench's `served_model` check needs brain's `brain` object on every reply.
- The agent examiner passes `--min-gain` and `--alpha` itself; bench's pre-registered exam plans (`bench verdict --plan FILE
  --slice NAME`, bench 8aeafd63) are not yet named by `[examine.agent]`. Name the plan and slice per generation there.
- A minimind sitting's checkpoint is named by its sitting, not by a content digest (identity none, like a base): its
  burns name the pair (none, none). It decides on bench's Promote, not on the gate.
- `generation run` (`pipelines::generations`) is a second multi-generation driver over `practice`; its release already
  goes through the one promotion rule (baa35b8fe). Not retired yet: retiring it needs repository rollouts and an
  in-process examiner for the campaign and a campaign gate stage on `decide_unexamined`; acquisition.md's
  "Generations on the coding path" entry lists what is missing and the consumers to move.
- No real-model campaign has run (no GPUs in this lane).
