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

## Open

- `release` and `harness_opt` still decide with their own gates (release: sign test, retention bound, anchor, serve,
  canary over Splinter's in-process probes; harness_opt: family sign test or evidence gate). Moving them onto
  `promotion::decide` needs bench verdicts for their candidates (a release from `learn` has none today) or a reviewed
  rule for when the sealed slice does not apply. Not done.
- The examiner is `bench examine` (direct completion). Agentic candidates are judged by `bench run` + `bench verdict`
  (`bench.verdict.v1`), which carries no candidate identity and no burn record: the campaign cannot gate on it yet.
- The rollout stage records no generated tokens through `PracticeRollouts` (bench's export has per-trial token counts;
  not summed yet), so a token budget only binds with an executor that counts them.
- `generation run` (`pipelines::generations`) is a second multi-generation driver over `practice`; retire it onto
  the campaign once its consumers move.
- No real-model campaign has run (no GPUs in this lane).
