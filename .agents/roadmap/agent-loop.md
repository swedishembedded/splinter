# Agent loop and residual-survival research roadmap

Working roadmap for the supervised coding-agent loop (`samples/agent/loop`)
and the pre-registered survival experiments it is asked to carry out. Update
this file in the same commit that finishes or changes an item. The paper that
documents the work is `papers/agent-driven-residual-survival/paper.md`; it is
reviewed in full after every milestone.

## Standing rules for this effort

- One paper path for the whole effort (above). Findings land in it as they appear.
- Project drive stays under 500G; clean build targets first if it nears the cap.
- GPU use stays under 96G of VRAM; one GPU process at a time.
- Each repository (brain, sven, splinter) gets only changes it owns, as
  self-contained commits on a linear history, pushed to origin main.
- Delegated work must be authored by the loop; supervisor feedback and
  repairs are recorded so assisted and unaided results stay distinguishable.

## Open decisions

| Id | Question | Default if unanswered |
|----|----------|-----------------------|
| D1 | Delegate the experiments to the loop (A) or run them directly with the loop proven on a fixture only (B) | A, falling back to B per task; fallbacks are recorded as assisted or failed attempts |
| D2 | Loop gets caller-supplied tools through a new public `splinter-agent` solve (keeps the sample SDK-only) | yes |

## Baseline (2026-10-07)

| Repository | Commit | State |
|------------|--------|-------|
| brain | f5cde671 | clean, main |
| sven | 9c73800 | clean, main |
| splinter | 80ef1f9 | clean, main; Cargo.lock restored from HEAD, old stash kept |

Known failures before any change: see the findings ledger (F-001).

## Milestones

### M0 Protocol (done)
- [x] Paper skeleton and the decision rule fixed before any result
- [x] Map the three codebases (splinter agent/sdk, sven sdk and home conventions, health experiment state)
- [x] This roadmap

### M1 Loop bootstrap on a disposable fixture
- [ ] Repair the local build: samples must build against current brain (F-001)
- [ ] Public `splinter-agent` solve with a caller toolset and project root (D2)
- [ ] `samples/agent/loop`: task, workspace, limits (time, output tokens, tool calls, retries), local default, `--allow-api-models` opt-in mapped to the SDK's remote opt-in
- [ ] Run id, status, cancel, resumable checkpoint with reconciliation of uncertain actions, structured outcome, append-only JSONL trace with bounded events and artifact references, redaction
- [ ] Workspace wiring: `samples/agent/*` member, architecture entry, manifest gate glob, README with only working commands
- [ ] Seeded-failure fixture with an acceptance check outside the worker's writable scope
- [ ] Proof with a real local model; if blocked, name the concrete blocker
- [ ] Failed-attempt record and retry feedback exercised
- [ ] Budget exhaustion, cancel, provider failure and restart each show a controlled outcome
- [ ] Paper: loop section and fixture results; full read-through

### M2 Pre-registered experiments on the existing cohort data
Delegated to the loop. Decision rule: IBS gain of at least 0.001 with a paired
bootstrap interval excluding zero; anything less is a negative result, not
proof of equivalence. Evaluation uses one set of paired out-of-fold
predictions, subject and PSU bootstrap, and repeats model selection inside the
resampling loop. No architecture search against the same folds.
- [ ] Recover and commit the uncommitted recipe and learning-curve work held in the recipe worktree (rebase, do not recreate)
- [ ] Strong baselines on identical folds: elastic-net Cox, spline Cox, additive model
- [ ] Residual model: additive predictor plus neural residual, multiplier initialised at zero
- [ ] Capacity sweep (about 10k to 250k parameters) with train and validation NLL and IBS
- [ ] Learning curves by training fraction, additive versus neural, fit of error against N
- [ ] Feature-block ablation with diet and meal timing added last
- [ ] Numerical-encoding ablation including piecewise-linear embeddings
- [ ] Timing-shortcut test (values only, timing only, both)
- [ ] Replace the 25-fold test with paired bootstrap; recompute power from the result
- [ ] Paper: results, uncertainty, negative-result wording; full read-through

### M3 Audit of calibration and evaluation code
- [ ] Verify the reported claim that ECE uses Aalen-Johansen risk inside bins (not censored-as-negative) by reading the code and a hand-computable test
- [ ] Verify slope and intercept use censoring weights; check isotonic and Venn-Abers outputs are judged by curve, intercept, ICI and horizon Brier, not slope alone
- [ ] Fix any defect through the loop with a regression test

### M4 Data and external validation
- [ ] External examination data for the older cohort is not on disk (only a drug file and the mortality file): acquire or record as blocked
- [ ] Activity-monitor minute data is not on disk (codebooks only): acquire or record as blocked; run the long-sequence experiment only if acquired
- [ ] Access checklist for repeated-measure cohorts, with the one question each custodian must answer about distributing derived weights (user action)

### M5 Learning and promotion
- [ ] One small local component trained with a real forward, backward, optimiser step, save, reload and evaluation
- [ ] A failing candidate is rejected and the previous version stays usable (rollback pointer)
- [ ] Autonomy stage reported honestly: unaided, assisted, supervisor interventions

## Frozen

- Gated delta-net work on current sequence lengths: no evidence of benefit on sparse short histories; revisit only with dense long sequences.

## Findings ledger

| Id | Severity | Finding | Evidence | Status |
|----|----------|---------|----------|--------|
| F-001 | high | splinter main does not compile against current brain main: the adams and jefferson samples build `FineTune` without the new `held_out_text`, `keep_evaluations`, `weight_decay` fields | `cargo build --release --workspace` with local override at splinter 80ef1f9 and brain f5cde671 | open |
| F-002 | medium | Reported (by a mapping pass, unverified): the p-values behind the power estimate come from 25 folds treated as independent | review text and results table | open, resolved by M2 bootstrap |
| F-003 | low | Sven SDK bounds a run by deadline, output tokens and cancel only; no per-run tool-call cap | sdk RunOptions | loop enforces its own cap |
