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

## Time budget

The whole effort is closed within 8 hours of wall clock: started 2026-10-07
19:53 (+02:00), hard stop 2026-10-08 03:53. Anything not done by then is
reported as open with its resumable state, never claimed.

| Window | Work |
|--------|------|
| 19:53 - 20:40 | Reconcile the 44-section plan with what exists; sven-reuse audit (R); this roadmap |
| 20:40 - 23:00 | M2 experiments on the existing cohort (baselines, residual model, capacity, learning curves, bootstrap) |
| 23:00 - 00:30 | M3 calibration and evaluation audit; F-008 |
| 00:30 - 02:00 | R repairs (sven SDK gaps first, then the loop), M5 learning and promotion proof |
| 02:00 - 03:20 | Paper full pass, documentation, full verification chains in all three repositories |
| 03:20 - 03:53 | Buffer, push, final report |

## Open decisions

| Id | Question | Default if unanswered |
|----|----------|-----------------------|
| D1 | Delegate the experiments to the loop (A) or run them directly with the loop proven on a fixture only (B) | A, falling back to B per task; fallbacks are recorded as assisted or failed attempts |
| D3 | The loop and `splinter-agent` reuse sven's SDK, trajectory format, hash-chained log and home conventions instead of parallel copies; a missing piece is added to sven-sdk (generically, never naming splinter) | yes |
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
- [x] Repair the local build (F-001, fixed upstream; rebased)
- [x] Public `splinter-agent` worker with a confined toolset, tool-call cap, observer and suspend/resume (D2)
- [x] `samples/agent/loop`: task, workspace, limits (time, output tokens, tool calls, retries), local default, `--allow-api-models` opt-in mapped to the SDK's remote opt-in
- [x] Run id, status, cancel, resumable checkpoint with reconciliation of uncertain actions, structured outcome, append-only JSONL trace with bounded events and artifact references, redaction (specs with a scripted model pass)
- [x] Workspace wiring: member, architecture entry, manifest gate glob, README (commands to be verified against the real run)
- [x] Seeded-failure fixture with an acceptance check outside the worker's writable scope
- [x] Proof with a real local model: Qwen3-8B in-process (rejected, two attempts) and Qwen3.8-27B served by brain (accepted, one attempt, unaided); see the paper
- [x] Failed-attempt record and retry feedback exercised (the 8B run's attempt 1, and specs)
- [x] Budget exhaustion (total time), cancel, provider failure (seen for real: a served model's context too small) and restart each show a controlled outcome (specs)
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

### R Reuse sven, do not reinvent it
Audit at 2026-10-07 20:25. Already reused: `sven_sdk` engine, tools, `AgentState`
suspend and resume and the ATIF trajectory (`crates/agent/src/work.rs`).
Candidate duplication to resolve:
- [x] `trace.rs`: the event stream is now sven's hash-chained log. `sven-sdk` re-exports `chain` (sven 79bca37); the stream is verified whenever it is read, concurrent writers extend one chain and a torn last line is repaired. The conversation itself was already sven's ATIF trajectory
- [x] Effective definitions: `sven-sdk` re-exports sven's discovery (`workspace`, sven d4ddb0c); the loop records every subagent, skill, command and the project context file in effect, with text digests, in the contract and in a `definitions` event, and flags `definition_changed` on resume
- [x] `redact.rs` stays: sven has no secret redaction in its traces to reuse (searched `crates/` for it), so the loop's is the only one; if sven gains one, replace this
- [x] `store.rs` keeps its own run directory under `~/.sven/loop/`: `sven-session-store` stores chat sessions as ATIF trajectories, which a run (contract, checkpoint, patches, outcome) is not. The two share the home directory and the trajectory format
- [x] No sven change names splinter

## Frozen

- Gated delta-net work on current sequence lengths: no evidence of benefit on sparse short histories; revisit only with dense long sequences.

## Findings ledger

| Id | Severity | Finding | Evidence | Status |
|----|----------|---------|----------|--------|
| F-001 | high | splinter main did not compile: the adams and jefferson samples built splinter's own `FineTune` without the fields its trainer gained (`held_out_text`, `keep_evaluations`, `weight_decay`); a release-manifest test lacked `rehearsal`. Not a brain break. | `cargo build --release --workspace` at 80ef1f9 | fixed upstream in e3c62a8 (found on origin; my identical edits dropped, diffs kept) |
| F-002 | medium | Claim from the external review that our p-values treat 25 folds as independent. REFUTED for the existing analysis: `results.md` uses the corrected resampled t-test (Nadeau-Bengio variance correction). The power estimate stands as an approximation; a subject and PSU bootstrap on pooled out-of-fold predictions is still the stronger check | `baselines/results.md`, paired comparison blocks | closed as stated; bootstrap kept in M2 |
| F-004 | medium | A `samples/*` workspace glob takes `samples/agent` (no manifest) for a package and fails every cargo command; `exclude` also drops its children | cargo error when `samples/agent/loop` was added | fixed: samples listed explicitly |
| F-005 | high | A timed-out acceptance check left its grandchildren holding the output pipe, so the kill took as long as the command (30 s for `sleep 30`) | spec `a_command_that_runs_too_long_is_killed_and_fails` failed after 30 s | fixed: each check runs in its own process group and the group is killed |
| F-006 | medium | A small local model ends a turn by announcing its next step ("Let's fix the indentation and run the tests again") without calling a tool; the runtime counts a text-only reply as a successful run, so a whole attempt was lost with the checks red | run run-20261007T071804.536-bbcb, attempt 1 (Qwen3-8B) | fixed: the attempt goes on in the same conversation, told what failed (`--follow-ups`, default 2); effect measured in the paper |
| F-007 | high | A bytecode cache left by the previous run of a check made the next check judge code that was no longer there (same size and mtime second) | spec `a_worker_that_stops_with_the_checks_red...` failed with a stale `__pycache__` in the candidate | fixed: checks run without writing bytecode, stale caches are cleared before each check, bytecode is kept out of the patch |
| F-008 | high | `timeline_release::a_better_candidate_is_released...` fails deterministically at origin/main against the pinned brain: the candidate's 5-year calibration intercept is 0.788 outside the gate's [-0.5, 0.5]. brain made logistic calibration the default after the test was written | same value in three runs, also in a pristine worktree of e3c62a8 | open: not caused by this work; decide whether the spec should pin the calibrator it was written for or the toy cohort should be larger |
| F-009 | medium | The prompt listed each acceptance command, so the worker tried to read the hidden acceptance script (the file tools refused; the shell is not a sandbox) | 27B run, transcript | fixed: a hidden check is named, not shown (`--accept` vs `--accept-visible`) |
| F-010 | medium | The per-request output cap equalled the whole attempt budget (60000), so a served model with a smaller context refused every request | brain serve log: prompt 6668 + max_new 60000 exceeds context capacity 2048 | fixed in the loop's default (24000); the served model needs a context of at least prompt + budget (`BRAIN_QWEN35_GGUF_CTX`) |
| F-011 | low | sven `cargo clippy -p sven-sdk --all-targets -- -D warnings` fails on five `chunks_exact` lints in `sven-audio` under the toolchain on this host (clippy 1.99); not touched by this work | `crates/sven-audio/src/lib.rs` lines 195-215 | open, pre-existing |
| F-012 | high | A model served by brain reasoned before every reply (5k to 13k output tokens per call, empty visible text), because the remote-model path used the user's sven configuration without asking for no reasoning block; two attempts of the spline-Cox task ended at the output-token cap with no file written, and each cost about twelve minutes | run run-20261007T181133.166-6059, `model_usage` events | fixed: a `brain` provider model is sent `enable_thinking: false` like the in-process model (spec in `splinter-model` selection); whether reasoning helps hard tasks is untested |
| F-013 | high | The first version of the chained event stream could not read runs written before it: `agent-loop cancel` on an older run failed with `malformed entry: missing field prev_hash` | cancel of run-20261007T080242.372-ce1c | fixed: legacy streams are read, and migrated when resumed (regression specs in `trace`) |
| F-003 | low | Sven SDK bounds a run by deadline, output tokens and cancel only; no per-run tool-call cap | sdk RunOptions | loop enforces its own cap |
