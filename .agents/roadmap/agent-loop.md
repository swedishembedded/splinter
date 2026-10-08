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
- [x] Strong baselines on identical folds: elastic-net Cox, additive model (existing); spline Cox authored by the loop (run run-20261007T203058.568-370c, accepted on attempt 2; specs run run-20261007T205231.010-e99c, accepted on attempt 1). Open: the README does not yet describe the two spline baselines (my acceptance check for it was defective, F-017), so the loop task for it is still to be run
- [ ] Residual model: additive predictor plus neural residual, multiplier initialised at zero. NOT DELIVERED: three attempts (run run-20261007T215753.920-5a61: time limit, behaviour check failures, torch API misuse) and an assisted continuation with a hint (run run-20261007T225456.244-8b2c: hidden check passed, its own spec test failed, time limit); the candidate then crashed on real folds (F-019). No residual-network result exists
- [ ] Capacity sweep (about 10k to 250k parameters) with train and validation NLL and IBS (blocked on the residual model; the existing recipe sweep of the deep encoder covers width, depth and masking only)
- [x] Learning curves by training fraction (repeat 0): elastic-net Cox, spline Cox, logistic, additive and the deep encoder at 10/25/50/75/100 percent; fits by a supervisor script kept outside the repository (the loop's tool for it was not delivered, F-018)
- [ ] Feature-block ablation with diet and meal timing added last
- [ ] Numerical-encoding ablation including piecewise-linear embeddings
- [ ] Timing-shortcut test (values only, timing only, both)
- [ ] Replace the 25-fold test with paired bootstrap; recompute power from the result
- [ ] Paper: results, uncertainty, negative-result wording; full read-through

### M3 Audit of calibration and evaluation code
- [x] Read the code: the expected calibration error takes the observed risk of each equal-weight risk group from an Aalen-Johansen estimate (a subject censored before the horizon is never a non-event); slope and intercept use inverse-probability-of-censoring weights from the training data's marginal censoring curve and drop subjects censored earlier; cause-specific observed risk is Aalen-Johansen; the time-dependent AUC is cumulative/dynamic with competing events excluded from the controls and says so (recorded in the paper, Appendix B)
- [ ] Gap: no integrated calibration index, E50 or E90 (the curve itself) for recalibrated, step-function outputs. Task text and hidden check written (`t4`, with hand-computed cases including heavy early censoring); not run for lack of server time. Rerun first next session
- [ ] A check that a marginal censoring curve is adequate when censoring depends on the predictors: not done (here censoring is administrative and depends on the survey cycle, which the horizon restriction handles)

### M4 Data and external validation
- [ ] External examination data for the older cohort (NHANES III) is not on disk: only a drug file and the mortality file. Two candidate addresses for the examination file returned not-found and no further search was made; recorded as blocked
- [x] Activity-monitor minute data acquired: 2003-04 (428 MB zip, 2.5 GB expanded) and 2005-06 (471 MB, 3.0 GB), integrity tested, recorded in the resource manifest. Not extracted or used: the long-sequence experiment needs a derivation of daily summaries and a sequence encoder in brain, which did not fit this session
- [ ] Access checklist for repeated-measure cohorts, with the one question each custodian must answer about distributing derived weights (user action; nothing was requested)

### M5 Learning and promotion
- [x] One local component trained with a real forward, backward, optimiser step, save, reload and evaluation: an 8B student, rank-16 adapter, 30 steps on patch-form records of accepted runs (loss 1.41 to 0.04, held-out 1.216 to 1.193), reloaded by eight fresh processes. The whole-conversation form is refused by the trainer (F-021, open in the model engine)
- [x] A failing candidate is rejected and the previous version stays usable: v1 solved 0 of 8 held-out tasks against 1 of 8; `models current` still names the base; nothing to roll back
- [x] Autonomy stage reported honestly: supervised local. Unaided local evaluation on unseen tasks: not run
- [ ] A candidate that improves: needs more diverse accepted runs, a trainer for multi-step tool conversations, and a baseline well above zero

### M6 Secondary estimands (pre-registered 2026-10-08, paper section 4.5)
Plan: new outcomes and labels go in sidecar files keyed by subject (`causes.jsonl`, `conditions.jsonl`); `timelines.jsonl` stays byte-identical (frozen digest). New code goes in new modules (`commands.rs`, `report.rs`, `external.rs` are near the 800-line gate).
- [x] S1 brain: restricted mean survival time, its grouped calibration, Gompertz reference table with delayed entry (brain dc09a817)
- [x] S2 brain: weighted AUROC/AP/calibration and screening operating points for a binary outcome (brain 8d01f12f)
- [x] S3 lifecourse: multiple-cause flags kept beside the timelines (`causes.jsonl`; rebuild reproduces `timelines.jsonl` byte for byte; cohort deaths with a mention: diabetes 971, hypertension 1325, one without multiple-cause data)
- [x] S4 cause curves for the weakest comparator: `cs-cox-agesex` baseline on 25 folds (the planned Rust recipe arms were not needed: the Python cause-specific Cox models already keep per-cause curves)
- [x] S5 T1 cause-specific accuracy at 5, 10, 15 years (`lifecourse causes`; paper 5.5; bootstrap O/E interval not run)
- [x] S6 T2 death with diabetes/hypertension flagged (`baselines/flags.py`, `lifecourse flags`; paper 5.6)
- [ ] S7 T3 restricted mean survival and mortality-equivalent age
- [ ] S8 T4 condition labels (sidecar)
- [ ] S9 T4 prevalence models, full and non-definitional inputs
- [ ] S10 T5 undiagnosed-disease screening
- [ ] S11 optional: accelerometer daily summaries as an input block (2003-06)
Open: creatinine standardisation for 1999-2000 and 2005-06 (read the laboratory notes before applying any equation); source for the bone-density T-score reference; PhenoAge needs alkaline phosphatase, which is not a concept yet.

## Next session, in order
1. Rerun the task for the integrated calibration index with E50 and E90 (`t4`), then the pooled paired bootstrap in `compare` (`t3`), after fixing the checks (F-017, F-019).
2. Rerun the residual-network task with a check on shuffled, offset folds; run the capacity sweep and the feature-block ablations on it.
3. Describe the two spline baselines in the lifecourse README (loop task).
4. Learning curves on repeats 1 to 4 and with several seeds, to tighten the fits.
5. Extract the accelerometer minute data, derive daily summaries as an input block, and compare a summary-feature model with a sequence encoder.
6. Brain: exact loss-mask boundaries for multi-step tool conversations under templates whose last-turn rendering differs (F-021).

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
| F-014 | high | With the reasoning block off, the 27B repeated one identical probe command about forty times and wrote no file; the attempt ended at the tool-call cap with nothing to judge | run run-20261007T184234.467-a64e, attempt 1 trajectory (calls 12 to 60 are the same command) | fixed: an attempt is stopped as `stagnation` when one call returns the same answer `--max-repeats` times (default 4), the retry is told which call looped; specs in `splinter-agent` work and the loop |
| F-015 | high | Root cause of F-014: the served 27B was decoding greedily (no temperature set) and fell into repeating one probe command; with a temperature of 0.7 the same task edited files within five minutes. Turning the reasoning block off (F-012) did not cause it; the reasoning-on run also repeated | runs 9e1c (greedy, thinking env ignored by the loop's hand-built configuration) and c44b (temperature 0.7: four `edit_file` calls in the first 75 events) | fixed: `--temperature` and `--thinking` are loop flags, carried through `Config`, applied to the served model and written in `model_selected` |
| F-016 | high | A follow-up round at a 58k-token conversation was refused by the server (`prompt 58498 + max_new 40000 exceeds context capacity 98304`): the remote path built its provider without the server's real context window, so sven could neither bound a reply by the room left nor compact in time | run c44b attempt 3, brain serve log | fixed: a remote model's provider is built with the window its server reports (`from_config_probed`, on a thread of its own); spec serves a `/v1/models` listing |
| F-017 | medium | My own acceptance script for the spec-test task chained two `grep` checks with `&&` under `set -e`, which does not stop on the first failure; the README requirement went unchecked and an accepted patch lacked it. Found by reading the accepted patch against the task | run run-20261007T205231.010-e99c | fixed in the check (separate statements); the README item is carried to a follow-up task |
| F-018 | medium | A larger loop task (learning-curve fit with bootstrap) was not completed by the 27B in two attempts (tool-call limit, then time limit); cancelled at attempt 3 for budget. Not counted as a model result beyond that: the task text asked for a lot in one run | run run-20261007T205818.780-7b26 | open: not delivered |
| F-019 | medium | My hidden check for the residual-network task built synthetic folds whose training rows were rows 0..n, so a model that indexed an array of training rows by global row numbers passed it and then crashed on the real folds (`IndexError: index 38223 is out of bounds for axis 0 with size 38223`). The accepted-looking candidate (hidden check passed, its own spec test failing, run run-20261007T225456.244-8b2c attempt 1) was therefore not usable; the loop did not deliver it | exploratory run of that candidate on repeat 0 | open: a stronger check (shuffled, offset folds) is needed before the task is rerun |
| F-020 | high | The first training run on the loop's own dataset was refused by the trainer: `message 2 is not prefix-stable under this template`. A tool-calling assistant turn carried the whitespace a model leaves around an empty reasoning block (`"\n\n"`), which the chat template renders differently alone and in context, so no loss mask could be fixed for it | `agent-loop train` on train-v1.jsonl | fixed: an assistant turn's content is trimmed in the projection; spec asserts no assistant turn keeps stray whitespace |
| F-021 | high | brain's SFT trainer cannot fix a loss mask for a multi-step tool conversation under Qwen3's chat template (an assistant turn renders differently when it is the last message), so the loop's whole-conversation records are refused (documented in brain as a known limitation, failing loudly). A trainer primitive for multi-turn tool trajectories is missing | `agent-loop train` on train-v1/v2 | worked around in the loop: `dataset --patch-form` writes one exchange (task, accepted patch); open in brain: exact boundaries for such templates |
| F-003 | low | Sven SDK bounds a run by deadline, output tokens and cancel only; no per-run tool-call cap | sdk RunOptions | loop enforces its own cap |
| F-022 | medium | `told_weak_kidneys` reads only KIQ022, but 1999-2000 asks KIQ020, so those subjects lack the input. Fixing it changes `timelines.jsonl` and the frozen digest | `nhanes/1999/KIQ.htm` against `concepts.rs` | open: recorded, not silently fixed; the new labels read KIQ020 or KIQ022 |
| F-023 | low | The treatment-for-diabetes variable is DIQ070 in all cycles but 2005, where it is DID070 | `nhanes/2005/DIQ_D.htm` | handled in the new labels |
