<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Campaign audit: self-improving platform (Splinter + Brain + Sven), 2026-10-16 15:10 CEST

This is an independent, read-only audit. Nothing was edited, built or run on the GPU; the only
command run was `make -C papers/splinter-brain-sven-flywheel verify`. The Slurm lease ends at
16:48 CEST, about 1 h 35 min after this audit was written.

Heads audited:
- brain `c305c3ab` (the T1 lane was rebased onto main during the audit);
- sven `ebd169a2`;
- splinter `a556710`;
- L3 worktree `s-l3` at `e2f7a94` (not on main);
- T1 worktree at `0b39996a`.

Severity scale: **S1** blocks a correct or honest deliverable; **S2** is a material
over-claim or risk; **S3** is debt.

---

## 0. Summary (read this first)

1. **The flywheel paper on main is stale, and the useful work is stranded on an unmerged
   branch.** `papers/splinter-brain-sven-flywheel` on main has one commit, `ccc3f1d` (06:38). It
   reports only wave 1, where E2/E3 stopped at the split refusal.
   - The L3 branch holds 19 commits beyond main and is 29 behind it (162 files, about 49k lines). They contain:
     - the root-cause fix of that refusal (`468d2a8`, committed at 09:01);
     - the serve-check fixes (`bd868b4`, `0606a5b`);
     - the verifier fix (`2ca7dfe`);
     - the wave-2 data and pre-registrations.
   - None of this is on main. Until it is rebased onto main, splinter main still contains the bug
     that made E2 fail. **(S1)**
2. **The E2/E3 split failure is understood, and was fixed on L3 only.**
   - `crates/data/src/holdout.rs` (main), `choose_held_out` fallback: when every group is larger
     than half the units, it holds out "the newest tenth of the units" *by index*. That divides
     the one group the records share.
   - Every record from a one-file handbook is one overlap group (`crates/pipelines/src/grouping.rs:84`).
   - `verify_disjoint` then refuses the split it was handed (`holdout.rs:434`), with exactly
     "1 group(s) and 0 unit(s)".
   - The rule and its own gate contradict each other. No test covered a dataset that is a single
     group: `holdout/tests.rs` on main has none. **(S1, fixed on L3)**
3. **A full flywheel turn is now close, but on a 1.7B policy, not the 27B, and post hoc.**
   - L3 wave 2 with Qwen3-0.6B: every candidate passed improvement, retention and serve, and the
     anchor check refused every one.
   - Post hoc E2d with Qwen3-1.7B made the **first real release**: anchor 40/40, generation 0
     15/30 facts after the release, against 0/30 at base.
   - Its generation-1 `learn` was running at 15:10 (`l3work/pilot6g1`). If it releases and the
     rollback then runs, the loop has turned end to end once. It is one run, post hoc, on a
     sub-2B policy.
4. **The 27B cannot enter the flywheel at all today.** This is the largest architectural gap. Three
   pieces are missing:
   - Brain's SDK chat fine-tune refuses qwen35. `crates/sdk/src/study.rs:849` asserts the refusal
     "qwen35 does not provide chat fine-tuning yet".
   - Splinter never sets `.arch(...)`, so every flywheel training is qwen3.
   - `brain serve` adapters are Qwen3-only (`crates/cli/src/run_cli.rs:176`), and
     `qwen35/src/serve.rs` has no adapter path, so the gate's serve check cannot run on a 27B
     adapter.

   27B training exists only as the `brain qwen35 finetune` CLI, and 27B generation with an
   adapter only as `sample-adapter`, which has no KV cache (`qwen35/src/adapter_sample.rs:17`).
   **(S1 for the stated goal)**
5. **Throughput is real progress but not 1,000 tok/s.**
   - T1 measured 550-560 tok/s on a 64-layer random-weight probe at 8192 tokens with an NF4 base
     (knowledge 310). That was steps 1-2 only, on an idle-at-start card, with a neighbour
     arriving in step 3.
   - Nothing has been measured with the real checkpoint through the CLI on the new code. The last
     real-CLI number is about 2-5 tok/s on old code.
   - 1,000 tok/s needs about 6.6 s cut from a 14.8 s step: host launch overhead of about 5 s
     (114k dispatches) plus the Gated-DeltaNet chain at about 3.7 s.
   - Probability by 16:48: under 10%. Probability for the strict definition (real checkpoint,
     CLI, bf16 and NF4): about 0.
6. **The paper's existing text is mostly honest.**
   - `verify` passes: 116 STRUCTURE claims, 223 checks, 144 claims, 17 MEASURED, 0 problems.
   - The checks are weaker than the headline implies (section 2.3). The paper has also fallen
     behind brain (section 2.4): E4 says "no generation path for an adapter", and that is no
     longer true.
7. **Tests that claim GRPO correctness never run in any make target.** `grpo_fixes`,
   `grpo_qwen35`, `improve_cycle_qwen35` and `continual_study` need feature `qwen3`
   (`crates/rl/Cargo.toml:86-98`). `make test` passes no features, and `make test/rl`
   (`Makefile:444-448`) lists the older targets only. **(S2)**

---

## 1. Truly done vs only claimed

| Claim | Verdict | Evidence |
|---|---|---|
| GRPO trains the distribution that sampled; no step on an empty batch; per-token normalisation | **DONE** (code + non-tautological tests), **but no gate runs the tests** | `crates/rl/src/objective/grpo.rs:497-520`; `crates/model/src/train.rs:1163-1171`; `crates/rl/tests/grpo_fixes.rs` has a negative control (`:103`) and a sampler/trainer log-prob gap below 1e-3 on an independent path. Tests and code landed in one commit (`d1807c4e`), so red-first cannot be shown. The tests are missing from `make test/rl`. |
| Reduced GRPO ratio-basis ablation (E1 of grpo-corrections) | **Honestly reported as NOT MET** | Knowledge 292: the rule was committed at 11:29 and the results at 13:33. B loses to A on seed 3, and C beating A is labelled as outside the rule. The MET/NOT MET verdict is printed, not asserted (an `#[ignore]` experiment): acceptable. |
| Real 27B NF4 QLoRA adapter changes the greedy answer (knowledge 280, Q6) | **PARTIAL, over-labelled "Result: pass"** | Knowledge 280:102 says "Result: pass", but 280:106-111 itself shows the France control answered "Quenthara": catastrophic overwrite, no retention. The gate row in `training-27b.md` Q6 requires 50 steps, 8 examples, save/reload **and serving**. The run did 40 steps on 4 facts with `sample-adapter` and no serving. The held-out rewordings reuse the taught entities, so they do not show generalisation. |
| Real 27B NF4 step at 8192 tokens via CLI (knowledge 281) | **DONE, honest** | It refuses to quote throughput from a shared card. |
| 27B qualification in the paper (M11-M17) | **DONE as stated** | Logs in `data/runs/qual`. M16 "loss 2.78 to 0.0002" is memorisation of 8 examples, and the paper says so (validation = 2 training examples). M14: loss *rises* 2.99 to 3.58 over a one-step run. It is reported without comment and should be explained (first and last of what?) or dropped. |
| Throughput 550-560 tok/s | **MEASURED on a probe only** | Knowledge 310 (T1). Random weights, 2 steps, idle only at start. The W4 1,030-3,168 tok/s figures are 4-layer rates and must never be quoted as 27B rates. |
| "Generic across models" training | **PARTIAL** | Generic: `TokenObjectives`, `LoraSites`, GRPO/DPO over `Model`, conformance over qwen3/gpt2/qwen35/qwen35moe (`crates/rl/tests/conformance.rs:106`). Qwen35-only: the whole QLoRA machinery (tiers, recompute, NF4 frozen base, streamed FP8: `qwen35/src/train_base.rs:27`, `finetune.rs:213`). The SDK chat fine-tuners and `AdapterBank` are qwen3-only. C8 (`conformance.rs:243`) is `#[ignore]`. |
| Flywheel cycle completes end to end | **NOT DONE on main.** On L3: one post hoc 1.7B release (E2d g0); g1 in progress | Section 0 items 2-3. |
| Flywheel runs on the 27B | **NOT POSSIBLE with current code** | Section 0 item 4. |
| On-policy RL data from Sven (token ids, logprobs, sampling) | **Roadmap only; Sven emits text** | ATIF has `prompt_token_ids`/`completion_token_ids`/`logprobs` fields, but only `crates/atif/tests/spec_roundtrip.rs` fills them. `CompletionRequest` has no seed, top_p or logprob request. `TraceStep.model_name` is never set. The trajectory is rewritten whole at each flush (not crash-safe). This is enough for SFT and rejection sampling, **not** for importance-weighted RL. Splinter flywheel item 9 (GRPO stage) is open. `splinter crates/model/src/rl.rs` re-exports brain's `Improve`, but no pipeline uses it. "Fixed on-policy RL data" is therefore true only *inside brain* (its own sampler). |
| 8 repository task families with measured funnels | **The claim is wrong in the other direction: all 8 funnels are no-model mutation funnels** | Every catalog entry uses `--max-tasks 1000 --mutants-per-file 80 --budget 1s --no-release`, and `tasks/README.md` says "No model is loaded". "Candidates" are seeded mutants. **No family has a model-generated funnel.** The "5" is a measurement batch: caldate (`615f431`) plus linediff, mdrender, graphkit and tokenbucket (`a556710`); iniparse, ttlcache and exprcalc came earlier (`d36b5ad`) under an older binary with no `over_time_budget` bucket. Catalog tree ids and test counts match the disk for all 8. |
| Sven gate and memory records in trajectories | **Unit-tested, not wired** | `StepAssembler::push_gate_decision`/`push_memory_retrieval` (`sven crates/session-store/src/trace_session.rs:469,477`) have no production caller. `ThreadStore::gate_decisions()` (`crates/executors/src/thread_store.rs:228`) is never passed on. The memory tool never calls `record_memory_retrieval`. No written trajectory carries `extra.sven.gate`/`memory` today, yet `training-instrumentation.md:11` lists them under "What exists". |
| Paper claims verified | **TRUE for what the checks check** | `make verify`: 0 failures. See 2.3 for what the checks do *not* check. |

---

## 2. Mistakes, shortcuts and risks

### 2.1 Correctness

- **S1: the holdout fallback divides a group, and its own gate then refuses the split.** Main,
  `crates/data/src/holdout.rs`:
  - `choose_held_out`, the `if held.is_empty()` branch, holds out "the newest tenth of the
    units" by index. That contradicts `verify_disjoint` at `:434`.
  - Trigger: any learn from a single-text source. All records share one overlap group (`crates/pipelines/src/grouping.rs:84`).
  - The fix is on L3 at `468d2a8` (the set holds nothing out and names the cause), with a test.
    Rebase it.
  - Follow-up: "holds nothing out" means such a run has no held-out improvement suite of its own.
    Check that the gate then uses the reserved exam families and does not report an improvement
    from training data.
- **S1: the 27B is not reachable from Splinter.** The three gaps are in section 0 item 4.
  Nothing in splinter, the paper or the flywheel roadmap says plainly that "the flywheel trains
  qwen3-family models only". The selection test names `brain/unsloth/Qwen3.8-27B-Q8_0` as a
  *served remote* model (`crates/model/src/selection.rs:367`). That is inference through sven,
  not training.
- **S2: the anchor check cannot pass at n = 40 if even one answer is lost, and the
  pre-registration says otherwise.**
  - The gate bound is `DEFAULT_ANCHOR_BOUND = 0.02` (`crates/eval/src/gate.rs`), while one
    anchor question is 1/40 = 0.025. Any single lost anchor answer refuses.
  - `prereg/protocol.md:37` on main states "anchor correct may fall by at most 2 of 40 (the gate's
    own tolerance)". That is false for the code.
  - Every 0.6B candidate on L3 was refused by this check alone (candidate anchors 37, 33, 35,
    36, 34 of 40, against a base of 39).
  - The honest correction is **not** to loosen the bound post hoc. Either enlarge the anchor suite
    so that 0.02 means something (at least 200 questions), or register a bound in questions
    before the next run. Keep labelling E2d (a model swap after seeing the result) as post hoc,
    as L3 already does.
- **S1 (memory safety): CPU-backend heap corruption in the Gated-DeltaNet backward at batch > 1.**
  - Documented in knowledge 291, `generic-training-capabilities.md:12-15` and
    `training-27b.md:60-61`.
  - **No runtime guard.** `brain qwen35 finetune` on CPU uses the default `batch_size: 32`
    (`crates/model/src/train.rs:98`).
  - The ignored bf16 CLI test (`crates/cli/src/qwen35_cli.rs:764`) passes no `--batch`, so the
    "bf16 CLI path" failure is very likely the same bug.
  - Add a typed refusal now, then find the root cause under a sanitizer.
- **S3:** on a skipped GRPO step the AdamW step index still advances (`train.rs:1168`), which
  skews the bias correction.
- **S3:** `sample-adapter` has no KV cache (2.3 s per token). It is correct but unusable for any
  gate or eval over more than a handful of probes.
- **S3:** `l3work/pilot6g1` continues from a copied state whose serving manifest points at an
  adapter inside *another* state directory (`pilot6/state/artifacts/...`). The lineage is
  consistent today, but a rollback or relocation breaks it. Keep one state directory per
  lineage.

### 2.2 Over-claiming

- **S2: knowledge 280 "Result: pass"** and the `training-27b.md` statement "Q6 passes for the
  facts". Correct wording: *partial: behaviour change only; control answer overwritten;
  retention not shown; no serving.*
- **S2: paper E4** (main, `sections/06-results.tex`, last paragraph of E4) says "the
  qualification found no generation path for an adapter on this model". That is stale since brain
  `513bbcab` (`sample-adapter`). When it is updated, it must carry the overwritten control, or it
  becomes an over-claim in the other direction.
- **S2: the 1,030-3,168 tok/s W4 figures** in `throughput-27b.md` §8 are 4-layer rates. Any
  summary that quotes "over 1,000 tok/s" from them is false for the 27B.
- **S2: paper title and framing.** "Self-learning flywheel" and an "Autonomous Self-Improving AI
  Platform" are not yet supported:
  - no release has been made under the registered protocol;
  - no generation has been retained across two releases;
  - no rollback to a real release has been done;
  - no 27B flywheel run exists.

  The abstract on main is careful. Keep it that way.
- **S3: paper reproducibility.**
  - E2 ran on "Splinter `ce42a55` plus uncommitted repairs" (`sections/08-reproducibility.tex:16`),
    and the brain serve binary's commit was not recorded (`:18`). That run cannot be rebuilt.
  - The L3 wave-2 runs use the L3 binary, so wave-2 raw data is reproducible only once L3 is on
    main. Record the binary's commit and confirm the tree was clean.
  - Structure claims are pinned at splinter `2aed856d`, while the behaviour ran on a different
    build. The paper admits this in Limitations.

### 2.3 Weak verification (the claims verifier)

All 223 structure checks are `kind: regex` (`data/structure_checks.jsonl`). They show that a string
exists at a commit. They do not show a count or a behaviour. Examples:
- S01 "ten tiers": the check finds the word `foundation`.
- D01 "twelve views and six objectives": the check finds `pub enum Objective`.
- A03 "twenty-one rejection reasons": the check finds `pub enum Rejection`.
- V02 "a verifier never sees the trajectory": the check finds `pub trait Verifier`.
- V08 "the decision rule ignores abstentions...": the check finds `pub fn decide`.
- B10 "head dims 40, 128, 256": the check finds two function names.
- B11 "3:1 split": the check reads brain's AGENTS.md, not code.

At the pinned commits I spot-checked S01 (10 tiers), A03 (21 variants) and D01 (6 objectives).
All are true. The risk is drift: a future edit can make a claim false while `verify` stays green.

Fix: add a `count` check kind, and point the behavioural claims (V02, V08, G01) at a named test
that `verify` runs or whose name it requires. **(S2 for a paper whose selling point is "every
claim verified")**

### 2.4 Paper and ledger drift against the code

- The paper pins brain `9b495d21` and E4 brain `258eaba9`. Since then, brain has added:
  - `sample-adapter` and knowledge 280;
  - the GRPO fix (290);
  - the ratio-basis ablation, NOT MET (292);
  - the real CLI 8192 step (281);
  - the throughput work (310).

  None is in the paper. Either add them as MEASURED/CITED rows with raw logs copied into
  `data/runs`, or state explicitly that they are out of scope.
- The paper's E3 means "generation 1 of invented facts". `tasks/e3-plan.toml` (commit `29c89c9`,
  10:31) defines a *different* E3: three generations over 8 repository families with a sealed
  exam. It is not in `prereg/protocol.md` on main or on L3. Before it runs, register it under its
  own name (for example E5) with a decision rule, or it is an unregistered analysis.

### 2.5 Leakage and contamination in the flywheel data

- The split leak above is a refusal, not a silent leak. The gate worked. The design concern is
  that grouping by *whole source text* makes a one-document source unsplittable. For fact sheets,
  group by section or fact family (L3 uses one sheet file per generation and held-out whole
  families; that is fine).
- The E2 generator and teacher are the same model (Qwen3-4B, `data/runs/e2/raw/commands.log:5`).
  The teacher sees the source sections, so the "teacher verified" answers are open-book copies.
  That is acceptable for recall, but say so: the student is distilled from a teacher reading the
  sheet. It is not self-improvement from the student's own successes. The student solved 0 of
  36 attempts closed-book.
- Sealed exam families in `e3-plan.toml` (mdrender, graphkit) are enforced by `practice --plan`.
  The sub-audit could not confirm that a sealed family's files are unreadable to every stage
  (generator, teacher, rehearsal). Add a negative-control test: plant a sealed-family string and
  expect a refusal.

### 2.6 Splinter tasks, Sven, ownership and licences

- **Sven ownership:** `git grep -i splinter` finds one hit, the gate regex itself
  (`scripts/gates/check-repo-scope.sh:22`), which excludes its own file. Acceptable.
- **Sven:**
  - (S3) `docs/technical/rlm-context-tools.md:5` dates arXiv 2512.24601 to 2024; the id means
    December 2025.
  - (S3) `research-upgrades.md:4-5` points to a research note that is not in sven.
  - (S3) 13 new non-code files lack SPDX headers, consistent with repo convention.
- **E3 sealing and retention (S2):**
  - Sealed families are refused only by `practice --plan` (`crates/pipelines/src/experiment/mod.rs:351-386`,
    fail-closed on an unknown tree). Nothing seals them against `learn`, or against `practice`
    without `--plan`.
  - The plan's `retention` list (`tasks/e3-plan.toml:18`) is validated but **not consumed**. Gate
    retention comes from the champion's release lineage (`release/gating.rs:243`).
  - Either enforce both, or reword the plan's comment.
- **Risks in L3's split fix (S2):**
  - The refusal (`NoGroupToHoldOut`) still comes only after rollouts are paid for. Add a
    group-count preflight at intake.
  - `monitor_split_file`/`monitor_split_named` share `split_file`
    (`pipelines/src/train/mod.rs:336`, `train/rehearsal.rs:109`). A run with `--eval-every > 0`
    now hard-fails when the training records form one group. Monitoring should degrade to "none"
    with a recorded reason.
  - Add a test for fixed-side records that share a group with free records
    (`held_out_flags`, `holdout.rs:170-190`). That path is latent today only because absorb's
    fixed-side records have `group: None`.
  - The "0 unit(s)" count is always 0, because unit ids are synthetic per side (S3).
  - Knowledge numbers on L3 (056-060) and main (066/067) must be reconciled in the index on rebase.
  - L3 is about 19 commits ahead of main and 29 behind.
- **Eponym in a task tree (S3):** `tasks/graphkit/src/topo.lua:150` names an algorithm after a
  person. Editing it changes graphkit's tree id, so regenerate the catalog in the same commit.
- The ownership rule "brain never names splinter or sven" holds in code. The remaining hits are
  exempted by `scripts/gates/check-repo-scope.sh:18-20,42`:
  - CHANGELOG lines 2770, 4738, 12220, 12797;
  - `.gitignore:97` (`.sven/`);
  - the gate script itself;
  - a BPE token in `crates/data/assets/encoder.json`.

  Under a strict "never" these exemptions are themselves violations (S3). Decide and record the
  decision.
- Author-name attributions remain in older material:
  - brain: 94 "et al" lines in the repository overall, 58 files outside CHANGELOG. Examples:
    `crates/model/src/train.rs:136,1567`, `.agents/roadmap/fly.md:1217-1223`.
  - splinter: 23, for example `.agents/research/corpus-size-evidence.md:42-78` and
    `.agents/roadmap/voice-agent-paper-materials.md`.
  - The campaign's new files are clean: brain knowledge 270-310, the six campaign roadmaps, and
    the flywheel paper including `references.bib`, which has no author fields.
  - (S3) A scrub lane should replace each one with the work's title, year and arXiv id.
- Licences:
  - No new external dependency in brain's last 40 commits.
  - New source files carry SPDX headers.
  - `generic-training-capabilities.md`, `grpo-corrections.md` and `throughput-27b.md` lack
    headers, as do the other roadmap files (S3, consistent with convention).
  - cuBLAS is linked dynamically from the toolkit lane (T1). Record its licence position under
    the NVIDIA EULA for redistribution in `licensing-audit.md` before any binary is shipped (S2,
    release-time).
- Test gaps:
  - The GRPO tests are not in any make target (section 0 item 7).
  - Pre-existing silent skips on `MOE_SKIP_GPU_TESTS`: `crates/gradcheck/src/bf16_train.rs:394,407,422`
    and `arcface.rs:215,249,282`. They violate `.agents/rules/testing.md:82-90`.
  - On splinter main, `holdout/tests.rs` lacks the single-group case.

---

## 3. Critical path and course corrections

### 3.1 Before 16:48 today (about 95 min), in priority order

1. **Let L3 finish E2d generation 1 (running), then the rollback.** Owner: L3.
   - Do not change flags or retry.
   - If g1 releases, rollback to the g0 release is the first ever demonstration of release,
     retention and rollback. Report it as **post hoc, one run, Qwen3-1.7B**.
   - If the anchor refuses g1, that is the result.
2. **Rebase L3 onto splinter main, run the full gate chain, and push.** Owner: Sonnet lane "L3-land".
   - Gate chain: `make build test check` plus `make -C papers/splinter-brain-sven-flywheel verify`.
   - This lands `468d2a8`, `bd868b4`, `0606a5b`, `2ca7dfe` and the wave-2 paper.
   - Watch `structure_checks.jsonl` re-pinning, and confirm `commits.json` names commits that are
     on the remote.
   - This is the single most valuable action today.
3. **brain: one-line test gate fix.** Add `--test grpo_fixes --test grpo_qwen35 --test
   improve_cycle_qwen35` to `Makefile:444-448` (`test/rl`). Owner: Sonnet lane "B-gate".
   - CPU/qwen3-tiny cost only.
   - Run `make test/rl` on CPU, because the card belongs to T1.
4. **brain: runtime refusal of batch > 1 Gated-DeltaNet training on the CPU backend** in the qwen35
   and qwen35moe finetune entry, with a typed error and a test. Owner: "B-gate". No GPU needed.
5. **brain: correct the wording of knowledge 280 and `training-27b.md` Q6** to "partial". Owner:
   "B-gate". Five minutes.
6. **T1: stop optimising before the lease ends. Spend the last GPU slot on one real-checkpoint
   NF4 CLI step at 8192 tokens under the gpu lock with `other_contexts` recorded.** This turns
   "550 tok/s on a probe" into a real-model number, or shows the gap. Owner: T1. A real number
   at 300-550 tok/s is worth more to the paper than a probe at 700.
7. **Do not attempt today:**
   - a 27B flywheel run;
   - the repository-family E3;
   - a persistent Gated-DeltaNet kernel;
   - CUDA graphs.

### 3.2 Beyond the lease (ranked)

1. **Make the 27B a flywheel citizen** (brain + splinter, M-L). Owners: Sonnet "B-qwen35-sdk" and "S-arch".
   - (a) `ChatFineTune`/`PreferenceFineTune` dispatch to `qwen35::finetune_streamed` with tier
     options (bf16/NF4, recompute), and remove the refusal assertion at `study.rs:849`.
   - (b) A qwen35 adapter in `brain serve` (`AdapterBank` generic, KV-cached decode with LoRA)
     so the gate's serve check runs.
   - (c) Splinter passes `.arch()` from the policy's card.
   - (d) An acceptance test: tiny qwen35 random-init learn → gate → release → serve.
2. **The anchor suite is underpowered:** at least 200 frozen questions, the bound re-registered
   in questions, and a power note.
3. **Fix Q6 properly** (owner "B-q6"):
   - 8 facts plus rehearsal of the base's own answers (splinter T05 already does 0.25);
   - save, reload and serve;
   - the control answer must be preserved.

   That is the honest "27B learns without forgetting" claim.
4. **The GRPO stage in splinter** (flywheel item 9), fed by Sven trajectories with token ids,
   logprobs and sampling parameters. Sven roadmap `13409d03` must land first. Until then, the
   "fixed on-policy RL data" claim is a **brain-internal** claim only.
5. **Throughput toward 1,000 tok/s** (T1):
   - K8 (arena, parameter ABI, per-layer CUDA graphs) for the launch overhead of about 5 s;
   - a persistent Gated-DeltaNet backward kernel for about 3.7 s;
   - an exclusive-card baseline (a reserved node, or a lock that refuses to start with other
     contexts present).
   - Realistic horizon: days, not hours.
6. **Harden the claims verifier:** `count` checks, and test-backed behavioural claims.
7. **Root-cause the CPU heap corruption** (sanitizer build of `backend-cpu`/`wgsl-cpu`).
8. **Register the repository-family E3 (`tasks/e3-plan.toml`) as its own experiment** with a
   decision rule, then run it.
   - All 8 family funnels are mutation-only, so measure at least one model-generated candidate
     funnel per family first.
   - Re-measure iniparse, ttlcache and exprcalc on the current binary.
   - Enforce sealing in `learn` and plan retention in the gate.
9. **Sven training-grade trajectories** (sven `training-instrumentation.md` items 1-3): token ids,
   logprobs, seed and sampling parameters, append-only writer. Also wire the existing gate and
   memory records end to end, with an agent-run integration test. Owner: Sonnet "V-traj".
10. **Scrub author names** from older material (S3).

### 3.3 What to cut or defer

- **Cut from today's paper:**
  - any throughput number not measured on the real checkpoint;
  - any 27B "learns" claim beyond knowledge 280's partial result;
  - any GRPO improvement claim (E1 is NOT MET).
- **Defer:**
  - the scheduler (flywheel item 10);
  - environment lifecycle;
  - the efficiency and safety gate dimensions;
  - the sven memory and continual-learning roadmaps;
  - the repository-family E3.
- **Stop:** new families, new synthetic-author features, and new persona work until the
  flywheel turn and the 27B entry are landed. The 10:xx commit burst on synthetic-author is
  off the critical path.

---

## 4. Verdict

- **1,000 trained tok/s on the 27B by 16:48:** not achievable; under 10% even for the probe.
  - The engineering path is credible: 20 to 550 tok/s in one day, with a profile that names the
    remaining 6.6 s.
  - 1,000 tok/s within several working days is plausible on the probe, and less certain through
    the CLI on the real checkpoint with the bf16 base, which holds no freed blocks.
  - The minimal honest claim today is: *"From 20 to 550-560 trained tokens/s on a 64-layer,
    8192-token random-weight probe of the 27B architecture with an NF4 base, on one GH200 idle at
    step start (two steps); not yet measured on the real checkpoint; target not met."*
- **Completed flywheel E2/E3:**
  - **Registered protocol:** not met. No release in wave 1 or wave 2. That is a valid,
    reportable result.
  - **Post hoc (1.7B policy):** one release has happened. A full two-release turn with retention
    and rollback may complete today if E2d g1 passes the anchor. It must be labelled post hoc,
    single run, sub-2B.
  - **On the 27B:** impossible until 3.2(1) lands.
- **Minimal honest paper claims (if nothing else lands):**
  1. The architecture and gate exist as described. STRUCTURE: string-level checks, with their
     limits stated.
  2. E1 funnel numbers.
  3. The registered E2/E3: the gate refused every 0.6B candidate *on the anchor alone*, while
     the candidates learned 12-18 of 30 invented facts (base 0) with no gain on never-taught
     facts. That is a real negative control showing that the gate catches drift.
  4. Post hoc E2d: a 1.7B candidate learned 15/30 facts and kept 40/40 anchor answers, and was
     released and served.
  5. E4: 27B steps fit and run on one card (bf16 to 16k, NF4 to 8k), and memorisation of 8
     examples.
  6. The 27B adapter changes greedy answers but overwrites a control: forgetting, observed.
  7. GRPO: the on-policy correction is implemented and unit-verified; the reduced ablation did
     **not** meet its rule.
  8. Throughput as in the claim above.

  Everything else is roadmap.
