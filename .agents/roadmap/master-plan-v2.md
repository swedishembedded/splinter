<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Master plan v2: real coding-task competence first (2026-10-16, evening)

This plan merges the user's course-correction plan (input:
`~/project/shared/containers/runs/plan-2026-10-16/correction-plan-input.md`, Phases 0-7, gates
G0-G15, experiments E0-E14) with the grounded `master-plan.md` (v1) into one executable order.
v1 stays in the tree as the record of how its packages were derived; where v1 and v2 disagree,
v2 decides. Nothing here is a finding for a paper.

Heads read:
- brain `origin/main` = `8b652351` (T0 landed on top of `21d4e13c`); local `main` holds one
  unpushed L0 commit `4137781d` (B1).
- sven `origin/main` = `5d08cbea`. The checkout `~/code/sven` is on `ebd169a2`, 16 commits
  behind; every sven line number below is read at `5d08cbea` with `git show origin/main:<path>`.
- splinter `origin/main` = `17e9202` (OPS lease kit landed on `f5527c2`).

Labels on every fact:
- **VERIFIED-BY-RUN**: a command run during this planning session (CPU only, no GPU, no build).
- **READ-IN-CODE**: read at the named file:line at the heads above.
- **DERIVED**: computed from the two kinds above; the arithmetic is shown.
- **UNVERIFIED**: stated by the input plan and not checked (all of its arXiv citations).

---

## 0. Summary

1. **The objective changes.** v1 ended in a registered 27B flywheel turn on invented facts (R2).
   v2 ends in G9: a statistically supported increase in verified real coding-task completion, on
   tasks no training run saw, by the same saved adapter served through `brain serve` and driven
   by sven. Invented facts leave the primary pipeline and stay as a test fixture.
2. **Most of the machinery exists; the benchmark does not.** Splinter already has `practice`
   (snapshot, profile, tasks, reserve, solve, verify, frontier, dataset, train, release), a
   workspace verifier with hidden tests and protected paths, an exposure audit, canaries,
   hidden-suite kill rates, an action-level SFT projection that masks tool observations, a
   clustered sign-test gate with power and pilot labels, `experiment check` with sealed families,
   a run record, `doctor` and a continual matrix. What is missing for Phase 1: tasks from anything
   other than seeded mutants, Rust tasks, partitions beyond "exam reserved per file", a per-task
   cost record, and a failure class.
3. **First wave after L0 (lease L1), all CPU except short probes:** P0-ACC (cross-repo acceptance
   test), P0-INV (inventory and discrepancy report), P1-OUT (TaskOutcome and cost capture),
   V4 (sven cost telemetry), B9 (adapter artifact contract), P2-VER (verifier proves the tests
   ran), plus the throughput lane T1 on the otherwise idle GPU.
4. **The first GPU-heavy science is E1, the untrained-agent baseline (lease L3).** Nothing is
   trained for a result before E1 is reported.
5. **The first training result (E4) runs on Qwen3-8B, not the 27B**, because qwen3 SFT,
   adapter save and adapter serving work today (`brain/crates/sdk/src/finetune.rs:437-625`,
   `brain/crates/qwen3/src/serve/adapters.rs:322-405`). The 27B runs the ten-stage
   qualification (E6) in parallel and becomes the policy only when it passes G7 and G8.
6. **GRPO (S8, S9, B7 rest, B8) moves behind G9** (input Phase 6). The AdamW skip-step index
   fix moves forward into Phase 4 because it corrupts SFT too.
7. **The throughput lane stays and never blocks Phases 0-2.** It runs in GPU windows the
   measurement lanes leave free (L1, L2) and its value is a 1.6x cheaper Phase 4 27B SFT and
   Phase 6 rollouts (DERIVED: 622 -> 1,000 tokens/s).
8. **v1 had one wrong fact.** F9 ("LoRA is not applied in the qwen35 decode path") is false:
   `lora_fwd` is reached from decode through the shared layer functions; the decode step instead
   panicked on `tok.weight` for a narrow-base build. The L0 lane found and fixed it
   (`4137781d`, red run named in the message).

---

## 1. Constraints that override the input plan

| # | Constraint | Label | Evidence |
|---|---|---|---|
| K1 | One GH200, about 96 GB device memory usable for one process at a time (the input assumed 2 x 96 GB) | READ-IN-CODE (lease) | v1 F1, F11; brain knowledge 312 |
| K2 | 8-hour leases; a node change loses the scratchpad, keeps the project drive | READ-IN-CODE | v1 section 6.3 |
| K3 | Project drive 500 GB, 130 GB free (75% used) | VERIFIED-BY-RUN | `df -h ~/project/shared` |
| K4 | Disk users: `targets/sven` 70 GB, `targets/splinter` 28 GB, `targets/brain` 25 GB, `targets/v-traj` 6.1 GB | VERIFIED-BY-RUN | `du -sh ~/project/shared/containers/targets/*` |
| K5 | Base models on disk: Qwen3-0.6B (1.5 GB), 1.7B (3.8), 4B (7.6), 8B (16), 14B (28), Qwen3.8-27B-FP8 (29, `model_type` qwen3_5), 27B Q8_0 GGUF (28) under `~/project/shared/containers/brain-models` | READ-IN-CODE (listing) | brain sweep |
| K6 | No container daemon and no rootless runtime on the node: `docker` client present, socket missing; no podman, apptainer or singularity | VERIFIED-BY-RUN | `docker version`: "failed to connect ... /var/run/docker.sock"; `command -v` |
| K7 | Splinter's container sandbox is docker only | READ-IN-CODE | `splinter crates/sandbox/src/container.rs:1-31`; config `crates/orchestrator/src/config.rs:101-124` |
| K8 | 27B training rate 622 tokens/s (probe, NF4, 8K), 635 (real checkpoint); target at least 1,000 | VERIFIED-BY-RUN in v1 | v1 F1, F5 |
| K9 | At most 2 concurrent cargo builds; every GPU process under the lock | READ-IN-CODE | `splinter scripts/lease/{slot,gpu}` (17e9202) |

Consequences:
- **Isolation (K6, K7).** Every workspace run in this campaign has isolation `none`, recorded
  as such (`orchestrator/tests/workspace_environments.rs:124`). The exposure audit detects, it
  does not prevent. The input's "unsupported environment class if isolation is missing" is
  applied as: tasks whose verifier needs isolation to be sound (network, secrets) are not
  admitted; every report states isolation `none`. Decision D-V2-3.
- **Storage budget (K3, K4), scaled from the input's 300 GB to our free space:** base models
  read-only and already counted; adapters and checkpoints 25 GB; experience, datasets and
  benchmark snapshots 20 GB; lane targets 60 GB (4 at once, as v1 6.4); build tmp 10 GB;
  reserve 15 GB. `lease/doctor` refuses a new lane target when free space falls under 25 GB.
  Deleting `targets/sven` (70 GB, rebuildable) before L1 is the first disk action (v1 6.4).

### 1.1 Research cited by the input: to verify, never facts

None of these has been checked. Each must be looked up (title, year, arXiv id) before it enters
the paper, knowledge or code comments. Some ids may not exist; an id whose month field exceeds
12 or whose date is after today is already suspect.

| Input name | Input id | Claimed use | Status |
|---|---|---|---|
| SWE-smith | 2504.21798 | repository environments and verified synthetic tasks | UNVERIFIED |
| Agentic RL for real-world code repair | 2510.22075 | transfer across environments is hard | UNVERIFIED |
| SWE-Bench Pro Verified | 2609.08149 | leakage, reward hacking, faulty tasks | UNVERIFIED (post-dates most references) |
| Harness-Bench | 2605.27922 | harness configuration matters | UNVERIFIED |
| HARBOR | 2604.20938 | constrained cost-aware harness search | UNVERIFIED |
| Mixture of Self-Improving Branches | 2609.37834 | complementary strategies | UNVERIFIED (id suffix 37834 above the usual five-digit monthly range) |
| forgetting mitigation | 2505.17496 | replay helps | UNVERIFIED |
| Online Process Reward Learning | 2509.19199 | credit assignment beyond terminal | UNVERIFIED |

Package R0 (section 5) does the check. Until then the plan cites none of them as evidence.

---

## 2. What exists, phase by phase (task A)

Paths are relative to each repo root. "Prod" = a production pipeline or CLI verb consumes it.

### 2.1 Phase 0: inventory, reproduction, acceptance

| Item | State | Label | Evidence |
|---|---|---|---|
| `splinter doctor` | exists; preflight verdicts | READ-IN-CODE | `crates/splinter/src/cli.rs:106`; `crates/pipelines/src/doctor.rs`; tests `crates/pipelines/tests/doctor.rs:24-125` |
| Lease kit (`lease/doctor`, `lease/gpu/run`, `slot`, worktrees) | landed | READ-IN-CODE | splinter `17e9202`; `Makefile:174-210`; `scripts/lease/test-lease.py` (12 tests) |
| Cross-repo test | only `test/e2e/brain-sven`: brain serves Qwen3-0.6B on CPU, sven reads a file with a tool | READ-IN-CODE | `Makefile:97-99`; `scripts/e2e/brain-sven.sh:30-88` |
| Experience record + training + deployed adapter in one test | **missing** (searched: `#[ignore]` tests, `test/e2e`, `first_run.rs`, `fake_brain`, paper scripts) | READ-IN-CODE | pipeline tests use `FakeTrainer`, scripted agent, `tests/common/fake_brain.rs`; nearest is the paper's manual `flywheel_cycle.py` (closed-book `learn`, no sven tools) |
| Run record | id, command, args, status, stages (`duration_ms`), outputs, seeds; **no commits, no cost** | READ-IN-CODE | `crates/store/src/runs.rs:67-80,123-150` |
| Release manifest provenance | `brain_commit`, `splinter_commit` fields exist, left empty by standard release | READ-IN-CODE | `crates/orchestrator/src/release_manifest.rs:189-194`; `crates/pipelines/src/release/mod.rs:437-440` |
| Paper verification | `verify_structure.py` reads claims at pinned commits; splinter pin `dcb03816` is on no ref | VERIFIED-BY-RUN | `git merge-base --is-ancestor dcb03816 origin/main` exit 1; `git cat-file -t dcb03816` = commit (local object only) |
| Paper ledger | 168 claims; 232 structure checks | READ-IN-CODE | `papers/splinter-brain-sven-flywheel/data/claims.csv`, `structure_checks.jsonl` |
| Brain scope | names neither splinter nor sven (gate pattern, CHANGELOG, `.gitignore`, a GPT-2 vocab token only) | READ-IN-CODE | brain `scripts/gates/check-repo-scope.sh:30` |
| Sven scope | names splinter only in the gate that forbids it | READ-IN-CODE | sven `scripts/gates/check-repo-scope.sh:22` |

### 2.2 Phase 1: real competence benchmark

| Item | State | Label | Evidence |
|---|---|---|---|
| `practice` stages | policy, snapshot, profile, tasks, reserve, solve, verify, frontier, dataset, train (+ release) | READ-IN-CODE | `crates/pipelines/src/practice/mod.rs:83-90`; `practice/stages.rs:103-120,142-525` |
| Task sources | **seeded mutants only** in prod (`from_mutations`); authored fixtures used by tests only | READ-IN-CODE | `practice/stages.rs:307`; `crates/knowledge/src/tasks/workspace/mutate.rs:129`; `fixtures.rs:89,120,248` |
| Mutation operators | flip comparison, off by one, swap boolean, delete line | READ-IN-CODE | `crates/eval/src/verifiers/mutation.rs:62,88` |
| Admission | held key, kill, split killing tests out, reproduce twice, self-contained, near-dup, hidden kill rate >= 0.5, reverse patch restores | READ-IN-CODE | `mutate.rs:9-32,198-254,265,309-313,345-361,370,429,507`; constants `workspace/mod.rs:75-113` |
| Tasks from history / real issues | **missing** (searched: `git log`, revert, regression, from_commit, historical, issue, SWE-bench) | READ-IN-CODE | only `capture_tree` of one revision, `crates/knowledge/src/capture/tree.rs:180`; `repository-practice.md` item 5 open |
| Rust profile | `CARGO_TEST = cargo test --offline` detected from `Cargo.toml`; **mutation seeding refuses it** (needs a per-test runner, only `python3 -m unittest` is read per test) | READ-IN-CODE | `profile.rs:42,96-97`; `mutate.rs:128-139,407`; knowledge 042 |
| Families | 8 own small projects (5 Python, 3 Lua), 132-186 admitted tasks each, plus `config-discovery` (hidden-world, not in catalog) | READ-IN-CODE | `tasks/catalog.toml:8-153`; `tasks/README.md:102-111` |
| Family unit | the mutated **file** (`family: path`); exam reservation and clustering use it | READ-IN-CODE | `mutate.rs:249`; `release/probe.rs:582-598` |
| Partitions | per-run exam reserved before rollouts; plan generations + sealed families; no dev/qualification/external roles | READ-IN-CODE | `practice/stages.rs:333`; `experiment/mod.rs:39-70,351-372`; `tasks/e3-plan.toml:30-43` |
| Paired seeds | exam seed recorded per run | READ-IN-CODE | `crates/pipelines/tests/practice.rs:351` |
| Cost per attempt | **not stored**: no wall/token/tool/GPU field on Task, Experience or Annotation; `WorkReport.tool_calls` transient; local provider `cost_usd: None` | READ-IN-CODE | `crates/core/src/experience.rs:240-306,493-512`; `crates/agent/src/work.rs:142-154`; `crates/model/src/local/events.rs:44-52` |
| Cost in a sample | `samples/agent/loop` keeps seconds, tokens, tool calls, `Usage{cost_usd, wall_seconds}` | READ-IN-CODE | `samples/agent/loop/src/outcome.rs:56,71-113` |
| Outcome/decision | `Outcome {Pass, Fail, Abstain}`, strength order, `decide`, `reward` | READ-IN-CODE | `crates/core/src/annotation.rs:77,90-125`; `crates/store/src/decision.rs:34-71` |
| Failure class | **missing**; refusal enums exist per layer (generator `Reason`, verifier `Refusal`, probe `Outcome`, serve `Failure`, data `Exclusion`) | READ-IN-CODE | `workspace/mod.rs:118`; `verifiers/workspace.rs:301`; `release/agentic.rs:63`; `release/serve/server.rs:31`; `crates/data/src/lib.rs:290` |
| Frontier, pass@k | `FrontierClass {Always, Never, Frontier, Taught, Unmeasured}` | READ-IN-CODE | `crates/eval/src/frontier.rs:39`; `pipelines/src/curriculum/frontier.rs:70,239` |
| Power, pilot label | exact sign-test power, alpha spending, PILOT under 100 families, Wilson interval | READ-IN-CODE | `crates/eval/src/power.rs:292,306`; `research-upgrades.md` RS-P1 done list |
| Solver | sven `Toolset::coding()` minus 8 withheld tools; visible pass-to-pass checks as follow-ups; tool cap, repetition stop | READ-IN-CODE | `crates/agent/src/work.rs:70-79,184,309-410,448,504`; `pipelines/src/workspace_solve.rs:1-54` |
| Sven benchmark machinery | `sven-ablation` (scripted-model harness-variant matrix), Terminal-Bench adapter, ledger baseline | READ-IN-CODE | sven `crates/ablation/src/lib.rs:9-44`; `Makefile:313-372` (`benchmark/ablation` :370) |

### 2.3 Phase 2: roles, retention, anchors, verifier conformance

| Item | State | Label | Evidence |
|---|---|---|---|
| Model roles | Policy, Teacher, Generator, Planner, Judge, Critic, Router (who plays a role, not what a task set is for) | READ-IN-CODE | `crates/core/src/role.rs:34-48,108` |
| Suite kinds | eval: HeldOut, Retention, Anchor; exam, powered exam, exam set, reserve, practice exam | READ-IN-CODE | `pipelines/src/eval.rs:50-56`; `exam.rs:456`; `practice/exam.rs:123,261` |
| Gate | improvement (clustered sign test), retention (point drop, **not clustered, no interval**), anchor (rate 0.02), serve (75% agreement) | READ-IN-CODE | `crates/eval/src/gate.rs:93-99,559-576,605,656,694-698` |
| Retention source | every earlier release's held-out and practice exam; plan `retention` lists validated but **not consumed** | READ-IN-CODE | `release/gating.rs:243-262,471-499`; `experiment/mod.rs:200,238,249` |
| Capability ledger, competence | **missing** (searched capability, ledger, competence: only claim ledger, frozen-exam pin ledger, loss ledger, `TrainingCapabilities`) | READ-IN-CODE | `crates/knowledge/src/claims/ledger.rs:106`; `crates/data/src/frozen.rs:57`; `crates/model/src/capabilities.rs:27` |
| Continual matrix | ACC, BWT, Forgetting, Forward Transfer, `eval matrix` | READ-IN-CODE | `continual-learning.md` CL-P1 done list; `cli.rs:663` |
| Anchor | 40 general questions; L0 `s-gate` adds `AnchorBound::NetLost` and `ANCHOR_NET_LOST_FOR_200 = 6` (uncommitted) | READ-IN-CODE | `samples/generations/generations.py:232,262`; worktree `s-gate` diff of `crates/eval/src/gate.rs` |
| Workspace verifier | fail-to-pass, pass-to-pass, hidden overlay, protected/hidden paths fenced; **pass = exit 0, no signal, no timeout; does not prove which tests ran** | READ-IN-CODE + VERIFIED-BY-RUN (read) | `crates/eval/src/verifiers/workspace.rs:9-44,90-117,421-449,600-604` |
| Canaries | inverted checks; runner tampering beats a plain exit-code grade (test); `release --canaries`; **`practice` never passes canaries** | READ-IN-CODE | `verifiers/workspace_canary.rs:63`; `crates/eval/tests/workspace_verifier.rs:474`; `release/mod.rs:121`; `practice/stages.rs:525` |
| Exposure audit | hidden path read, 12-token shingles, forbidden strings; Constraint verdict | READ-IN-CODE | `crates/eval/src/exposure.rs:1-24,52,212,266`; `pipelines/src/workspace_verify.rs:41-61` |
| Hidden-suite strength | kill rate on nearby mutants, `WeakHiddenTests` below 0.5 | READ-IN-CODE | `mutate.rs:225-231,265`; finding 048 |
| Verifier conformance suite | **missing** as a suite; individual tests cover deleting a failing test, canary tampering, exposure | READ-IN-CODE | `crates/eval/tests/workspace_verifier.rs:214-508`; `exposure_audit.rs:77-114` |
| Sealing | sealed families refused only under `practice --plan` | READ-IN-CODE | `practice/stages.rs:162-168`; `experiment/mod.rs:368-370` |

### 2.4 Phase 3: failure taxonomy and interventions

| Item | State | Label | Evidence |
|---|---|---|---|
| ATIF trajectory | `Trajectory`, `TraceStep` (`model_name` never set), `StepMetrics` (token ids/logprobs never filled), `FinalMetrics` | READ-IN-CODE | sven `crates/atif/src/model/trajectory.rs:14,44-98`; `step.rs:40-86`; `metrics.rs:11-35` |
| Gate and memory records in ATIF | `GateDecision`/`GateRecord` (`sven.gate.v1`), `MemoryRetrieval`; wired through `CallRecords` (CI) and SDK capture; **only asked-about calls get a gate record** | READ-IN-CODE | sven `crates/atif/src/model/sven_records.rs:59-220`; `crates/ci/src/call_records.rs:24-63`; `crates/sdk/tests/call_records.rs:208` |
| HSM audit | `AuditRecord` per dispatch; `ToolAuditRecord` Started/Denied/ApprovalRequired; **Succeeded and Failed never constructed**; hash-chained `.sven/audit.jsonl` | READ-IN-CODE | sven `crates/hsm/src/audit.rs:21-286`; `crates/kernel/src/lib.rs:534-667`; `crates/executors/src/audit.rs:38` |
| Tool timing | **missing** (`ToolCallFinished` has no duration; searched duration_ms, elapsed_ms, started_at, finished_at) | READ-IN-CODE | sven `crates/vocab/src/lib.rs:637` |
| Run usage | `RunOutcome.usage` = input/output tokens only; `OutcomeFold` counts tool calls privately; SDK trajectory has no per-step metrics, no `final_metrics` | READ-IN-CODE | sven `crates/sdk/src/run.rs:144,250`; `crates/session-model/src/outcome.rs:48,70`; `crates/sdk/src/agent.rs:115-139` |
| Run conclusion | Success, AgentError, Cancelled, Timeout, BudgetExhausted, Waiting | READ-IN-CODE | sven `crates/session-model/src/outcome.rs:48` |
| Retrieval | semantic memory lexical (FTS5 + bm25), no embeddings; withheld from the workspace solver | READ-IN-CODE | sven `crates/memory/src/tool.rs:32-45,235`; `crates/memory-sqlite/src/ops/retrieve.rs:75-103`; splinter `work.rs:70-79` |
| Critic | closed-book critique of a failed attempt; used by `learn`, not `practice` | READ-IN-CODE | `crates/agent/src/critic.rs:89`; `pipelines/src/learn/stages.rs:611` |

### 2.5 Phase 4: training contract, SFT, 27B

| Item | State | Label | Evidence |
|---|---|---|---|
| Action-level SFT view | one record per action; feedback turns unsupervised context; tool observations masked (`train=false`); tool schemas kept; compaction/markup refused; over-block counted | READ-IN-CODE | `crates/data/src/views/sft_step.rs:4-47`; `trajectory.rs:4-40,244,272,313`; `lib.rs:290,511-521` |
| Other views | sft-final, critic, preference, verifier, decision, retrieval, outcome, denoise, cpt, voice, rehearsal | READ-IN-CODE | `pipelines/src/datasets.rs:54-83` |
| Dataset dedup | exact only at data level; near-dup at task level | READ-IN-CODE | `corpus.rs:103`; `knowledge/src/tasks/dedup.rs` |
| Practice dataset filter | `min_strength = Executable`, `assert_no_exam_family` | READ-IN-CODE | `practice/stages.rs:438,467,470` |
| Brain chat masking | template rendered once, each message masked by its `train` flag; `train.mask.bin` | READ-IN-CODE | brain `crates/data/src/chat.rs:317-345,586-614`; `crates/model/src/train.rs:289-295` |
| Chat template | `ChatFormat::for_checkpoint`; flavour hard-coded Qwen3 at render; cross-check tests skip without a checkpoint | READ-IN-CODE | brain `crates/qwen3/src/chat.rs:430-479`; `crates/data/tests/chat_template_cross_check.rs:102-312` |
| SDK SFT/DPO/full FT | qwen3 only; qwen35 refused | READ-IN-CODE | brain `crates/sdk/src/finetune.rs:437-439`; `preference.rs:274-348`; `study.rs:113,136,843-851` |
| Default LoRA targets | SDK trains all 7 linears; `brain serve` bank defaults to attention only | READ-IN-CODE | brain `crates/qwen3/src/finetune.rs:461,614`; `crates/sdk/src/finetune.rs:536`; `crates/catalog/src/adapter_registry.rs:82-86` |
| Adapter card | `kind, rank, base, alpha, targets, dataset_id, per_target`; provenance with `base_digest` | READ-IN-CODE | brain `crates/checkpoint/src/st.rs:22-61,78-201` |
| Card targets validated | **never** at read; serve checks tensor names in the file | READ-IN-CODE | brain `crates/model/src/lora.rs:1194-1314`; `adapter_registry.rs:256-271`; `qwen3/src/serve/adapters.rs:385-405` |
| `base_digest` | recorded by SDK SFT/DPO/full FT; **not** by qwen35 `finetune_streamed`, qwen3 CLI finetune, `Improve` | READ-IN-CODE | brain `sdk/src/finetune.rs:571,608`; `qwen35/src/finetune.rs:250-252`; `cli/src/qwen_cli.rs:1056`; `rl/src/improve.rs:485` |
| Adapter dtype, tokenizer digest | **not recorded**; `WeightsIdentity` hashes weights only | READ-IN-CODE | brain `checkpoint/src/st.rs:418-422`; `sdk/src/chat.rs:427-468` |
| AdamW on a skipped step | bias correction counts the skipped step | READ-IN-CODE | brain `crates/model/src/train.rs:1159-1170` |
| Held-out selection | `FitOpts` patience, `kept_best`, `score_records` | READ-IN-CODE | brain `model/src/train.rs:33,80,957-976,1304-1335`; `sdk/src/finetune.rs:540-543,616-617` |
| qwen35 decode with LoRA | works on fp32 through shared layer functions; narrow-base decode fixed in L0 `4137781d` (unpushed) with `tests/lora_decode.rs` | READ-IN-CODE | brain commit `4137781d` message and stat |
| qwen35 KV sampler | `AdapterSampler::greedy` still full-forward | READ-IN-CODE | brain `crates/qwen35/src/adapter_sample.rs:10-18,89` |
| qwen35 serve adapters | **none**; `adapter_release::read` refuses family != "qwen" | READ-IN-CODE | brain `crates/qwen35/src/serve.rs:136-146`; `crates/catalog/src/adapter_release.rs:149-150` |
| qwen35 in-process chat | `TextGenerationPipeline` engine holds `qwen3::Qwen` | READ-IN-CODE | brain `crates/sdk/src/text.rs:305-326` |
| Arch of a base dir | `modelstore::declared_architecture`, `family_of_config`; loader returns no arch; splinter never sets `.arch(` | READ-IN-CODE | brain `crates/modelstore/src/plan.rs:326,362`; splinter grep of `\.arch(`: none |
| Kernel auto-selection | WGSL upgrade and CUDA-native tier fall back per dispatch outside served shapes; `--no-native-kernels`; `BRAIN_NO_KERNEL_UPGRADE` | READ-IN-CODE | brain `crates/gpu-core/src/upgrade.rs:30-44,303`; `native_upgrade.rs:24-36,966`; `backend-api/src/select.rs:781,1409` |
| Throughput make targets | `qwen35/train/probe`, `qwen35/train/step` (T0) | READ-IN-CODE | brain `Makefile:1452-1458` at `8b652351` |

### 2.6 Phase 5: harness configuration

| Item | State | Label | Evidence |
|---|---|---|---|
| Config loader | YAML, serde, unknown-key warnings only, no range checks; golden fixture | READ-IN-CODE | sven `crates/config/src/document.rs:60-165`; `crates/bootstrap/src/config/mod.rs:49-124` |
| Context | `compaction_threshold` 0.85, `compaction_keep_recent`, `compaction_strategy`, `tool_result_token_cap`, `compaction_overhead_reserve`; assembly `mode`, `budget_tokens`, `keep_recent_rounds` | READ-IN-CODE | sven `crates/bootstrap/src/config/agent.rs:63-103`; `config/context.rs:22-71` |
| Compaction off switch | **none** (threshold above 1 suppresses it) | READ-IN-CODE | sven `crates/executors/src/turn.rs:140-171,383-398` |
| Turn limits | `max_tool_rounds` 200 (config), timeouts | READ-IN-CODE | sven `config/agent.rs:57,111-175` |
| Delegation | `delegation.policy` Off/Advise/Enforce (default Off = always launch); `task` tool; team tools exported, **not registered** | READ-IN-CODE | sven `config/delegation.rs:19-58`; `bootstrap/src/registry.rs:421,480`; `crates/team/src/lib.rs:34-41` |
| Decision guidance | `decision.enabled` (default false) and per-feature flags | READ-IN-CODE | sven `crates/decision/src/config.rs:102-231` |
| Planning, search-before-edit, parallel reads, memory top_k | **not config**: planning is a mode/SDLC state; batched calls always parallel; recall `limit` per call (default 10, cap 50) | READ-IN-CODE | sven `crates/vocab/src/lib.rs:385`; `crates/executors/src/tool.rs:1-13`; `crates/memory/src/tool.rs:235`; `crates/memory-api/src/access.rs:36-43` |
| Hard-coded | `MAX_RECOVERY = 3`, stall thresholds | READ-IN-CODE | sven `crates/machines/src/machines/sdlc/mod.rs:66`; `loop_core.rs:192,201` |
| Workflows | markdown workflows (`sven --file`), skills, YAML `HarnessSpec` with `LimitsSpec` | READ-IN-CODE | sven `crates/session-store/src/markdown.rs:45`; `crates/harness-spec/src/spec.rs:207,244` |
| Solver protocol knobs in splinter | `WorkProtocol{follow_ups, max_tool_calls}`; agentic defaults: attempts 1, follow-ups 2, tool calls 60, repeats 3 | READ-IN-CODE | splinter `crates/agent/src/work.rs:448-455`; `crates/eval/src/gate/agentic.rs:17-27` |

### 2.7 Phases 6-7

| Item | State | Label | Evidence |
|---|---|---|---|
| GRPO in brain | `GrpoOptions`, `with_reference`, `with_rollout`; `Improve` accepts qwen35; tests in `make test/rl` | READ-IN-CODE | brain `crates/rl/src/objective/grpo.rs:88-242`; `sdk/src/study.rs:1350-1406`; `Makefile:440-450` |
| Rollout for qwen35 | **none** (only `qwen3::rollout::EngineRollout`) | READ-IN-CODE | brain `crates/qwen3/src/rollout.rs:83`; `model/src/rollout.rs:89-100` |
| Logprobs from serving | refused by name | READ-IN-CODE | brain `crates/apiserve/src/sampling.rs:57-80` |
| Sven sampling seam | no sampling fields, no token events; L0 `v-traj` is adding them (uncommitted, 12 files) | READ-IN-CODE | sven `crates/model/src/types.rs:505-595`; worktree `v-traj` diff |
| Splinter GRPO | re-exported, only `samples/adams` uses it | READ-IN-CODE | splinter `crates/model/src/rl.rs:20-23`; `samples/adams` |
| Preference (DPO) | splinter `train_preference`; brain DPO qwen3 | READ-IN-CODE | splinter `crates/model/src/train.rs:578`; brain `sdk/src/preference.rs:274` |
| Generations, consolidation | `absorb`, `consolidate`, `rehearse`, replay 0.25 | READ-IN-CODE | splinter `pipelines/src/absorb/mod.rs:300`; `absorb/consolidate/mod.rs:230`; `rehearsal.rs:66,131`; `train/mod.rs:120-126` |

### 2.8 Discrepancy report seed (input P0.2)

Classes: (a) historical defect already fixed; (b) currently reproducible defect; (c) evaluation
or architectural flaw that remains when code behaves as written. P0-INV (section 5) completes
it with runs; these are the rows already established.

| # | Item | Class | Evidence |
|---|---|---|---|
| D1 | Holdout fallback divided one group | (a) | v1 1.2; `2df4237` |
| D2 | v1 F9 "no LoRA in qwen35 decode" | plan error; real defect was narrow-base `tok.weight` lookup, (a) once `4137781d` is pushed | brain `4137781d` |
| D3 | Paper splinter pin `dcb03816` unreachable | (b) | VERIFIED-BY-RUN above |
| D4 | Retention holds a facts candidate to never-taught held-out facts (knowledge 061) | (c) | `release/gating.rs:243-262` |
| D5 | Anchor 40 at rate 0.02 refuses any lost answer | (c) | `gate.rs:99`; v1 F24 |
| D6 | SDK trains 7 targets, `brain serve` banks attention by default | (b) | brain `adapter_registry.rs:86` vs `qwen3/src/finetune.rs:461` |
| D7 | Card `targets` never validated against the tensors | (b) | brain `model/src/lora.rs:1260-1314` |
| D8 | qwen35 streamed trainer, qwen3 CLI and `Improve` write no `base_digest` | (b) | section 2.5 |
| D9 | No tokenizer digest in model identity | (b) | brain `sdk/src/chat.rs:427-468` |
| D10 | AdamW bias correction counts skipped steps | (b) | brain `model/src/train.rs:1168` |
| D11 | Workspace pass = exit status only | (c) | splinter `verifiers/workspace.rs:602-604`; tampering test `:474` |
| D12 | `practice` gate runs without canaries | (c) | `practice/stages.rs:525` |
| D13 | Family = one mutated file; clusters are files, not subsystems | (c) | `mutate.rs:249`; `probe.rs:582-598` |
| D14 | Retention is a point bound, not clustered, no interval | (c) | `gate.rs:605`; `gating.rs:471-499` |
| D15 | Plan `retention` lists validated but unused | (c) | `experiment/mod.rs:200-249` |
| D16 | Release manifest commits empty; run record has no commits or cost | (b) | `release_manifest.rs:189-194`; `runs.rs:123-150` |
| D17 | Sven tool audit never records Succeeded/Failed or timing | (b) | sven `hsm/src/audit.rs:156-255` |
| D18 | Sven SDK trajectory has no metrics; `TraceStep.model_name` never set | (b) | sven `sdk/src/agent.rs:115-139` |
| D19 | All 8 family funnels are mutation-only; three predate `over_time_budget` | (c) | v1 F27 |
| D20 | `~/code/sven` checkout 16 commits behind origin | ops | VERIFIED-BY-RUN `git log` |
| D21 | `tasks/README.md:12-19` describes `reference/` and `negative/` directories that do not exist | (b) docs | splinter sweep |
| D22 | Sven `Feature::decision_point` says only ToolRouting has a decision point; SkillSelection and Delegation are consumed too | (b) docs | sven `crates/decision/src/config.rs:141-145` |

---

## 3. v1 packages remapped (task B)

| v1 | v2 phase | Action | Reason |
|---|---|---|---|
| OPS | done | keep | landed `17e9202` |
| T0 | done | keep | landed brain `8b652351` |
| G1-G6 (brain gates, scrub, licences, stale rows) | P0 | keep (L0 `b-gate` landing) | G0 source integrity |
| B1 LoRA in qwen35 decode | P4.5 | keep, rescoped: the fix is the narrow-base embed (`4137781d`); push it | v1 F9 wrong (D2) |
| B2 KV-cached `AdapterSampler` | P4.4 stage 6 | keep | base vs adapted generation at usable speed |
| B4 SDK chat SFT + DPO for qwen35 | P4.4 stage 3 | keep; DPO half deferred to P6 | SFT first; preference is Phase 6 |
| B5 qwen35 in-process chat | P4.4 stage 7 | keep, after B3 | serve path is the deployed one; in-process is the serve check's reference |
| B3 qwen35 serve + bank | P4.4 stages 8-9 | keep; merge with B9's target contract; decision D-BANK: extract the generic adapter bank (lifecycle, slots, upload quantum, draining, namespaces) from `brain crates/qwen3/src/serve/adapters.rs` into `model`, driven by `LoraSites`; targets become site leaf names checked against the model's sites (replaces catalog's qwen3 bitmask `AdapterTargets`); per-model overrides via a trait with defaults (target groups, rank cap, slots) plus one decode hook; qwen3 moves first with `engine_adapter_bank.rs` and `multi_adapter.rs` unchanged as the regression proof, then qwen35 adopts it | I3 deployment equivalence; one bank for every model instead of a second per-model copy |
| B6 logprobs + token ids | split | B6a (seed echo, served identity) -> P1; B6b (logprobs, token ids) -> P6 | seeds are needed for paired baselines; logprobs only for RL |
| B7 GRPO remainder | split | AdamW index -> P4 (B10); entropy, batched KV rollouts, C8 -> P6 | the index bug hits SFT |
| B8 powered GRPO ablation | P6 | re-sequence behind G9 | input Phase 6 |
| T1-T5 throughput | parallel lane | keep, GPU windows only; never blocks P0-P2 | lowers P4/P6 cost (section 6.3) |
| V1 sven sampling seam | P1 | keep (L0 landing) | paired seeds on served models; sampling recorded |
| V2 ATIF fill, append-only | P3 | keep | failure taxonomy needs complete, crash-safe trajectories |
| V3 SDK trajectory metrics | P1 | merge into V4 | cost metric C |
| S1 arch from card + budgets | P4 | keep | 27B through splinter |
| S2 tiny-qwen35 e2e | P0/P4 | merge into P0-ACC as its qwen35 variant (P0-ACC35) | one acceptance test, two arches |
| S4 anchor 200 + bound in questions | P2 | keep the `NetLost` bound (L0); anchor-200 becomes one stratum of P2-ANCH | small anchor = smoke test only (input P2.3) |
| S5 retention on taught facts | P2 | merge into P2-LEDGER (retention exams on demonstrated capabilities) | facts campaign leaves the primary pipeline |
| S6 group preflight, monitor degrade | P0 | keep (L0 landing) | correctness |
| S7 sealing everywhere, plan retention consumed | P2 | keep as P2-PART | G3 contamination |
| S8 GRPO stage phase 1 | P6 | re-sequence behind G9 | input Phase 6 |
| S8a E5 prereg (8 families) | P1/P7 | drop as written; its families feed P1 dev partition, its generations feed E11 | E5's question is superseded by E1/E4/E11 |
| S9 GRPO phase 2 (sven trajectories) | P6 | re-sequence | needs B6b, V2 |
| P1 paper re-pin | P0 | keep | D3 |
| P2 check kinds (`count`, `test`) | P0 | keep, low priority | historical-record honesty |
| P3 `verify` in `make check` | P0 | keep | G0 |
| P4 run records carry commits, dirty flag, binary digest, GPU contexts | P0 | merge into P0-PROV (in splinter's run record, not a paper script) | D16; every experiment needs it |
| P5 descriptive title | P0 | keep | no two releases exist |
| R1 Q6 rerun (facts on 27B) | - | drop | invented facts are not the objective; P4.4 stages 3-9 on a coding dataset replace it |
| R2 registered 27B facts turn | - | drop; replaced by E6 then E11 | same |
| research-upgrades RS-P2 canaries | P2 | wire into practice and benchmark (P2-CONF) | D12 |
| RS-P3 test strength | P2 | keep; differential check of passing patches in P2-CONF | verifier FP rate |
| RS-P5 SFT filters | P4 | arm of E5 dataset ablation | input P4.3 |
| RS-P8 generator diversity | P1 | realised by P1-HIST, P1-AUTH | mutation must not be the only source |
| RS-P9 higher-order tasks | backlog | defer | after E2 shows task quality |
| RS-P10 contamination probe | P1 | keep in P1-PART for the external partition | leakage |
| RS-P12 best-of-k | P4 | arm (rejection-sampled SFT) of E5 | input P4.3 |
| RS-P14 recipe sweep | P4 | pilot inside E4 (rank, LR, 4-bit vs bf16) | pilots choose steps |
| flywheel 2 contracts/provenance | P0 | P0-PROV | |
| flywheel 3 isolation | P2/G14 | decision D-V2-3 (isolation `none`, reported) | K6 |
| flywheel 6 efficiency/safety in gate | P1/P2 | P1-OUT supplies cost; P2-ROLES adds the checks | metric C |
| flywheel 7 pipeline resume | P1 | P1-RESUME (needed by long baselines crossing a lease) | G14 |
| flywheel 8 doctor + acceptance | P0 | P0-ACC | |
| continual CL-P1..P7 | P7 | keep; CL-P5 families are P1-AUTH | |

---

## 4. Dependency graph (v2)

```text
P0-INV --------------------------------------------------------------> P0 exit report (G0, G1)
P0-PROV (commits/cost in run record) --+
P0-ACC (0.6B CPU turn) ----------------+--> G0/G1 baseline report --> no new framework before it
V1 (seam) -> B6a (seed echo) ----------+
P1-OUT (TaskOutcome) <- V4 (sven usage/tool audit/timing)
P1-RUST (per-test cargo runner) --> P1-HIST (history tasks) --+
P1-AUTH (authored Rust families) ------------------------------+--> P1-PART (dev/qual/external) --> P1-BENCH (verb)
P2-VER (tests-ran proof) + P2-CONF (conformance, canaries) ------------------------------------+
P1-RESUME ------------------------------------------------------------------------------------+--> E1 baseline (G1/G2/G4) [GPU]
E1 --> P3-TAX (failure taxonomy, needs V2) --> E3 (non-training interventions) --+
E1 --> P2-ROLES, P2-LEDGER, P2-ANCH -----------------------------------------------+--> Phase 2 exit (5 tests)
B9 (adapter contract) + B10 (AdamW index) --> E4 SFT on Qwen3-8B --> E5 ablation --> G9 (8B)
B1 -> B2, B4 -> S1 -> B3 (+B9) -> B5 -> P0-ACC35 -> E6 27B qualification (G7, G8) --> G9 (27B)
G9 --> Phase 5 (V5 harness profile, E7) --> Phase 6 (B6b, S8, S9, B7 rest, B8; E8, E9) --> Phase 7 (E10, E11)
T1 -> T2 -> (T3, T4) -> T5   (GPU windows only; feeds E6 cost and Phase 6)
```

Serialisation points (in addition to v1 6.2):
- Splinter `crates/core` gains `FailureClass` and the partition role in one lane (`s-bench`),
  because both change canonical forms (`store/tests/identity.rs` golden test).
- Splinter `crates/eval/src/verifiers/workspace.rs` is touched by P2-VER only.
- Sven `crates/sdk/src/{run.rs,agent.rs}` is touched by V4 then V2/V3; one lane (`v-traj`).
- Brain `crates/catalog/src/adapter_registry.rs` and `model/src/lora.rs`: B9 before B3.

---

## 5. Work packages (task C)

Conventions (unchanged from v1 section 4): lanes in detached worktrees under
`~/project/shared/containers/worktrees/<lane>`, own target directories, at most 2 cargo builds
(`scripts/lease/slot`), `CARGO_INCREMENTAL=0`, GPU work only under `make lease/gpu/run`. Every
package's TDD tests are written first and run red; the commit message names the red run. Brain
never names splinter or sven, sven never names splinter; paths reach them as flags.

GPU column: **none**; **S** = shared short (under the lock, minutes); **X** = exclusive.

### Phase 0

**P0-INV. Capability inventory and discrepancy report** (splinter, lane `s-inv`, 6 h, none)
- Output `.agents/roadmap/inventory.md`: one row per crate and per CLI verb / SDK entry of the
  three repos that the target pipeline could touch (splinter learn, practice, absorb,
  consolidate, rehearse, exam, release, tasks, dataset views; sven HSM, tools, agent, team,
  memory, delegation, compaction, decision; brain training, model store, adapters, serving):
  interface, implementation and tests (file:line), production consumers (searched in all three
  repos), used by the Phase 1-4 path (yes/no), runtime validation status, resource cost,
  overlap, recommendation (retain/repair/merge/isolate/delete). Seeded from section 2.
- Completes section 2.8 with runs: `make -C papers/splinter-brain-sven-flywheel verify` at the
  pins and at `--latest`; `make test` in each repo at the heads; `make test/e2e/brain-sven`;
  `make test/families`. Each failure is classified (a)/(b)/(c).
- TDD: not code. Acceptance: every row has a consumer search command recorded; no row says
  "unused" without one.
- Deps: none. First command: `make -C ~/code/splinter lease/doctor`.

**P0-PROV. Provenance in every run and release** (splinter, lane `s-p0`, 6 h, none)
- Files: `crates/store/src/runs.rs` (`Run` gains `revisions {splinter, sven, brain}`, `dirty`,
  `binary_digest`, `gpu_contexts`, skip-if-absent so addresses of old runs do not change);
  `crates/orchestrator/src/runs.rs` fills them (`Config` reads the revisions as values handed in
  by the CLI from `CARGO_PKG_*` build-time info and the lock); `release/mod.rs:437-440` fills
  `brain_commit`/`splinter_commit` and adds `sven_commit`.
- The commits are those the binary was built from: the splinter build script embeds `git
  rev-parse HEAD` and the locked sven/brain revisions from `Cargo.lock` (no runtime git call).
- TDD: `crates/store/tests/run_log.rs` gains a case reading a pre-change run (fields absent,
  digest unchanged) and a new run (fields present); `crates/pipelines/tests/release_workspace.rs`
  asserts the manifest names three commits. Red today.
- Gate: `make build test check`.

**P0-ACC. Cross-repo acceptance test, qwen3** (splinter, lane `s-p0`, 10 h, S 20 min)
- New `make test/e2e/turn` (beside `test/e2e/brain-sven`, same exit-3 convention):
  1. `brain doctor` admits Qwen3-0.6B;
  2. `splinter practice` on one tiny Python family (a new 6-test fixture under
     `tasks/acceptance-tiny`, not a catalog family) with `--max-tasks 6`, policy
     `local:Qwen3-0.6B`: snapshot, profile, tasks, reserve, solve through sven (at least one tool
     call recorded in the trajectory), verify;
  3. the experience store holds at least 6 experiences with workspace verdicts;
  4. `splinter train` 20 steps on whatever records verified (if none, on the fixture's
     reference trajectories admitted as teacher-origin, marked as such);
  5. the release gate runs and its decision is printed whatever it is (the gate is never
     bypassed); the candidate adapter is then served by `brain serve
     --adapter-manifest` (via a test-only alias pointing at the candidate) and sven solves one
     task through it, the served model id naming the adapter digest.
- Asserts stages and artifacts, not quality. CPU backend allowed (minutes to tens of minutes);
  the GPU variant runs under `lease/gpu/run` in 20 minutes.
- TDD: the script's own spec `scripts/e2e/test-turn.py` with a fake `brain`/`sven` on `PATH`
  proves each missing artifact fails the run with its name.
- Deps: P0-PROV (so the run report names revisions). Closes flywheel item 8 for qwen3.

**P0-ACC35. Acceptance test, qwen35 variant** (splinter, lane `s-arch`, 4 h, S 30 min)
- Same script with a random-init tiny qwen35 written by brain (v1 S2). Deps: B4, S1, B3.

**P0-REPORT. Baseline report** (integrator, 2 h, none)
- `campaign/report/baseline`: revisions, test results per repo, model artifacts and digests, disk,
  memory cap, GPU contexts; JSON + Markdown under `runs/<run-id>`. Exit gate of Phase 0.

**R0. Citation check** (splinter, lane `s-inv`, 2 h, none, network)
- Look up each id of section 1.1 on arXiv; record title, year, existence. Only checked entries may
  enter `references.bib`. Output appended to `.agents/research/` as a dated note.

### Phase 1

**V4. Sven cost and tool telemetry** (sven, lane `v-traj` after V1, 12 h, none)
- `RunOutcome.usage` gains cached tokens, tool calls, tool errors, turns, wall time (the
  `OutcomeFold` counts exist: `crates/session-model/src/outcome.rs:70`).
- `ToolAuditOutcome::{Succeeded, Failed}` emitted with `duration_ms` on every finished call
  (`crates/kernel/src/lib.rs:601-667`); `SessionEvent::ToolCallFinished` gains `duration_ms`.
- `Agent::trajectory()` fills per-step `StepMetrics` and `final_metrics`, and sets
  `TraceStep.model_name` (absorbs v1 V3).
- TDD: `crates/sdk/tests/trajectory_metrics.rs` (mock model: totals equal the sum of steps;
  model name present); `crates/kernel/tests/runtime.rs` case: a failing tool yields
  Started then Failed with a duration. Red today.
- Gate: `make test check`. No splinter name anywhere.

**B6a. Seed echo and served identity** (brain, lane `b-srv`, 4 h, none)
- `apiserve` echoes the `seed` used and the served model id (adapter digest when an adapter
  answers) in an extension field; D-Bus `generate` returns the same under `out_meta`.
  `logprobs` stays refused until B6b.
- TDD: `crates/apiserve/tests/api.rs`: a seeded request repeated gives identical text; the reply
  names the seed and, on an adapter model, its digest.

**P1-OUT. TaskOutcome, reconciled** (splinter, lane `s-bench`, 12 h, none)
- Not a new experience format. `TaskOutcome` is a derived evaluation (expdb keeps evaluations
  apart from raw experience) computed from the experience, its annotations, its trajectory's
  metrics and the attempt's report:
  `{task_id, experience_id, partition, verified_success, verification_strength (Strength),
  wall_seconds, gpu_seconds?, prompt_tokens, completion_tokens, tool_calls, tool_errors,
  attempted_actions, unauthorised_actions, submitted, failure: Option<FailureClass>,
  human_minutes?}`. Absent values stay absent (`Option`), never 0 (AGENTS.md).
- `gpu_seconds`: device-busy seconds of the attempt from the local provider when in-process;
  absent for a served policy unless brain reports it (not in scope).
- `human_minutes`: absent unless a person-in-the-loop run records it.
- `FailureClass` enum in `splinter-core` (Phase 3 fills it; P1 defines the variants of the input
  P3.1 list plus `environment` and `unclassified`).
- Persisting the attempt facts that are transient today (`WorkReport.tool_calls`,
  `stopped_by_tool_cap`, `stopped_by_repetition`, wall time): an annotation `attempt-report` on
  the experience (experiences stay immutable).
- Metrics in `splinter-eval::competence`: Q = verified / attempted, C = total cost / verified,
  A = verified-without-human / attempted, each with family-cluster bootstrap intervals and the
  raw counts; no scalar collapse.
- TDD: `crates/eval/tests/competence.rs` (hand-computed Q/C/A with absent costs; an
  environment failure excluded from the denominator and counted); `crates/store/tests/identity.rs`
  golden test unchanged for existing experiences.
- Deps: V4 for token/tool totals from served runs (works on mock trajectories before it lands).

**P1-RUST. Per-test cargo runner** (splinter, lane `s-bench`, 10 h, none)
- A per-test reader for `cargo test --offline` (libtest's `test <name> ... ok|FAILED` lines,
  `-- --exact <name>` for single tests, `--no-fail-fast`), beside the unittest reader
  (`mutate.rs:407`), so mutation seeding and P2-VER work for Rust.
- Build cost: a task's tree gets a pre-built `target` dir from its snapshot (built once by
  `profile`, content-addressed by snapshot digest, kept under the state root, garbage-collected
  with the snapshot). DERIVED need: without it every attempt rebuilds the crate.
- TDD: `crates/knowledge/tests/workspace_tasks.rs` gains a tiny Rust crate fixture: profile
  qualifies it twice; a seeded fault is caught by exactly the named tests; a test the run did
  not execute is reported missing. Red today (`mutation-seeded tasks need a test runner read per
  test`).

**P1-HIST. Tasks from repository history** (splinter, lane `s-bench`, 16 h, none)
- `repository-practice.md` item 5 remainder: for each commit that changes source files and test
  files together, the task is the parent tree with the source change reversed and the commit's
  new or changed tests as hidden fail-to-pass checks; the rest of the suite as pass-to-pass.
  Admission is the existing chain (kill, reproduce twice, self-contained, near-dup, hidden
  strength, reverse patch restores) via a `HistoryGenerator` beside `from_mutations`.
- Request text: built by code from the test names and failure output of the hidden tests, never
  the diff; a model-written symptom description is a later option, judged by the
  `helper-restates-the-answer` rule (knowledge 005).
- Sources for Phase 1 (decision D-V2-5): Rust leaf crates whose suites run in seconds offline:
  splinter `crates/expdb` (no internal dependencies), sven `crates/atif`, `crates/hsm`,
  `crates/config`, brain leaf crates chosen by P0-INV with a suite under 60 s on CPU. Each
  source is captured with `capture_tree` at a pinned revision.
- TDD: `crates/knowledge/tests/history_tasks.rs`: a fixture repo with three commits (one fix
  with test, one refactor without tests, one test-only) yields exactly one admitted task whose
  hidden test fails on the reversed tree and passes on the commit. Red today.

**P1-AUTH. Authored change families** (splinter, lane `s-tasks`, 16 h, none)
- Four small own Rust projects under `tasks/` (own code, Apache-2.0 like the catalog), each with
  authored task kinds the input P1.3 lists that mutation cannot produce: tests-first features,
  API migrations, comprehension-plus-change, multi-module integration faults, performance tasks
  with a threshold, environment/config failures, multi-step debugging with controlled logs.
  Authored by a lane that never sees the qualification or external partitions.
- Each family passes `scripts/tasks/families.py check` (five green baselines) and the existing
  audit (witness solves, negatives rejected). Creates the `reference/` and `negative/`
  directories `tasks/README.md` promises (D21).
- TDD: `families.py` check is the test; plus one negative per task kind.

**P1-PART. Partitions and roles in experiment metadata** (splinter, lane `s-bench`, 10 h, none)
- `ExperimentPlan` (`experiment/mod.rs:61`) gains `partition` blocks: `development`,
  `qualification` (protected), `external` (final), each a list of families or sources with tree
  ids. Family unit for real repos: the subsystem (crate or top-level module), not the file
  (D13); the file stays the unit inside the 8 Python/Lua families.
- Sealing (v1 S7 merged): qualification and external tree ids written to the state root by
  `experiment check`; every stage that reads a source (learn, practice, generator, teacher,
  rehearsal) refuses them unless the stage runs the qualifying exam itself.
- Contamination probe (RS-P10) per external family, reported.
- TDD: `crates/pipelines/tests/sealed_everywhere.rs` (v1 S7), `partition_roles.rs`: a
  development-only run cannot read a qualification family; a plan whose partitions overlap is
  refused. Red today.

**P1-RESUME. Pipeline resume** (splinter, lane `s-p0` after P0-ACC, 12 h, none)
- Flywheel item 7: stage outputs content-addressed; `practice --resume <run>` and the
  benchmark verb skip finished stages and finished attempts (append-only job log exists in
  `splinter-store`). Run liveness: a died process is reported `interrupted`, not cancelled.
- TDD: kill a scripted benchmark after 3 of 6 attempts; `--resume` runs exactly 3 more and the
  outcome set is identical to an uninterrupted run (same seeds).

**P1-BENCH. `splinter benchmark` verb** (splinter, lane `s-bench`, 10 h, none)
- A pipeline in `crates/pipelines/src/benchmark/` composing the existing practice stage
  functions (shared, not copied): `benchmark prepare --plan <file> --output <dir>` (snapshot,
  profile, tasks from mutation + history + authored, partition assignment, admission report;
  no rollout); `benchmark run --experiment <dir> --partition development --policy <ref>
  --attempts k` (solve, verify, exposure audit, TaskOutcome per attempt; no training);
  `report --experiment <dir>` (Q/C/A with intervals by family and by task kind, failure
  classes, admission funnel). `improve` is defined in Phase 3/5, not here.
- `report` also replaces the missing generic `report` verb (flywheel item 8): it reads run
  records and lineage.
- TDD: `crates/pipelines/tests/benchmark.rs` with the scripted agent: prepare is deterministic
  for a seed; run never trains; a qualification partition cannot be run with `--attempts`
  above the registered value; resume as P1-RESUME.

**P1-THRU. Agent-throughput probe** (integrator, 1 h, S 45 min)
- 10 development tasks each with `remote:` Qwen3-8B (qwen3 paged engine) and the 27B Q8 GGUF
  served base, concurrency 1 and 4. Records wall per task, prompt and completion tokens, peak
  memory. Decides partition sizes and k (OQ-V2-1).

**E1-RUN. Untrained-agent baseline** (integrator, lease L3, X 6-8 h)
- See section 7 (E1). Exit gate of Phase 1.

### Phase 2

**P2-VER. The verifier proves the tests ran** (splinter, lane `s-gate`, 10 h, none)
- `WorkspaceVerifier::grade` (`verifiers/workspace.rs:371`) reads each command per test
  (unittest reader, P1-RUST cargo reader): a pass requires every named fail-to-pass and hidden
  test to be reported as run and passed, the number of tests run to be at least the baseline's,
  the runner to terminate within limits, and the test runner files and harness configuration to
  be byte-identical to the snapshot's (protected by digest, not by path list alone).
- TDD in `crates/eval/tests/workspace_verifier.rs`: (1) a patch that makes the runner print
  "OK" and exit 0 fails; (2) a patch that deletes the hidden test module fails; (3) a patch that
  edits `conftest`/runner config fails; (4) the existing runner-tampering canary case (:474)
  now fails the cheat at grade time, not only via a canary. Red today on (1)-(4).

**P2-CONF. Verifier conformance suite** (splinter, lane `s-gate`, 8 h, none)
- `crates/eval/tests/verifier_conformance.rs`, one table per verifier (workspace, formal,
  executable code, reference-check): correct, incorrect, equivalent representations (the
  thousands separator of knowledge 060), edge cases, missing evidence, early exits,
  hidden-test disclosure attempts, verifier-config tampering, success-log mimicry. Reports
  false-pass and false-fail counts separately.
- Differential check (RS-P3 remainder): for 50 admitted tasks, 10 seeded plausible-but-wrong
  patches each (mutants of the reference); the false-pass rate is recorded per task kind.
- Canaries wired into `practice` and `benchmark run` (D12): every qualification run carries
  `tasks canaries` of its partition.

**P2-ROLES. Typed evaluation roles** (splinter, lane `s-gate`, 8 h, none)
- `EvalRole {Training, Development, Transfer, UnknownControl, Retention, GeneralAnchor,
  Qualification, FinalExternal}` in `splinter-core` with the input's selection/gate rules as
  code (which roles may select checkpoints, which gate a release, which may never be queried
  twice by the same plan). A suite carries its role in immutable experiment metadata; the gate
  refuses a suite whose role does not allow the decision asked of it.
- Frozen-final ledger (flywheel item 6): every query of a `FinalExternal` suite is recorded; a
  second query by a plan that selected on its results is refused.
- TDD: `crates/eval/tests/roles.rs`: a development suite cannot gate a release; an unknown-control
  suite cannot count as retention; a final suite queried twice is refused.

**P2-LEDGER. CapabilityLedger** (splinter, lane `s-gate`, 12 h, none)
- `Capability {id, verifier_kind, family, acquired_by: release, evidence_families,
  competence (rate + interval), retention_exam}` in `splinter-store` (state) and
  `splinter-eval` (decision). A capability is recorded only when demonstrated on tasks the
  training run never saw (development role) with at least the pre-registered count; fewer is
  `provisional`.
- Retention exam = freshly generated tasks of the same family (new mutants or history commits
  not in any training set, held keys deduplicated), never replayed training items; supersedes v1
  S5 and fixes D4 and D15 for workspace families.
- Retention decision: family-clustered paired comparison with a non-inferiority margin and an
  interval (fixes D14); the champion's profile is frozen before each generation.
- TDD: `crates/pipelines/tests/capability_ledger.rs` (scripted): a candidate that forgets an
  acquired capability fails retention; a capability the champion never demonstrated is not in
  the profile; unknown-control tasks do not count.

**P2-ANCH. Stratified anchors** (splinter, lane `s-gate`, 10 h, S 30 min for admission)
- Strata built by code with formal references: reasoning (arithmetic chains), code comprehension
  (outputs of short functions from the own families, executed for the reference), instruction
  following (format constraints checked by code), tool arguments (valid calls against sven's
  tool schemas, checked by schema), structured output (JSON schema), calibration (abstain on
  unanswerable). The 40-question anchor stays as the smoke test; anchor-200 (v1 S4) is the
  general-knowledge stratum.
- Each stratum: a `NetLost` bound with its power note (v1 S4 arithmetic); zero tolerance for
  deterministic security invariants (exposure count, canary pass, unauthorised action count).
- Registered before any candidate is gated; a bound is never changed after a failing candidate.
- Admission: base Qwen3-4B, 8B and the 27B each answer at least 70% per stratum.

**P2-EXIT. The five exit-gate tests** (splinter, lane `s-gate`, 6 h, none; plus E2 data)
- `crates/pipelines/tests/phase2_exit.rs`, each a scripted lineage over real verifiers:
  1. a new capability is detected on development tasks disjoint from training examples;
  2. a candidate that forgets an acquired capability fails retention;
  3. unknown-control tasks never enter the retention numerator or denominator;
  4. an invalid patch that prints success or bypasses tests fails (P2-VER);
  5. a known-good candidate passes; three degraded candidates (corrupt adapter, changed test
     definitions, injected regression) fail (flywheel item 6 negative-control suite).

### Phase 3

**V2. ATIF fill and append-only writer** (sven, lane `v-traj`, 12 h, none): as v1 V2. Needed so
an interrupted attempt's trajectory still parses for diagnosis.

**P3-TAX. Failure taxonomy** (splinter, lane `s-diag`, 16 h, none)
- `splinter-eval::diagnosis` maps an attempt to a `FailureClass` from objective evidence first:
  - resource: run conclusion Timeout/BudgetExhausted, sandbox `timed_out`, tool cap or
    repetition stop;
  - tool: `ToolAuditOutcome::Failed`, `ArgumentViolation` records (sven gate records);
  - environment/verification: verifier refusal (`patch_does_not_apply`, runner crash) or a
    canary/differential disagreement;
  - information missing: the trajectory never read a file the reference change touches;
  - retrieval: searched (search tools called) but the touched file never surfaced;
  - planning: repeated edit-test cycles past N without new files touched;
  - implementation: touched the right files, tests still fail;
  - unclear task: the task is refused on re-admission or the witness/reference disagrees.
  - else `unclassified`; a model (the 27B as judge role) may propose a class for these, kept
    as a weaker annotation; it never overrides an objective rule.
- Intervention map as data (class -> candidate interventions), the input P3.1 table.
- TDD: `crates/eval/tests/diagnosis.rs` with one recorded trajectory fixture per class.
- Adjudication: a 60-attempt sample labelled independently (OQ-V2-4) gives per-class agreement.

**P3-INT. Intervention comparisons** (splinter, lane `s-diag`, 10 h, none to write)
- `improve --experiment <dir> --strategies retrieval,harness,sft --budget-gpu-hours H`:
  for the largest failure clusters, cheap screening first (retrieval: give the solver
  `semantic_memory` or a repository map in the system turn; harness: `WorkProtocol` follow-ups,
  tool cap, compaction threshold; tool: a sven tool-definition fix) on the development
  partition, then verified SFT (Phase 4 machinery) only for clusters where screening failed.
  Portfolio record per intervention: implementation cost, execution cost, applicability,
  reversibility, safety impact, measured effect with interval, observed transfer.
- Exit (E3): at least one non-training and one training intervention against the same baseline,
  reported whether or not either helps.

### Phase 4

**B9. Adapter artifact target contract** (brain, lane `b-q35` after B1 push, 10 h, none)
- The card's `targets`, `base_digest`, adapter dtype, tokenizer digest and arch are the
  contract: `read_adapter` (`model/src/lora.rs:1260`) checks the card's targets equal the tensor
  set and refuses a mismatch by name; every trainer writes `base_digest` (qwen35
  `finetune_streamed`, qwen3 CLI, `Improve`); `WeightsIdentity` gains a tokenizer digest; the
  serve bank derives its sites from the card unless the operator narrows them, with the refusal
  naming the card's targets and the flag (fixes D6-D9); the stale comment at
  `adapter_registry.rs:82-85` goes.
- TDD: `crates/model/tests/adapter_contract.rs` (card says attention, file has MLP tensors:
  refused with both lists; missing `base_digest` from a trainer: test per trainer);
  `crates/catalog/tests/multi_adapter.rs` case: an SDK-trained 7-target adapter is served with
  no `--adapter-targets` flag. Red today.

**B10. AdamW applied-step index** (brain, lane `b-q35`, 3 h, none): v1 B7 first bullet; TDD as
v1. Moved here because it biases every SFT run with a skipped step.

**B2, B4 (SFT half), S1, B3, B5**: as v1 section 4, with B3 consuming B9's contract and adding
adapter switching, concurrent requests and cancellation tests (input P4.5).

**P4-SFT. Trajectory SFT through `practice`/`benchmark`** (splinter, lane `s-train`, 8 h, none)
- The `sft-step` view already preserves the model-input context, the target action and masks
  tool observations; P4-SFT adds: admission only of experiences with a workspace pass at
  `Executable` strength, a clean exposure audit and an intact context (no compaction marker);
  RS-P5 filters as an option; per-task cap; sequence weighting option (effective steps only).
- `TrainRequest` gains `arch`, `base_dtype`, `seq_len`, `recompute` (v1 S1 merged).
- TDD: `crates/pipelines/tests/sft_admission.rs`: an exposed, a compacted, a judged-only and a
  failed trajectory are each excluded with a counted reason.

**P4-ABL. Dataset ablation driver** (splinter, lane `s-train`, 6 h, none to write)
- Arms (input P4.3): no training; final-answer SFT (`sft-final`); action-level SFT (`sft-step`);
  + near-dup removal; + failed-attempt critique (`critic` view); rejection-sampled SFT (best of
  k verified, RS-P12 without the learned selector); SFT + replay. One registered budget; pilots
  on development pick steps; confirmatory comparison on the qualification partition once.

**P4-QUAL27. 27B ten-stage qualification** (integrator, lanes `b-q35`, `b-srv`; GPU X, see E6)

### Phase 5

**V5. Typed harness profile in sven** (sven, lane `v-harness`, 12 h, none)
- `HarnessProfile` in `crates/bootstrap/src/config`: a serialisable, digestible grouping of
  existing fields (compaction threshold/strategy/keep-recent, assembly budget, `max_tool_rounds`,
  timeouts, `delegation.policy`, `decision` features, `tools.disabled`, memory recall default
  limit) with range validation. New knobs only where an experiment needs one and nothing exists
  (`compaction.enabled`, memory default `top_k`). Applied through `EngineBuilder::config`; the
  permission model and markdown workflows untouched; a profile cannot change permissions,
  sandbox or tool authorisation (type-level: those fields are not in it).
- TDD: `crates/bootstrap/tests/harness_profile.rs`: round trip, digest stable, out-of-range
  refused, a profile cannot widen `PermissionPolicy`.

**P5-OPT. Paired harness experiments** (splinter, lane `s-harness`, 10 h, none to write)
- Splinter records the profile digest per attempt; one parameter group per comparison, family
  paired, same seeds; candidates cannot alter sandbox, hidden-test exposure or evaluation code
  (the profile type excludes them). `sven-ablation` (scripted models) is the cheap pre-screen.

### Phases 6-7 (deferred; specified so the interfaces are not built twice)

- Phase 6 needs: B6b (logprobs, token ids), V2/V3 complete, B7 rest (entropy, batched KV
  rollouts for qwen35 through B1's step, C8), S8 (brain-sampler GRPO stage), S9 (sven
  trajectories), B8 (powered ablation, pre-registered, v1 arithmetic). New: preference pairs
  from verified outcomes on the same task and start state (P6-PAIR, splinter `preference`
  view variant); terminal reward = correctness - lambda x cost with hard disqualification
  (exposure, canary, unauthorised action); rejection sampling and grouped comparison before
  GRPO. Process rewards experimental only.
- Phase 7 needs: P2-LEDGER, CL-P1 matrix, three generations over families of three kinds
  (Rust bug fixes from history; integration/API changes from P1-AUTH; concurrency/async),
  replay arms (none, fixed fraction, capability-balanced, forgetting-targeted, adapter isolation,
  adapter merge; consolidation into a new base deferred per v1 D4 for the 27B).

### Ablation and removal backlog (input Part III)

Removal process per item: consumer search across the three repos (command recorded), baseline,
ablation behind a flag, powered equivalence assessment, compatibility review, deprecation note,
delete dead tests/docs, cross-repo test. "Inconclusive" never means "remove".

| Item | Repo | Flag/arm that exists | Consumers to search |
|---|---|---|---|
| Invented-fact campaign | splinter | keep `samples/generations` as a fixture; remove from the primary pipeline's docs and defaults | `learn --kinds recall`, `generations.py`, paper E2/E3 |
| Generator paraphrase variants | splinter | `tasks variants` (`cli.rs:360`) | learn stage `variants` |
| Critique/retry generation | splinter | `critique` stage | `learn/stages.rs:611`, `critique_set` |
| Teacher demonstrations vs student rollouts | splinter | frontier teacher | `curriculum/teacher.rs:77,108` |
| Generic rehearsal vs capability-balanced replay | splinter | `DEFAULT_REHEARSAL_SHARE`, replay fraction | `rehearsal.rs:66`, `train/mod.rs:120-126` |
| Automatic consolidation | splinter | `consolidate` | `absorb/consolidate/mod.rs:230` |
| Per-step SFT vs final/filtered | splinter | `sft-step`, `sft-final` | E5 arms |
| Context compaction | sven | threshold > 1 (no flag yet: V5) | `executors/src/turn.rs:383-398` |
| Auto delegation | sven | `delegation.policy` | `task_tool/weigh.rs:42-46` |
| Semantic memory | sven | cargo feature `memory`, tool withheld by splinter | `bootstrap/src/registry.rs:153-180,328-332` |
| Decision-head guidance | sven | `decision.enabled` | `decision_ranker.rs:80-288`, `delegation_decision.rs:126,180` |
| Team tools (exported, unregistered) | sven | none | `src/run/team.rs:21-56` |
| Brain training backends, adapter variants | brain | kernel upgrade flags | P0-INV |

---

## 6. Schedule over 8-hour leases (task D)

### 6.1 L0 (running, ends 00:20)

Lanes: `s-ops` (landed), `b-q35` (B1 committed locally `4137781d`, push), `b-gate` (G1-G6, 64
files modified), `s-gate` (S4 bound + S6 preflight, 17 files modified), `v-traj` (V1, 12 files),
`t-epi` (T0 landed as `8b652351`).

Scope changes for L0 lanes (task: "any L0 lane whose scope should change"):
- **`s-gate`**: land the `NetLost` bound and S6 as they are; do **not** spend the rest of the
  lease authoring anchor-200 content or S5 (taught-facts retention): anchor-200 becomes one stratum
  of P2-ANCH and S5 is superseded by P2-LEDGER. Next in the lane: P2-VER.
- **`b-q35`**: B1 is done by `4137781d` (the v1 description of the change was wrong, D2). Push
  it, then start B9 and B10 before B2/B4: the adapter contract gates E4 on the 8B, B2/B4 only
  the 27B.
- **`v-traj`**: after V1, do V4 (cost telemetry) before V2: Phase 1 needs cost before Phase 3
  needs crash-safe trajectories. The 12-file diff must rebase onto `5d08cbea` (its base is
  already `5d08cbea`; the `~/code/sven` checkout is the stale one, D20).
- **`t-epi`**: T0 landed; the lane's next item (T4) waits for B1 to be on `origin/main`, as v1
  6.2 requires.
- `b-gate`: unchanged.

### 6.2 Leases L1-L6

GPU column: X = exclusive, S = short shared under the lock, - = none.

| Lease | Lanes (CPU unless marked) | GPU use (one process at a time) | Done by end |
|---|---|---|---|
| **L1** | `s-p0` P0-PROV then P0-ACC; `s-inv` P0-INV + R0; `s-bench` P1-OUT then P1-RUST; `s-gate` P2-VER then P2-CONF; `v-traj` V4; `b-q35` B9 + B10; `b-srv` B6a; `t-gdn` T1 | S 20 min P0-ACC GPU variant; S 45 min P1-THRU; then X 2 h T1 (card otherwise idle) | P0 exit report (G0, G1); cost captured; verifier proves tests ran; adapter contract; agent throughput known |
| **L2** | `s-bench` P1-HIST then P1-PART; `s-tasks` P1-AUTH; `s-p0` P1-RESUME; `s-gate` P2-ROLES, P2-ANCH; `b-q35` B2, B4 (SFT); `b-srv` B3 start; `t-gdn` T2 | S 30 min P2-ANCH admission; X 2 h T2; S E2 task-quality sample (generated vs history vs authored, witness and reference runs need no model) | benchmark buildable; partitions registered; P2 roles |
| **L3** | `s-bench` P1-BENCH finish; `s-gate` P2-LEDGER, P2-EXIT; `v-traj` V2; `b-srv` B3; `s-arch` S1; `t-att`/`t-epi` T3/T4 (CPU parts) | **X 6-8 h E1 baseline** (development + qualification for Qwen3-8B and 27B GGUF base; sized by P1-THRU) | **Phase 1 exit**: E1 report; G2, G4 |
| **L4** | `s-diag` P3-TAX, P3-INT; `s-train` P4-SFT, P4-ABL; `b-q35` B5; `b-srv` B3 finish; `s-arch` P0-ACC35 | X 2 h E3 screening; X 4 h E4 pilot SFT on Qwen3-8B (data from E1 verified trajectories + 27B teacher successes) | Phase 2 exit (with E2 data); Phase 3 exit (E3); E4 pilot |
| **L5** | `s-train` E5 driver; T5 prep; `v-harness` V5 | X 4 h E4 confirmatory on qualification (once); X 3 h E6 stages 1-7 (27B) | **G9 on 8B decided**; 27B qualification half |
| **L6** | `s-harness` P5-OPT; paper lane | X 3 h E6 stages 8-10; X 2 h T5 (M4); remaining X for E5 arms | G7/G8 27B; throughput milestone measured; E5 |

Rules:
- CPU-only packages (all of P0, P1-OUT/RUST/HIST/AUTH/PART/RESUME/BENCH, P2-*, P3-TAX, V4, V2,
  V5, B9, B10, B6a) run while the GPU measures; they never wait for the GPU.
- Measurement windows (E1, E4, E6) take precedence over throughput windows; T-lane exclusive
  runs go first in a lease only when no measurement is scheduled in it (L1, L2), else last.
- At most 2 cargo builds at once; lanes beyond that queue on `slot`. With 7-8 lanes per lease the
  build queue is the bottleneck; lanes write tests and code while waiting.
- Disk: before L1 delete `targets/sven` (70 GB) and `targets/plan`; at most 4 lane targets alive.

### 6.3 Why the throughput lane stays, and where (input requirement)

- Phase 4 27B SFT: an E6 controlled coding dataset of about 400 records x 4,096 tokens x 3
  epochs = 4.9M tokens. At 622 tokens/s that is 2.2 h; at 1,000 tokens/s 1.4 h (DERIVED). The
  qualification and confirmatory runs (E6, E11) repeat this several times, and Phase 6 rollouts
  plus updates multiply it.
- It needs exclusive GPU, which Phases 0-2 do not use much: L1 and L2 have idle card time.
- Optimised kernels keep auto-selection with transparent fallback: T1/T2 register through
  `native_upgrade` (falls back per dispatch, `crates/gpu-core/src/native_upgrade.rs:24-36`), and
  each kernel's test asserts both that it was selected (launch census) and that the fallback path
  stays green (brain knowledge 020).
- Abandon order unchanged (v1 D9).

---

## 7. Gates and experiments mapped to commands (task E)

New make targets live in splinter (it may name both other projects):
`campaign` (lists), `campaign/gate` (lists), `campaign/gate/g<N>`, `campaign/experiment`
(lists), `campaign/experiment/e<N>`, `campaign/report/<name>`. Each wraps existing verbs;
GPU ones run through `lease/gpu/run`. Owner "integrator" = the session that holds the GPU lease.

| Gate | Command (existing or to add) | Evidence | Runs it |
|---|---|---|---|
| G0 source integrity | `make check test` in each repo; `make deps/lock` pins on remotes; `campaign/gate/g0` (new) checks lock revisions are ancestors of `origin/main` and trees clean | P0-PROV fields | integrator, every lease start |
| G1 baseline fidelity | `campaign/gate/g1`: P0-REPORT + `make test/e2e/turn` + paper `verify` at pins | baseline report | integrator, L1 |
| G2 environment correctness | `make test/families`; `splinter benchmark prepare` admission funnel (reset by fresh tree per attempt, replay via `experiences replay`); P1-RESUME test | admission report | `s-bench`, L2-L3 |
| G3 data integrity | `experiment check` + P1-PART sealing + `assert_no_exam_family` + contamination probe; zero overlaps | refusal counts | `s-bench`, before each training |
| G4 verifier correctness | `verifier_conformance.rs`, differential false-pass rate, canaries in every qualification run | FP/FN table | `s-gate`, L2-L3 |
| G5 failure diagnosis | `diagnosis.rs` + adjudicated sample agreement | per-class agreement | `s-diag`, L4 |
| G6 intervention comparison | `splinter improve` (E3), at least 2 classes vs the same baseline | portfolio record | integrator, L4 |
| G7 training correctness | brain `make test test/rl test/gradcheck`; B9/B10 tests; nonzero finite grads and weight delta on the real base (E6 stage 4) | test logs | `b-q35`, integrator |
| G8 deployment parity | release serve check (in-process vs served verdict agreement) + E6 stage 9; tolerances, not token identity | serve report | integrator |
| G9 acquisition | `benchmark run --partition qualification` champion vs candidate, family-clustered exact sign test, registered alpha 0.05 one-sided, on the served adapter through sven | E4/E6 report | integrator; once per registered comparison |
| G10 retention | P2-LEDGER retention exam, clustered, non-inferiority margin | ledger report | integrator |
| G11 cost per verified outcome | `report` C metric with interval; candidate not worse than registered bound | P1-OUT | integrator |
| G12 ablation | E13 per backlog item | equivalence report | lane per repo |
| G13 three accepted generations | E11 | lineage + matrix | integrator (Phase 7) |
| G14 recovery | P1-RESUME kill test; disk-pressure test (`doctor` refuses under reserve); lease-crossing resume of a real run | test + one real interrupted run | `s-p0`, L2-L3 |
| G15 final unseen-repo generalisation | `benchmark run --partition external`, one query per plan (P2-ROLES ledger) | final report | integrator, campaign end |

| Exp | Question | Command | Partition | GPU | Lease |
|---|---|---|---|---|---|
| E0 | historical results reproduce at recorded revisions | P0-INV runs + paper `verify` (pins and `--latest`) | - | - | L1 |
| E1 | what the untrained agent can do | `campaign/experiment/e1` = `benchmark run` for Qwen3-8B and 27B GGUF base, k from P1-THRU | dev + qual (qual once) | X 6-8 h | L3 |
| E2 | are generated tasks representative and verified | admission funnels per source (mutation, history, authored); witness/negative pass rates; E1 solve rates by source; FP rate from P2-CONF | dev | S | L2-L3 |
| E3 | do tools/retrieval/harness help without training | `splinter improve --strategies retrieval,harness,tool` | dev | X 2 h | L4 |
| E4 | does trajectory SFT improve real tasks | `practice`/`benchmark` train arm on Qwen3-8B, confirmatory on qualification | dev (pilot), qual (once) | X 8 h | L4-L5 |
| E5 | which dataset selection | P4-ABL arms | dev; qual once for the chosen arm | X 6-10 h | L5-L6 |
| E6 | does the 27B train and deploy | ten stages (below) | controlled + dev | X 6 h | L5-L6 |
| E7 | harness optimisation | P5-OPT | dev; qual once | X | after G9 |
| E8 | preference vs SFT | Phase 6 | dev; qual once | X | after G9 |
| E9 | RL per unit cost | Phase 6 (S8, B8) | dev; qual once | X | after E8 |
| E10 | replay strategies | Phase 7 | dev + retention | X | Phase 7 |
| E11 | three generations | Phase 7 | per generation | X | Phase 7 |
| E12 | held-out repositories | G15 | external, once | X | end |
| E13 | feature removal | backlog table | dev | S/X | rolling |
| E14 | unattended fault-injected long run | a benchmark + train + gate run with injected kill, disk pressure, tool failure; `--resume` | dev | X 8 h | after P1-RESUME and G9 |

E6 stages (input P4.4) mapped:
1. load exact model + tokenizer: brain load + B9 tokenizer digest (works; digest new);
2. production-equivalent tool-calling inference: base via GGUF serve (works) and FP8 streamed
   serve (B3);
3. small LoRA on a controlled coding dataset (verified trajectories from E1): B4 + S1;
4. nonzero finite grads and real weight updates: existing gradcheck + a weight-delta assertion
   in the SDK run report;
5. save + reload: `qwen35::lora::save_adapter` + B9 contract;
6. base vs adapted generation on fixed prompts: B2;
7. held-out task evaluation: B5 in-process or B3 served;
8. serve the adapter through the production API: B3;
9. sven runs full tasks on the served adapter: B3 + V1 + B6a;
10. compare correctness, tool-call validity, resources against the base: `report`.
Adapter decode qualification (input P4.5): B1 tests (prefill/decode, NF4), B3 tests (adapter
switching, concurrent requests, cancellation and resource recovery), stage 9 vs stage 7 task-level
agreement within the serve check's tolerance.

Sizes (DERIVED from the sign-test table computed in this session, exact binomial, one-sided
alpha 0.05): critical wins and power at a true win share of 0.75 / 0.80 among non-tied clusters:
n = 15: 12 wins, 0.46 / 0.65; n = 20: 15, 0.62 / 0.80; n = 25: 18, 0.73 / 0.89; n = 30: 20,
0.89 / 0.97. The qualification partition must therefore yield about 25-30 **non-tied** family
clusters; at an observed discordance of d per cluster that is 25/d clusters (d = 0.3 gives
about 85 clusters). E1 measures d; P1-PART sizes the qualification partition from it before
any training (input P1.2 "final sizes from observed effect sizes").

---

## 8. Paper strategy (task F)

- **One paper: `papers/splinter-brain-sven-flywheel`** (decision D-PAPER, user, 2026-10-16;
  replaces the earlier two-paper split, which applied the input plan's section 11 literally and
  conflicted with the goal that the flywheel works end to end). The coding-task campaign is the
  flywheel measured on real engineering tasks, so it belongs in the flywheel paper.
  `papers/training-code-skills-with-splinter` (protocol and audit, no measured row) is folded in:
  its protocol becomes the coding-task experiment section, its audit merges into limitations and
  corrections, and the directory is removed with `papers/README.md` updated.
- The first registered campaign (E2/E3) stays exactly as registered and reported, as negative
  results, with the defects it exposed (D4 retention blind to taught facts, D5 anchor bound, exit
  code verifier) and their fixes. The v2 experiments (E1, E4, E6) get a separate pre-registration
  dated before their first run, so nothing reads as a post hoc amendment.
- Fixes still due: P1 re-pin (`dcb03816` -> reachable `79a864f` or later), P3 `verify` in
  `make check`, P2 count/test check kinds, P5 descriptive title.
- **Hypothesis** (input): verified engineering-agent experience can produce a sustained
  improvement in previously unseen repository task completion, under controlled compute
  budgets, without unacceptable loss of earlier capabilities.

| Claim the paper would need | Supporting experiment | Gate |
|---|---|---|
| The benchmark measures real engineering tasks with verified grading | E2 (source funnels, witness/negatives, FP/FN), P2-CONF | G2, G4 |
| Baseline capability and failure profile of the untrained agent, with cost | E1 | G1 |
| Failures are diagnosable into actionable classes | P3-TAX + adjudicated sample | G5 |
| Non-training interventions do or do not help | E3 | G6 |
| SFT on verified trajectories improves qualification-partition completion | E4 (8B), E6 + G9 (27B) | G9 |
| The improvement survives deployment (served adapter, sven) | E4/E6 stage 9 | G8 |
| Which dataset construction matters | E5 | G12 |
| Cost per verified outcome | E1, E4 | G11 |
| No unacceptable loss of earlier capabilities | E10, E11 | G10, G13 |
| Transfer to unseen repositories | E12 | G15 |
| Training is correct (not just loss down) | E6 stages 3-6, brain gates | G7 |

Every row is reported as exploratory, development selection, registered confirmatory or
untouched final, as the input requires. A gate that does not pass is a result.

---

## 9. Decisions, open questions, stop conditions (task G)

### 9.1 Decisions taken on the user's behalf

- **D-V2-1.** E4, the first training result, uses Qwen3-8B (qwen3 SFT, save, serve work today);
  the 27B becomes the policy only after E6 passes G7 and G8. Smaller-policy cost/quality is
  reported (input P4.4).
- **D-V2-2.** The 27B base for E1 is the Q8_0 GGUF served by `brain serve` (works today); E6
  compares it with the FP8 streamed tier before any 27B adapter is judged (v1 D5 kept: the served
  27B adapter base is the FP8 checkpoint at its training tier).
- **D-V2-3.** Isolation is `none` on this node (K6); reported in every record; tasks needing
  isolation for soundness are not admitted.
- **D-V2-4.** `TaskOutcome` is a derived evaluation plus an `attempt-report` annotation; no new
  experience format (AGENTS.md law).
- **D-V2-5.** Real-repository tasks come from leaf crates of the three repos at pinned
  revisions (fast offline suites) and four authored Rust families; the external partition is
  OQ-V2-2.
- **D-V2-6.** The family unit for real repositories is the subsystem (crate or top-level
  module); it stays the file inside the 8 Python/Lua families.
- **D-V2-7.** The `splinter benchmark` and `splinter improve` verbs compose existing stages;
  `report` replaces the missing generic report. No second experiment model.
- **D-V2-8.** GRPO, DPO-for-qwen35, logprobs (B6b) wait for G9.
- **D-V2-9.** (Superseded by D-PAPER, section 8: one paper.) The flywheel paper absorbs the
  code-skills paper and carries the campaign.
- **D-V2-10.** L0 `s-gate` stops after the bound and preflight; anchor-200 content and S5 are
  not authored in L0.

### 9.2 Open questions, with the cheapest check

| # | Question | Cheapest check |
|---|---|---|
| OQ-V2-1 | Wall time per agentic task for Qwen3-8B (paged, concurrency 4) and the 27B GGUF (concurrency 1) | P1-THRU, 45 min GPU in L1 |
| OQ-V2-2 | Which unseen repositories form the external partition (licence, offline vendoring, size) | P0-INV lists 3-5 permissively licensed small Rust crates with offline deps; the user approves the list (licence obligations checked, not assumed) |
| OQ-V2-3 | Discordance d per family cluster (sizes the qualification partition) | E1 on development |
| OQ-V2-4 | Who labels the diagnostic sample independently (a person, or a different model with a person spot-checking 20%) | ask the user before L4 |
| OQ-V2-5 | Whether brain leaf-crate history yields enough admitted tasks (commits with tests that run under 60 s) | P1-HIST dry run on one crate's last 300 commits, CPU |
| OQ-V2-6 | Whether a rootless container runtime can be installed on the node (would lift D-V2-3) | ask the site; check `unshare -U` once |
| OQ-V2-7 | 27B NF4 KV decode tokens/s with LoRA (v1 OQ1) | 32 tokens through B2, 2 min GPU |
| OQ-V2-8 | Does the 27B chat template render tool calls as it was trained (v1 OQ4) | B5 round trip on the real tokenizer, CPU |

### 9.3 Stop conditions, wired to measurements

| Stop condition (input) | Measurement that triggers it | Then |
|---|---|---|
| No meaningful verifier | P2-CONF false-pass rate above 5% on any task kind after one repair | drop that task kind from the benchmark |
| No clean baseline | G0/G1 not passing by end of L2 | stop new packages; fix Phase 0 |
| Task quality too low for cost | E2: admitted tasks per CPU-hour below 2 for a source, or witness failures above 10% | drop that source |
| No held-out improvement after repeated training | G9 not met in E4 and in the best E5 arm (two registered confirmatory queries) | return to diagnosis (input Phase 4 exit); no Phase 5-6 |
| Gains fail deployment evaluation | G8 agreement below the serve check's bound on the candidate | candidate not released; investigate I3 |
| No transfer beyond training examples | development gain present, qualification gain absent (sign test p > 0.2) | report as memorisation |
| Retention regression over limit | G10 lower bound below the registered margin | candidate refused |
| Cost exceeds value | C of candidate above baseline C by more than the registered bound with no Q gain | refuse |
| Simpler intervention equal | E3 arm not inferior to SFT arm at lower cost | prefer it; report |
| Complexity without benefit | E13 equivalence shown | removal process |
| Results too uncertain | fewer than 25 non-tied clusters available | label PILOT; collect tasks, never change thresholds |

"No useful intervention found" is a legitimate final result.

### 9.4 Deliverables (input final package) and where each lands

Source manifest (P0-PROV), architecture map and cleanup report (P0-INV), benchmark suite
(P1-BENCH + tasks), baseline report (E1), failure report (P3-TAX), intervention experiments
(E3), training artifacts and release lineage (E4/E6, release store), retention report (G10),
ablation report (E5, E13), final evaluation (E12), reproduction commands (`campaign/*` targets),
paper (section 8).
