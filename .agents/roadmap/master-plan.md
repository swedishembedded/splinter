<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Master plan: the remaining work of the brain + sven + splinter campaign (2026-10-16, 17:00 CEST)

Written on node n592 (GH200, lease job 3935373, 16:20 to 00:20 CEST) at brain `21d4e13c`,
sven `ebd169a2`, splinter `d4c1ac3` (local main; `origin/main` has since moved to `59499d9`,
seven synthetic-author and docs commits, none of which touch the files this plan names).

The raw logs of this session's runs are kept in
`~/project/shared/containers/runs/plan-2026-10-16/`: `plan-probe-noprof.log` (F1),
`plan-probe.log` (F2), `plan-kinds2.txt` (F3), `plan-probe-512.log` and `plan-probe-2048.log`
(F4), and `plan-rl.log` (F16). The build target directory was deleted afterwards.

Labels on every fact:
- **VERIFIED-BY-RUN**: a command was run during this planning session on n592, output quoted.
- **READ-IN-CODE**: read at the named file:line at the heads above.
- **DERIVED**: computed from the two kinds above; the arithmetic is shown.

This document replaces the ordering in `campaign-audit.md` section 3. Where the audit is stale or
wrong it says so in section 1.2. It is a plan: nothing here is a finding for a paper.

---

## 0. Executive summary

1. **The 27B cannot run one flywheel turn today, and the gap is larger than the audit said.**
   Of the eleven hops of a turn (section 3.1), four work, three are partial and four are missing.
   The audit named the SDK fine-tune refusal, the missing `.arch()` and serving. It missed two
   more: brain's in-process chat generation (`ChatPipeline`, which splinter's local policy
   provider uses) is qwen3-only (`brain crates/sdk/src/text.rs:305` holds a `qwen3::Qwen`), and
   LoRA is not applied in the qwen35 KV-cached decode path at all (`lora_fwd` is called only from
   the training forward, `brain crates/qwen35/src/model.rs:2119-2434`; `run_decode_step` at :3316
   has none). One brain change unblocks both serving and in-process generation: LoRA in the
   decode step (WP B1). Everything 27B in the flywheel waits on it.
2. **Throughput: 622 tokens/s VERIFIED-BY-RUN at HEAD on this node** (probe, NF4, 8192 tokens,
   idle card). The device is busy 10.76 s of a 13.18 s step. Gated DeltaNet, together with its
   mixer glue, costs **5.13 s**. Library GEMMs cost 2.65 s and other element-wise glue 1.87 s.
   The plan reaches 1,000 tokens/s only if GDN falls to about 1.5 s. That needs fused tensor-core
   chunk kernels (WP T1, T2). It must also cut 0.7 s of glue (WP T4) and get the host-exposed
   gap from 2.4 s to about 1.2 s by removing about 45k dispatches.
   - 1,000 is reachable with about 15% margin, if T1/T2 meet their budget.
   - 2,000 is declared unreachable on this design: the GEMMs alone are 2.45 s of a 4.1 s budget.
3. **On-policy data from sven needs brain first.** Brain's OpenAI server refuses `logprobs`
   (`brain crates/apiserve/src/sampling.rs:69-71`). Sven's model seam has no sampling fields and
   no token or logprob events (`sven crates/model/src/types.rs:505-595`). The plan therefore runs
   the first splinter GRPO stage on brain's own sampler through the already re-exported
   `Environment`/`Verifier` traits (`splinter crates/model/src/rl.rs:20-23`, unused today), which
   is on-policy by construction. The sven-trajectory path follows as phase 2.
4. **The gate needs a 200-question anchor and a bound in questions.** At n = 40 with
   `DEFAULT_ANCHOR_BOUND = 0.02` (`splinter crates/eval/src/gate.rs:99`), no answer may be lost.
   The DERIVED false-refusal rate of a drift-free candidate is 36% at a 3% per-direction flip
   rate. At n = 200 with "net loss at most 6", it is 3.0% (1.1% at a 2% flip rate). The rate is
   computed exactly from the difference of two Poisson counts.
   - The retention check is still blind to taught facts (knowledge 061 open: `release/gating.rs:243-261`).
   - Sealed families are enforced only under `practice --plan` (`pipelines/src/practice/stages.rs:164-167`).
5. **Several audit items are already fixed.** These are: the holdout fix, the knowledge 058/059
   fixes, the wording of the prereg anchor line and of E4, `make test/rl` with the GRPO tests, the
   CPU GDN batch defect, and the wording "partial" for knowledge 280. One new defect: the paper's
   `data/commits.json` pins splinter `dcb03816`, a commit that is on no ref (the pre-rebase twin
   of `79a864f`), so `verify` cannot run on a fresh clone.
6. **The schedule (section 6) needs at least five 8-hour leases.**
   - Lease 0 (this one): tooling, gates and CPU-side work.
   - Lease 1: LoRA decode, adapter serving, fused GDN forward.
   - Lease 2: the Q6 rerun and a tiny-qwen35 end-to-end turn.
   - Lease 3: the first registered 27B flywheel turn.
   - Lease 4: throughput M4 measurement, the powered GRPO ablation and E5.
   - The paper closes after each lease with only what that lease measured.

---

## 1. Verified current state

### 1.1 Facts this plan rests on

**Throughput (brain).**

| # | Fact | Label | Evidence |
|---|---|---|---|
| F1 | Probe W8K NF4 at HEAD: steps 2-3 at 13.18 s, 622 tokens/s, 77,151 dispatches, `cpu_s` 13.13 (host busy all step), peak 44,725 MiB, `other_contexts_start=0 end=0` | VERIFIED-BY-RUN | `qwen35_train_probe --layers 64 --tokens 8192 --rank 16 --seed 7 --bf16-residuals on --base nf4 --gdn-segment 2048 --mlp-rows 2048 --steps 3`, built from `21d4e13c`, under `flock gpu.lock`, `BRAIN_BACKEND=cuda` |
| F2 | Profiled step 2: 10,760 ms device over 63 kernel kinds (profiling inflates wall to 20.9 s, device time unaffected) | VERIFIED-BY-RUN | same, plus `BRAIN_PROFILE=1 --profile` |
| F3 | Device time by component (step 2): GDN chain incl. mixer glue 5,125 ms / 22,080 dispatches; library GEMM 2,648 / 21,896; element-wise glue 1,868 / 21,187; head, optimizer, misc 374 / 5,266; NF4 dequant 306 / 2,960; attention element-wise 262 / 2,512; RMSNorm 178 / 1,250 | DERIVED | grouping of F2's table (below) |
| F4 | Probe at 2048 tokens: 2.86 s, 717 tokens/s, peak 36,053 MiB; at 512 tokens: 1.04 s/step, 492 tokens/s, peak 29,909 MiB | VERIFIED-BY-RUN | same probe, `--tokens 2048` / `512` |
| F5 | Real checkpoint, NF4, CLI, 8192: 12.9 s, 635 tokens/s; bf16 with freed blocks kept: 640 tokens/s at 79.8 GiB | READ-IN-CODE (recorded runs) | brain knowledge 312:21-31, 311:66-70 |

The GDN group in F3 covers these kernel kinds (ms):
- `bmm_tiled` 1,017;
- `gdn_chunk_loop_f32` 821, `gdn_bwd_sweep_f32` 540, `gdn_chunk_loop_train_f32` 430;
- `gdn_layout_permute` 516;
- `l2norm_scale` 345, `l2norm_scale_dx` 254;
- `nchw_nlc` 192, `nlc_nchw` 191, `kv_expand` 189, `conv1d` 156, `gdn_row_scale_off` 126,
  `gdn_ut_fwd` 96, the masks, `conv1d_dx` 71, cumsums and decays.

The glue group in F3 covers (ms):
- `scale_row` 314, `mul` 274, `splice_add` 265;
- `concat_split` 203, `silu_mul` 186, `add2` 141, `cast_f32_bf16` 129;
- `silu_bwd_*` 186, `concat2` 74, `scale_add` 63, `copy_cols` 59, `unpack_bf16` 57, `silu_*` 76,
  `sub` 32, `sigmoid*` 30, `region_copy` 26.

**The 27B in the flywheel (brain).**

| # | Fact | Label | Evidence |
|---|---|---|---|
| F6 | SDK chat fine-tune, full fine-tune and DPO refuse qwen35. `CHAT_FINE_TUNE` defaults false (:113) and only Qwen3 sets it (:136). Behind the check the code is hard-wired to `qwen3::finetune` | READ-IN-CODE | brain `crates/sdk/src/study.rs:113,136,201-213,849-851`; `finetune.rs:437-444`; `preference.rs:275-348`; `full_finetune.rs:227,383` |
| F7 | `Improve` (GRPO) accepts qwen3, qwen35, qwen35moe | READ-IN-CODE | `study.rs:1359-1372,1407` |
| F8 | The streamed qwen35 trainer reads the same prepared chat directory (token ids plus `train.mask.bin` and the `.ex.bin` example index) that the SDK's `prepare_chat_samples` writes. Chat SFT for qwen35 is therefore wiring, not a new data path | READ-IN-CODE | `qwen35/src/finetune.rs:215,248` (`load_dataset_with_itos`); `model/src/train.rs:243-294`; `data/src/chat.rs:587-610`; SDK `finetune.rs:476-477` |
| F9 | The qwen35 KV-cached decode (`Qwen35::step` :3181, `run_decode_step` :3316, `DecodeCaches`) exists and is built lazily for a training instance (:3167-3169). It never applies LoRA. Decode fusion is disabled on a LoRA build (:3888 `c.lora.is_none()`) | READ-IN-CODE | `qwen35/src/model.rs` |
| F10 | `sample-adapter` decodes by repeated full forwards (2.3 s/token on the 27B); `infer --adapter` folds into a host fp32 map. That needs about 108 GB for 27B, above the job's host cap | READ-IN-CODE + DERIVED | `qwen35/src/adapter_sample.rs:15-17`; `cli/src/qwen35_cli.rs:173-178`; 27e9 x 4 B |
| F11 | Host memory cap of this lease: 110,150,811,648 B (102.6 GiB) | VERIFIED-BY-RUN | `cat /sys/fs/cgroup/system.slice/slurmstepd.scope/job_3935373/memory.max` |
| F12 | `brain serve` adapters: a bank on one resident Qwen3 (`qwen3/src/serve/adapters.rs:297`, 7 sites incl. MLP, `--adapter-targets all`). `qwen35/src/serve.rs` has no adapter path; `model::dispatch::LoraBank` is generic | READ-IN-CODE | `cli/src/run_cli.rs:176-201`; `qwen3/src/serve/adapters.rs:10-46` |
| F13 | Brain's chat generation `ChatPipeline` wraps `TextGenerationPipeline`, whose engine holds `qwen3::Qwen` | READ-IN-CODE | `sdk/src/chat.rs:472-500`; `sdk/src/text.rs:252,305,332-372` |
| F14 | Brain's OpenAI server accepts `seed`, `top_p`, `top_k`, but refuses `logprobs: true` and `top_logprobs > 0` by name | READ-IN-CODE | `apiserve/src/sampling.rs:24-73` |
| F15 | GRPO: no entropy term; rollouts through `logits_all` (O(T^2)); a skipped step still advances AdamW's `t` | READ-IN-CODE | `rl/src/objective/grpo.rs:117,501,535`; `model/src/train.rs:1159-1170`; `grpo-corrections.md:7-19` |
| F16 | `make test/rl` now runs `grpo_fixes`, `grpo_qwen35`, `improve_cycle_qwen35`, `continual_study`; on this node the first three pass on CPU: 11/11, 2/2, 1/1 | VERIFIED-BY-RUN + READ-IN-CODE | `Makefile:439-450`; `cargo test --release --offline -p brain-rl --features qwen3 --test grpo_fixes --test grpo_qwen35 --test improve_cycle_qwen35` exit 0 |
| F17 | `make test/full` does not include `test/rl` or `test/gradcheck`; `make test` builds no feature-gated rl test | READ-IN-CODE | `Makefile:320-331,392-393` |
| F18 | `MOE_SKIP_GPU_TESTS`: 608 lines in 295 files, about 349 bare early returns, e.g. `gradcheck/src/bf16_train.rs:394,407,422`. The repository's own testing rule tells people to set it (`.agents/rules/testing.md:40-41,208`; `AGENTS.md:1230`). Brain has no test-skip gate (none in `scripts/gates/`) | READ-IN-CODE | grep counts |
| F19 | Vulkan concurrent use of one device returns wrong results (15/30 runs); the parity gate runs Vulkan gradcheck serially | READ-IN-CODE | knowledge 300:12-25; `backend-vulkan.md:972-983`; `scripts/gates/parity-gate.sh:34-44` |
| F20 | cuBLAS, NVRTC and libcuda are `dlopen`ed, never linked. The build container's stage takes the toolkit's "userspace redistributables". `licensing-audit.md` has no CUDA entry | READ-IN-CODE | `backend-cuda/src/cublas.rs:4-6,41,108-111`; `tools/container/aarch64-cuda.Containerfile:14,19-21` |

**Sven.**

| # | Fact | Label | Evidence |
|---|---|---|---|
| F21 | Sven: the ATIF token-id and logprob fields exist but only a round-trip test fills them. `CompletionRequest` has no sampling fields. `ResponseEvent` has no tokens. `TraceStep.model_name` is never set. Trajectories are rewritten whole at each flush. The gate and memory records have no production caller | READ-IN-CODE | `atif/src/model/metrics.rs:26-32`; `model/src/types.rs:505-595`; `ci/src/runner/mod.rs:728-762`; `session-store/src/trace_session.rs:469,477`; `executors/src/thread_store.rs:228` |
| F22 | Sven's brain chat provider is D-Bus `generate` with `messages`, `max_new` and `temperature` only, and it discards the reply metadata | READ-IN-CODE | `model-drivers/src/dbus/provider.rs:155-178,470-496` |

**Splinter.**

| # | Fact | Label | Evidence |
|---|---|---|---|
| F23 | Splinter never sets `.arch(`. The base id is the directory name | READ-IN-CODE | `model/src/train.rs:349,373,578,609,670,688`; `consolidate.rs:92` |
| F24 | Anchor: 40 questions (`samples/generations/generations.py:232,262`); decision on an accuracy-rate drop > 0.02 | READ-IN-CODE | `eval/src/gate.rs:99,663-669`; `eval/src/paired.rs:72-74` |
| F25 | Retention: the gate holds the candidate to earlier releases' held-out records, not to their trained facts (knowledge 061 open) | READ-IN-CODE | `pipelines/src/release/gating.rs:229-261`; `probe.rs:218-230`; `flywheel.md:78-80` |
| F26 | Sealed families are refused only in `practice --plan`. The plan's `retention` list is only printed | READ-IN-CODE | `pipelines/src/experiment/mod.rs:351-372`; `practice/stages.rs:164-167`; `splinter/src/experiment_output.rs:45-49`; `repository-practice.md:100` |
| F27 | All 8 family funnels are mutation-only (no model); iniparse, ttlcache and exprcalc predate the `over_time_budget` bucket | READ-IN-CODE | `tasks/catalog.toml:23..156`; `tasks/README.md:85` |
| F28 | No group-count preflight. The monitor split still hard-fails on a one-group train set | READ-IN-CODE | `pipelines/src/train/mod.rs:335-337`; `train/rehearsal.rs:108-113`; `data/src/holdout.rs:365,407` |
| F29 | GRPO surface re-exported, unused in any pipeline | READ-IN-CODE | `model/src/rl.rs:20-23` |
| F30 | Paper: 168 claims (121 STRUCTURE, 36 MEASURED, 11 CITED); 232 structure checks, 231 `regex` + 1 `absent`; `verify` not in `make check`; `commits.json` splinter pin `dcb03816` unreachable | READ-IN-CODE | `papers/splinter-brain-sven-flywheel/data/*`; `scripts/verify_structure.py:30,85-91`; `git merge-base --is-ancestor` |
| F31 | E2d: the g1 candidate was refused on the anchor; rollback was refused because there is no predecessor. No two-release turn exists | READ-IN-CODE | `sections/06-results.tex:182-188`; `data/runs/w2/e2d/raw/rollback.json` |

**Disk and memory budgets.**

| # | Fact | Label | Evidence |
|---|---|---|---|
| F32 | Project drive: 500 GB, 134 GB free. Qwen3.8-27B-FP8 29 GB, GGUF 28 GB. Targets: brain 25 GB, splinter 28 GB, sven 70 GB | VERIFIED-BY-RUN | `df -h`, `du -sh` |
| F33 | A rank-16 27B adapter is 445 MiB (fp32). NF4 training at seq 128 peaked at 24.0 GB on the device and 7.0 GB host RSS. Sampling with the adapter peaked at 23.4 GB | READ-IN-CODE (recorded run) | brain knowledge 280:10-16 |
| F34 | No GPU-lock or build-slot tooling is tracked in any repo; the earlier session's wrappers lived in a scratchpad that the node change removed | VERIFIED-BY-RUN | `grep -rl 'gpu.lock\|flock'` over the three repos' scripts and Makefiles: empty |

### 1.2 Audit claims re-checked

| Audit claim (campaign-audit.md) | Status now | Evidence |
|---|---|---|
| 0.1/2.1 holdout fix stranded on L3; no single-group test | **STALE, fixed on main** | `2df4237`; `holdout/tests.rs:315,346` |
| 0.4 three missing pieces for the 27B | **CONFIRMED and incomplete**: add F9 (no LoRA in decode) and F13 (no qwen35 chat generation) | above |
| 0.5 "1,000 needs 6.6 s cut: launch overhead about 5 s plus GDN about 3.7 s" | **WRONG in its split**: knowledge 312 and F2 show the device is busy 10.76 s; halving launches bought 2%. The cut is device time (GDN 5.1 s, glue 1.9 s) | F1-F3; knowledge 312:41-49 |
| 0.7 GRPO tests in no make target | **STALE**: in `test/rl`; still not in `test/full` (F17) | F16, F17 |
| 1 knowledge 280 "Result: pass" | **STALE**: reads "Result: partial" (280:102); title still says "changes the greedy answer" (accurate) | knowledge 280 |
| 2.1 S1 CPU heap corruption, no guard | **STALE, root-caused and fixed** (`ff981da2`, regression `qwen35/tests/lora_batch.rs:26`) | knowledge 291:41-47 |
| 2.1 S3 AdamW index advances on a skipped step | **CONFIRMED** | `train.rs:1159-1170` |
| 2.2 E4 "no generation path" | **STALE**: corrected (`06-results.tex:226-228`) | |
| 2.2 prereg "2 of 40" | **STALE**: corrected in `e7afd8e` with the correction disclosed in the line (post hoc edit) | `prereg/protocol.md:37` |
| 2.3 checks are string matches | **CONFIRMED** | F30 |
| 2.5 sealing only under `practice --plan`; plan retention not consumed | **CONFIRMED** | F26 |
| 2.6 L3 split-fix risks: no preflight, monitor hard-fails, fixed-side test missing | **CONFIRMED** (all three) | F28; `holdout/tests.rs:426` |
| 2.6 "cuBLAS linked dynamically" | **IMPRECISE**: it is `dlopen`ed. The redistribution question is real for the build container | F20 |
| 2.6 brain "94 et al lines" | **INFLATED by substrings** ("metal", "detail"). With word boundaries: 7 person-attribution lines outside CHANGELOG (`.agents/roadmap/splat.md:62,92`, `samples/decision/doom/src/env.rs:351`, `scripts/data/fetch-testdata.sh:403` (dataset attribution required by CC-BY: keep), `tools/goldens/lpips_dump_reference.py:8`, `NOTICE.md:94` (licence-required: keep)) | grep `\bet al\b` |
| 2.6 splinter 23 lines | **CONFIRMED** (list in WP G4) plus the eponym comment `tasks/graphkit/src/topo.lua:150` | |
| sven 13 headerless files, date of arXiv 2512.24601 | **CONFIRMED** (`docs/technical/rlm-context-tools.md:5`); `research-upgrades.md:4-5` points at a note that is not in sven | |
| 3.1 item 1: the E2d turn "may complete today" | **RESOLVED NEGATIVE**: g1 refused on the anchor, rollback refused (F31) | |
| (new) the paper's splinter pin is unreachable | **NEW, S1 for reproducibility** | F30 |
| (new) roadmap rows stale | **NEW, S3**: `training-27b.md:62-64` (TokenObjectives for qwen35 done), `:67-69` (flakes diagnosed), `:124` (`continuous_cycle` gone), `:175` (19-20 tokens/s); `throughput-27b.md` Q3 row (bf16 is now 2048/2048, `finetune.rs:146-147`); `generic-training-capabilities.md:14-17` (head_ce landed, C8 still `#[ignore]` at `rl/tests/conformance.rs:229-232`); `repository-practice.md:139-140`; `Makefile:427` comment | |

---

## 2. Dependency graph (from code)

Edges read "A -> B: B cannot start, or cannot be tested, until A exists in code". Square
brackets give the first command that proves the prerequisite is met.

```text
OPS (lease kit) ------------------------------------------------> every GPU lane
B1 LoRA in qwen35 decode step ---------+--> B2 KV-cached AdapterSampler --> Q6 rerun (R1)
   [cargo test ... -p brain-qwen35      |--> B3 qwen35 serve engine + bank --> brain serve --adapter-manifest (qwen35)
    --test lora_decode]                 |                                       --> S3 serve check on 27B
                                        +--> B5 ChatPipeline/TextGeneration for qwen35 --> splinter in-process policy (27B)
B4 SDK chat SFT + DPO dispatch to qwen35 --> S1 arch from card --> S2 tiny-qwen35 e2e turn (CPU)
                                                                    --> R2 27B flywheel turn
S4 anchor 200 + bound in questions (prereg) ---------------------> R2, E5
S5 retention on taught facts (061) ------------------------------> R2 (second release), E5
S6 group preflight + monitor degrade ----------------------------> R2 (one-document sources)
S7 sealing in learn/practice + plan retention consumed ----------> E5
B6 apiserve logprobs + token ids --> V1 sven seam sampling/tokens --> V2 ATIF fill + append-only
                                                                   --> S9 GRPO phase 2 (agentic)
S8 GRPO stage phase 1 (brain sampler, Environment/Verifier) <-- B7 F7 batched KV rollouts (speed, 27B only)
B7 entropy + AdamW index --> B8 powered ablation (A3)
T0 make targets for probe/CLI measurement --> T1 GDN fused fwd --> T2 GDN fused bwd --> T5 M4 measurement
                                       T3 flash hd256 fused --/      T4 epilogues/K4 --/
G* gates and scrub (independent) --> P paper re-pin (after every landing)
```

Hard serialisation points:
- Brain `crates/qwen35/src/model.rs` is touched by B1, T4 and B5. B1 goes first. T4 rebases onto it. B5 uses only public `step`/`prefill_chunked`.
- Brain `crates/kernels-cuda/src/lib.rs` is the append-only registry. T1, T2 and T3 each append in their own block. The integrator regenerates the registry on conflict (`make generate/kernels/registry`).
- Splinter `crates/pipelines/src/release/*` is touched by S4 and S5. They share one lane (S-GATE).
- The splinter lock (`make deps/lock`) re-pins after every brain push the splinter lanes need.

---

## 3. End-to-end walks (each hop: WORKS / PARTIAL / MISSING)

### 3.1 One flywheel turn on the 27B

| # | Hop | Implementation today | Status | Work package |
|---|---|---|---|---|
| 1 | Source and tasks | `splinter learn` stages (sources, tasks, admission) | WORKS (model-independent) | none |
| 2 | Policy attempts (closed book) through sven, policy = 27B with the champion's adapter | in-process provider `splinter crates/model/src/local/mod.rs:20` (`brain::ChatPipeline`, qwen3-only, F13); or `remote:` through a served model (F12: no qwen35 adapters) | **MISSING** | B1 then B5 (in-process) or B3 (served) |
| 3 | Sven trajectory as SFT material | `Agent::trajectory()` `sven crates/sdk/src/agent.rs:113`; projection `splinter-data` | WORKS for SFT (text) | none |
| 3b | Trajectory as RL material (token ids, logprobs, sampling) | F14, F21, F22 | **MISSING** | B6, V1, V2, S9 |
| 4 | Experience store, dataset | `splinter-store`, `splinter-data` (holdout fixed, F28 gaps) | PARTIAL (one-group monitor split hard-fails) | S6 |
| 5 | Train a candidate adapter on the 27B | `splinter crates/model/src/train.rs:373` -> `brain::ChatFineTune` -> refusal (F6) | **MISSING** | B4, S1 |
| 6 | Adapter artifact | `qwen35::lora::save_adapter` (`qwen35/src/lora.rs:46`) writes a carded adapter-only file | WORKS (fp32, 445 MiB) | optional: bf16 save in B4 |
| 7 | Gate: improvement, retention, anchor probes of champion and candidate | needs hop 2's generator for both arms; anchor 40 (F24); retention blind (F25) | **MISSING** (27B) / PARTIAL (rule) | B1, B5, S4, S5 |
| 8 | Serve check (served agent vs in-process agent by verdict) | `pipelines/src/release/serve/server.rs:126-133` runs `brain serve --adapter-manifest --adapter-targets all`; qwen35 has none (F12) | **MISSING** | B3 |
| 9 | Release, alias, manifest | `orchestrator/src/releases.rs:151-162,350-353` | WORKS (model-independent) | none |
| 10 | Rollback | `splinter rollback <alias>` (`splinter/src/cli.rs:162`); refused with no predecessor (F31) | PARTIAL (needs two releases; never exercised on a real lineage) | R2 makes two releases, then exercises it |
| 11 | Next generation from the champion | `train.rs` `continue_from` -> brain `lora_shape` (`sdk/src/finetune.rs:472`) | PARTIAL (qwen3 only; inherits hop 5) | B4 |

### 3.2 The 27B training path

| Hop | Implementation | Status |
|---|---|---|
| Checkpoint load (FP8 directory, streamed tensor by tensor) | `qwen35/src/fp8_source.rs`, `base_source.rs:35-43` (`BaseDtype::{F32,Bf16,Nf4}`) | WORKS (34 s NF4 load, knowledge 311) |
| Tier choice | `finetune.rs:157-188` `LongSeqOverrides`, `streamed_train_opts`; refusals at :224-233 | WORKS (CLI only) |
| Step | `finetune_streamed` (:215) | WORKS (F1, F5) |
| Save | `save_adapter` (adapter only); resume via `finetune_from(.., resume)` (:50) | WORKS |
| Reload and use | `sample-adapter` (full forwards, F10); no KV path with LoRA (F9) | PARTIAL: B1, B2 |
| SDK reachability | refused (F6) | MISSING: B4 |

### 3.3 The throughput path

| Hop | Implementation | Status |
|---|---|---|
| Instrument | `qwen35_train_probe` (`qwen35/src/bin/qwen35_train_probe.rs:15-90`), `--profile` + `BRAIN_PROFILE=1` | WORKS; **no make target** (Makefile hits for qwen35 only at :449) -> T0 |
| Exclusive-card protocol | `throughput-27b.md:634-657` (lock + `nvidia-smi pmon`) | WORKS by hand; T0 makes it a target that refuses a busy card |
| Kernels | F3 (fp32 CUDA-core GDN; blockwise attention around cuBLAS) | PARTIAL: T1-T4 |
| Real-checkpoint confirmation | CLI `brain --backend cuda qwen35 finetune ... --seq-len 8192 --batch 1` | WORKS by hand; T0 target |

---

## 4. Work packages

Conventions for every package:
- Lanes are Sonnet agents in detached worktrees `~/project/shared/containers/worktrees/<name>`,
  each with its own target directory `~/project/shared/containers/targets/<name>`.
- At most 2 cargo builds run at once (the `slot` wrapper from OPS), with `CARGO_INCREMENTAL=0`.
- In brain: never `cargo fmt`, never a bare `cargo test -p <crate>`.
- Every GPU process holds `gpu.lock`.
- The tests listed under "TDD" are written first and must fail (red) on the current head before
  the implementation lands. The commit message names the red run.

### OPS. Standing lease kit (splinter, lane `s-ops`, 3 h, no GPU, < 1 MB)

The node change lost the wrappers (F34). The kit lives in splinter because splinter is the
campaign controller and may name both other projects. The brain and sven lanes call it by path
(no file in brain or sven names it).

- **Files.**
  - `scripts/lease/env.sh`: sources the host build environment, given as an argument.
  - `scripts/lease/slot`: a counting build semaphore of N lock files.
  - `scripts/lease/gpu`: `flock` on the lock path given as an argument. It runs `nvidia-smi
    --query-compute-apps` before and after, and with `--exclusive` it refuses to start when
    another compute context exists. It records `other_contexts_start/end`.
  - `scripts/lease/doctor`: prints the node, lease end (`SLURM_JOB_END_TIME`), the cgroup memory
    cap, disk free, GPU contexts, lock holder, worktrees, and the size of each target directory.
  - `scripts/lease/worktree`: creates, lists and removes lane worktrees and targets.
  - Make targets `lease`, `lease/doctor`, `lease/gpu/run`, `lease/worktree/{add,list,remove}`.
  - Every path comes in as a flag. No machine path is written into a tracked file
    (`check-no-machine-paths`).
- **TDD.** `scripts/lease/test-lease.py`:
  - two concurrent `slot 1` holders serialise;
  - `gpu --exclusive` with a fake `nvidia-smi` on `PATH` listing a context exits non-zero with the
    context in the message;
  - `doctor` against a fixture cgroup file prints the cap.
- **Gate.** `make check` (`check-scripts` requires them referenced) plus `python3 -I scripts/lease/test-lease.py`.
- **Prerequisite.** None. **First command:** `make -C ~/code/splinter lease/doctor`.

### Lane B-Q35 (brain, worktree `b-q35`)

**B1. LoRA in the qwen35 decode step** (8 h, CPU for tests, shared GPU 15 min)
- **Files.** `crates/qwen35/src/model.rs`:
  - `run_decode_step` (:3316) and the chunked prefill stage (:3988) call `lora_fwd` (:1806) after each of the 12 targeted linears. These are the sites listed at :2119-2434.
  - For `m = 1` the scratch `lora_a`/`lora_out` (:1554-1556) is already large enough.
  - Keep `decode_fusion` off for a LoRA build (:3888).
- **Interface.** Unchanged (`Qwen35::step`, `prefill_chunked`).
- **TDD.** `crates/qwen35/tests/lora_decode.rs`:
  - (a) Tiny config, random non-zero `lora_b`. The logits of `prefill_chunked` + `step` equal
    those of `logits_all` on the same sequence, within the fp32 tolerance already used by
    `lora_adapter_file.rs`. Red today: `step` ignores the adapter.
  - (b) Zeroed `lora_b` gives bit-identical step logits to a non-LoRA build.
  - (c) An NF4 tier on CUDA agrees with (a) within the tier tolerance (`#[ignore = "needs CUDA"]`).
- **Gate.** `make test` plus `cargo test --release --offline -p brain-qwen35 --test lora_decode` (CPU) and `--ignored` under `gpu`.
- **Baseline and target.** 2.3 s/token (F10) -> at least 10 tokens/s greedy at batch 1 on the
  real 27B NF4 base. DERIVED target: the Q8 GGUF path decodes about 36 tokens/s on CUDA (user
  memory, not re-measured); NF4 with unfused decode is assumed at least 3x slower.
  - **Open question OQ1:** the real number. Cheapest check: 32 greedy tokens through B2 on the
    real checkpoint (2 min GPU).
- **Risk.** M. The GDN decode recurrence and conv history are independent of LoRA. Only the
  linears change.
- **Prerequisite.** None. **First command:** `grep -n lora_fwd crates/qwen35/src/model.rs` shows sites only below line 2500 (red evidence).

**B2. KV-cached `AdapterSampler`** (3 h, after B1)
- **Files.** `adapter_sample.rs:89` `greedy` switches to `prefill_chunked` + `step`. The
  full-forward path is kept behind `--exact-forward` as the reference.
- **TDD.** The existing tiny-config equality test is extended: the KV greedy tokens equal the
  full-forward greedy tokens for 3 prompts x 16 tokens.
- **Gate.** As B1. **Target:** at least 10 tokens/s (OQ1).

**B4. SDK chat SFT and DPO for qwen35** (12 h, CPU tests, 30 min exclusive GPU)
- **Files.**
  - `crates/sdk/src/study.rs`: add `chat_fine_tune = true` for `Qwen35`, with a per-arch
    `ChatTrainer` hook (load config/vocab, score records, train).
  - `crates/sdk/src/finetune.rs:437-520`: replace the four qwen3 calls (`store_checkpoint_path`,
    `checkpoint_config`, `score_records`, `finetune_lora_controlled`) with the hook.
  - qwen35's hook calls `finetune_streamed` (F8) with `StreamedRun`, built from new builder
    knobs `.base_dtype(nf4|bf16)`, `.recompute(bool)` and `.seq_len`.
  - `preference.rs:275-348`: the reference is the adapter switched off, the same arrangement
    the `Grpo::with_reference` objective already uses.
  - Delete the refusal assertions at `study.rs:849,851`; keep :850 (full fine-tune of qwen35moe).
- **Not in scope** (decision D4): `ChatFullFineTune` and `merge_adapter` for qwen35. A 27B full
  fine-tune does not fit one card; consolidation of 27B generations is deferred.
- **TDD.** `crates/sdk/tests/qwen35_chat_finetune.rs`, on a tiny random-init qwen35 checkpoint
  written by `qwen35::init` to a temp dir:
  - (a) `ChatFineTune::from_pretrained(dir).arch("qwen35")` trains 20 steps, the held-out loss
    falls, and the adapter card's family is `qwen35`;
  - (b) `continue_from` resumes from (a)'s adapter;
  - (c) DPO 10 steps raises the chosen-minus-rejected margin;
  - (d) `replay` and `held_out` are honoured (counts in the outcome).
  - All are red today with "qwen35 does not provide chat fine-tuning yet".
- **Gate.** `make test` plus `make test/conformance`; one real-checkpoint smoke (20 steps, seq 512, NF4) under `gpu --exclusive`.
- **Target.** Real 27B SFT at seq 512 costs about 1.04 s/step (F4) plus 34 s load.
- **Risk.** M-L. `score_records` for a streamed base needs a forward-only scoring path. Use the
  training model with recompute off.
- **Prerequisite.** None. **First command:** `rg -n 'does not provide chat fine-tuning' crates/sdk/src/study.rs`.

**B5. Chat generation for qwen35** (10 h, after B1)
- **Files.**
  - `crates/sdk/src/text.rs:305` becomes an engine enum over `qwen3::Qwen | qwen35::model::Qwen35`.
  - qwen35 is built through the streamed tier (no host fp32 map, F10). An adapter is attached
    unfolded through B1. The fp32 fold (`text.rs:332-372`) stays qwen3-only.
  - `sdk/src/chat.rs:472` needs no change (it wraps the engine).
- **TDD.** `crates/sdk/tests/qwen35_chat_pipeline.rs`, tiny config:
  - a chat turn with and without an adapter; the greedy tokens equal B2's sampler;
  - a tool-call render round trip as for qwen3.
- **Gate.** `make test`. **Target.** As B2.
- **Risk.** M (chat template for the 27B: `ChatFormat::for_checkpoint`).

### Lane B-SRV (brain, worktree `b-srv`)

**B3. Adapters on a served 27B** (16 h, after B1; exclusive GPU 1 h)

Decision D5: the served base for a 27B release is the same FP8 checkpoint streamed at the tier
it was trained on. The Q8_0 GGUF is not used. This keeps the serve check like-for-like. The
GGUF bank is a later throughput item, gated by an agreement measurement.

- **Files.**
  - Move `AdapterBank` behind `LoraSites` (`generic-training-capabilities.md:19-25`;
    `qwen3/src/serve/adapters.rs`; `catalog/src/adapter_registry.rs:155` decoupled from qwen3).
  - Add `qwen35::serve::Engine::from_streamed(base_dir, BaseDtype, ..)`, which loads the device
    weights directly with no host map.
  - Two bank dispatches per targeted linear in the decode tape, as in qwen3.
  - Register in `catalog/src/resident_qwen35.rs` so that `brain serve --adapter-manifest
    --adapter-targets all` offers the adapter as a model.
  - Update the help text at `cli/src/run_cli.rs:176`.
- **TDD.**
  - `crates/qwen35/tests/engine_adapter_bank.rs` (mirrors `qwen3/tests/engine_adapter_bank.rs`):
    one batch with rows on adapter A, adapter B and none equals three single-sequence B1 decodes;
    an adapter whose card base differs is refused by name.
  - `crates/catalog/tests/multi_adapter.rs` gains a qwen35 case.
- **Gate.** `make test`; the e2e HTTP test is `#[ignore = "needs CUDA"]` and runs under `gpu`.
- **Target.** Served greedy text equals in-process B5 text for 20 prompts (exact token match),
  at a throughput of at least 10 tokens/s.
- **Risk.** L. The paged `Engine` (`serve.rs:217-555`) is f32-map-built today. If
  `from_streamed` is too large, fallback D5b: a single-sequence streamed server mode that serves
  one adapter at a time. The serve check is per adapter, so that is enough for the gate.

**B6. Logprobs and token ids from brain's servers** (8 h)
- **Files.**
  - `apiserve/src/sampling.rs:69-73`: accept `logprobs: true` (chosen-token logprob only;
    `top_logprobs` stays refused).
  - `apiserve/src/openai.rs:755,788` fills `choices[].logprobs.content[].{token,logprob}`.
  - A brain extension field `choices[].token_ids` and `usage.prompt_token_ids`.
  - Echo the `seed` used. The D-Bus `generate` action returns the same under `out_meta`.
  - The engines must record the sampled token's log-probability under the sampling temperature
    and `top_p`: the distribution that sampled (the GRPO fix of knowledge 290 depends on it).
- **TDD.** `crates/apiserve/tests/api.rs`:
  - a request with `logprobs: true` on a tiny qwen3 returns one finite logprob per completion
    token;
  - the sum equals the log-probability recomputed by a teacher-forced forward at the same
    temperature (tolerance 1e-3);
  - a seeded request repeated gives identical token ids.
  - Red today: refused by name.
- **Gate.** `make test`. **Risk.** M (streaming chunks).

### Lane B-RL (brain, worktree `b-rl`)

**B7. GRPO remainder** (10 h)
- **AdamW index.** `model/src/train.rs:1159-1170` uses an applied-step counter.
  - TDD: a run with a forced skip at step 2 gives the same bias corrections at the next applied
    step as a run without the skipped step.
- **Entropy term.** `GrpoOptions` gains `entropy_coef` (default 0, so behaviour is unchanged)
  plus a logged entropy (the code exists as `mean_completion_entropy`, `rl/src/improve.rs:276-300`).
  - TDD: an FD gradcheck of the entropy term in `grpo_gradcheck.rs`. With coef > 0 on a peaked
    toy policy, entropy rises over 20 steps.
- **F7, batched KV rollouts.** `GrpoConfig` rollouts go through a `SyncRollout`
  (`model/src/rollout.rs:100`): for qwen3 the paged engine, for qwen35 B1's `step`.
  - TDD: on a tiny model, the KV rollout's tokens and logprobs equal the `logits_all` oracle's
    for the same seed.
  - Speed target: at least 10x fewer forward tokens per group (DERIVED: O(T) vs O(T^2)).
- **C8.** Implement `chunked_head_ce_equals_unchunked` (`rl/tests/conformance.rs:229-232`)
  against `model/src/head_ce.rs` and remove its `#[ignore]`.
- **Gate.** `make test/rl test/conformance`.

**B8. Powered GRPO ablation** (4 h writing, plus GPU, after B7)
- **Pre-registration first**, as a new knowledge entry committed before the run:
  - arms A, B, C as in knowledge 292, 20 drawn seeds each, paired by seed;
  - primary: ACC(B) > ACC(A) in at least 15 of 20 seeds (exact one-sided sign test, p = 0.021);
  - DERIVED power at a true per-seed win probability of 0.8: P(X >= 15 | 20, 0.8) = 0.80;
  - secondary: C vs A under the same rule, with a family-wise correction over the two.
- **Run.** `continual_study.rs` `reduced_ablation_*` with `--seeds 20`, release build.
- **Cost.** **OQ2:** the per-seed wall time. Cheapest check: one seed of arm A, release build,
  timed (knowledge 292 gives only "one CUDA run, debug build").
- **GPU.** Shared is acceptable (the outcome is accuracy, not time). It holds `gpu` while it runs.

### Lane T (brain throughput, worktrees `t-gdn`, `t-att`, `t-epi`)

Measured start (F1-F3), step 13.18 s.

**Per-component budget at 1,000 tokens/s** (step at most 8.19 s, DERIVED):

| Component | Now, s (dispatches) | Budget, s | Package | Floor argument |
|---|---:|---:|---|---|
| Library GEMMs (base, LoRA, head, attention products) | 2.65 (21,896) | 2.45 | T4 (K4 removes about 0.2 s of head recompute; K3 A-concat) | about 470 TFLOP/s already (knowledge 310) |
| GDN chain incl. mixer glue | 5.13 (22,080) | 1.50 | T1, T2 | bytes about 60 us/token, about 0.5 s (`throughput-27b.md:232`) |
| Element-wise glue | 1.87 (21,187) | 0.70 | T4 (K7, Q6) | bandwidth about 135 MB/token, about 0.33 s (:233) |
| Attention element-wise (GEMMs are in row 1) | 0.26 (2,512) | 0.15 | T3 | |
| NF4 dequant | 0.31 | 0.31 (0 on bf16) | none | |
| RMSNorm | 0.18 | 0.15 | T4 | |
| Head, CE, optimizer, misc | 0.37 | 0.30 | T4 | |
| **Device total** | **10.76** | **5.56** | | |
| Host-exposed gap (step minus device) | 2.42 | 1.20 | fewer dispatches (77k -> about 30k), then K8 arena only if still above budget | 35 us/launch host time (knowledge 311) |
| **Step** | **13.18** | **6.76 = 1,212 tokens/s** | | margin 21% over 1,000 |

If T1+T2 reach only 2.5 s of GDN, the step is 7.8 s, which is 1,050 tokens/s: still a pass, with
no margin. If GDN stays above 3.0 s, 1,000 tokens/s is not reachable without abandoning one of
the following (decision D9 says which, in order):
- (1) bit-identity of the GDN path: accept tolerance gates, which T1 already assumes;
- (2) full recompute of GDN layers: keep GDN activations for the last k layers.
  Memory: 1.31 MiB/token/layer x 8192 = 10.5 GiB per layer (knowledge 250). At most 4 layers fit
  in the NF4 tier's 50 GiB headroom, saving at most 4/48 of the GDN recompute, about 0.1 s.
  Not worth it, so dropped;
- (3) the "real checkpoint, both tiers" strictness: report NF4 only;
- (4) the target itself: report the measured rate and the per-component gap.

2,000 tokens/s is **declared unreachable** on this design: the GEMMs alone are 2.45 s of a 4.1 s
step (DERIVED). FP8 GEMMs and dropping recompute are both rejected (`throughput-27b.md:580-599`).

**T0. Make targets for the instrument** (2 h, `t-epi` lane, no GPU to write)
- `make qwen35/train/probe ARGS=...` and `make qwen35/train/step ARGS=...`. They build the probe
  or CLI and run it under the OPS `gpu --exclusive` wrapper. The wrapper path is given as a
  make variable, so brain names nothing. The output is the PROBE lines plus a
  component-grouped table, using a `scripts/perf/group-kernels.py` that applies this plan's
  grouping.
- TDD: the grouping script on a stored table fixture sums to the table total.
- Gate: `make check/make-targets`.

**T1. Fused GDN forward, chunked, tensor-core** (24 h, `t-gdn`, exclusive GPU 2 h)

The design follows the chunkwise WY/UT form of gated delta-rule training (arXiv 2406.06484,
arXiv 2412.06464). Brain already implements that form in whole-tensor steps (`model/src/gdn.rs:826-1200`).

- **New native kernel.** `crates/kernels-cuda/cu/gdn_chunk_fwd_tc.cu`, registered in
  `kernels-cuda/src/lib.rs`, with one CTA per (head, chunk of 64). It does the following:
  - (a) Load q, k, v, beta and g with strided reads from the projection layout. This deletes
    `gdn_layout_permute` (0.52 s), `nchw_nlc`/`nlc_nchw` (0.38 s) and `kv_expand` (0.19 s),
    folding head expansion into the index.
  - (b) Apply q/k L2-normalisation on load (deletes `l2norm_scale` 0.35 s forward).
  - (c) Form the decay-masked `K K^T` and `Q K^T` products with `mma.sync.m16n8k16` (bf16 in,
    fp32 accumulate), as in `flash_prefill_f16_hd256.cu:83-89`.
  - (d) Do the UT forward substitution in shared memory (deletes `gdn_ut_fwd`, the masks and
    `bmm_tiled`'s forward share).
  - (e) Write W and U for the inter-chunk pass.
- **Inter-chunk recurrence.** The existing `gdn_chunk_loop_f32.cu` is changed to
  - read W and U directly;
  - split value columns 8 ways instead of 4 (192 -> 384 CTAs over 132 SMs, addressing the
    occupancy issue of knowledge 311:101);
  - use `mma.sync` for the `S @ W^T` and `q @ S` products.
- **Interface.** `gdn_chunk_fwd_train` (:1102) and `gdn_chunk_fwd` (:939) call the fused path
  when `native_upgrade` offers it. The whole-tensor path stays the fallback and the reference.
- **TDD, red first.**
  - `crates/gpu-core/tests/gdn_chunk_fwd_tc.rs`: against the f64 host reference already used by
    `gdn_bwd_sweep_native.rs`, at chunk 64 and a ragged tail of 5 rows, bf16 tolerance
    (max-abs relative 2e-2 on outputs, 1e-2 on the final state).
  - `crates/model/tests/gdn_chunk_fwd_train_fast.rs` gains a case that asserts the fused kernel
    was selected, from the launch census.
  - `crates/qwen35/tests/train_tier_trajectory.rs`: 30 steps, loss within 2%, LoRA cosine at
    least 0.99. This test exists and must stay green.
- **Gate.** `make test/gradcheck` (`check_qwen35_lora`) plus the three tests above under `gpu`,
  plus `make qwen35/train/probe`.
- **Target.** The GDN forward share (`chunk_loop` 0.82 recompute + 0.43 train + permutes,
  l2norm, transforms) goes from about 3.0 s to about 0.8 s. Dispatches fall by about 12k.
- **Risk.** H. The UT substitution in shared memory is precision-sensitive. Mitigation: keep the
  substitution in fp32, and use bf16 only for the products.

**T2. Fused GDN backward** (24 h, after T1, `t-gdn`, exclusive GPU 2 h)
- **Native kernels.**
  - `gdn_chunk_bwd_intra_tc.cu` covers phase 1, the `d_state`-free recompute (`gdn.rs:1400-1432`).
  - The same kernel's second entry covers phase 2, items 10-20, in reverse (`gdn.rs:1520-1600`).
    It computes the closed-form UT backward (:1546) in shared memory.
  - The `gdn_bwd_sweep_f32` sweep is split 4 ways over value columns (48 -> 192 CTAs). This is
    exact because the transition acts on the key dimension, so value columns are independent.
  - `l2norm_scale_dx` (0.25 s) and `conv1d_dx` are folded into the store epilogue.
- **TDD.** `crates/model/tests/gdn_chunk_bwd.rs` (`gdn_chunk_bwd_gradcheck`, f64 FD) on the fused
  path; `gdn_bwd_sweep_native.rs` with the split.
- **Target.** GDN backward from about 2.1 s to about 0.7 s. T1+T2 combined: 5.13 -> at most 1.5 s.
- **Risk.** H.

**T3. Fused flash attention hd256 for training** (16 h, `t-att`, exclusive GPU 1 h)
- `crates/kernels-cuda/cu/flash_train_hd256.cu`: forward with LSE, plus a backward with the dQ
  and dK/dV split. This is the FlashAttention-2 algorithm (arXiv 2307.08691), grown from
  `flash_prefill_f16_hd256.cu`, in bf16 `mma.sync`.
- It replaces the blockwise GEMMs + `attn_softmax_fwd`/`attn_dscore`/`row_dot` in
  `gpu-core/src/provider/attention.rs:161-203`.
- **TDD.** `crates/model/tests/flash_gqa_train.rs`, `crates/qwen35/tests/lora_flash.rs` and a new
  `crates/gpu-core/tests/flash_train_hd256.rs` against the existing fp32 trio.
- **Target.** Attention from about 0.75 s (DERIVED, knowledge 310:49-53) to at most 0.35 s.
  This includes its share of row 1, and lowers the GEMM row by about 0.4 s.
- **Risk.** M. This was abandoned once in favour of the blockwise form (`throughput-27b.md:34`).
  It is attempted only after T1 lands. If T1 meets its budget and the step is under 8.19 s
  without T3, T3 is dropped (D9).

**T4. Epilogues, K4, Q6, K3** (20 h, `t-epi`, after B1 has landed in `qwen35/src/model.rs`)
- **Epilogue folds.**
  - `scale_row` (0.31 s) into the GEMM's alpha, or into the producer's store.
  - `splice_add` (0.27 s) and `add2` (0.14 s) into the cuBLAS `beta = 1` accumulation into the
    destination.
  - `mul`/`silu_mul` (0.46 s) and `cast_f32_bf16` (0.13 s) into one SwiGLU kernel that writes
    the bf16 operand shadow (Q6).
  - `concat_split`/`concat2` (0.28 s): use K3's A concatenation, so no concat at all.
- **K4.** A fused head CE gradient in the forward (`model/src/head_ce.rs`, arXiv 2411.09009).
- **TDD.**
  - `crates/qwen35/tests/train_tier_trajectory.rs` (unchanged);
  - `lora_gemm_precision.rs`;
  - a new `crates/qwen35/tests/train_dispatch_budget.rs`, `#[ignore = "needs CUDA"]`. It asserts
    a W8K probe step's dispatch count is at most 35,000. It is red today at 77,151 (F1).
- **Target.** Glue from 1.87 to 0.70 s; head and GEMM about -0.2 s.

**T5. M4 measurement** (2 h exclusive GPU, after T1-T4)
- Protocol: `throughput-27b.md:634-657` through `make qwen35/train/probe` and `qwen35/train/step`.
- Runs: real checkpoint, NF4 and bf16, 3 processes, min and median of steps 2-4.
- Recorded as a new brain knowledge entry, plus a paper row (section 8).

### Lane B-GATE (brain, worktree `b-gate`, 12 h, no GPU except one gradcheck run)

**G1. Silent skips.**
- New gate `scripts/gates/check-test-skips.sh` with an allowlist
  `scripts/gates/test-skips.allow` that may only shrink. The gate fails on a new
  `MOE_SKIP_GPU_TESTS` early return or a bare `eprintln!("skipping")` + return.
- Convert first the campaign-critical files: `gradcheck/src/bf16_train.rs:394,407,422`,
  `gradcheck/src/arcface.rs:215,249,282`, `gradcheck/src/lib.rs` (36), `optim/src/lib.rs`, and
  every test under `crates/{qwen35,rl,model}/tests`.
- Each becomes `brain_testutil::skip_unavailable` (`testutil/src/lib.rs:87`), which fails with
  the reason, or an `#[ignore = "needs a GPU"]`.
- Update `.agents/rules/testing.md:40-41,208` and `AGENTS.md:1230` to the rule.
- TDD: `scripts/gates/test-skips-spec.sh`, where a fixture file with a bare skip fails and an
  allowlisted one passes.

**G2. Make targets.**
- Add `test/rl` and `test/gradcheck` to `test/full` (`Makefile:392`).
- Correct the comment at `Makefile:427`.

**G3. Vulkan shared-device race.**
- A two-thread reproducer `crates/vulkan/tests/shared_device_two_threads.rs`. It is red at about
  50% of runs today (knowledge 300). Keep it `#[ignore = "reproduces the open race"]` until fixed.
- The fix itself is not on the campaign's critical path. The 27B runs on CUDA only.

**G4. Eponym scrub.**
- Brain: `.agents/roadmap/splat.md:62,92`, `samples/decision/doom/src/env.rs:351`,
  `tools/goldens/lpips_dump_reference.py:8`. These are rewritten as title, year and arXiv id.
- Keep `NOTICE.md:94` and `fetch-testdata.sh:403`: licence-required attribution (decision D12).

**G5. Licences.**
- `licensing-audit.md` gains a CUDA section:
  - the runtime image ships no toolkit library;
  - the build stage uses the toolkit's redistributables under the NVIDIA EULA terms for that
    stage, and does not ship them;
  - cuBLAS, NVRTC and libcuda are `dlopen`ed (F20).
- No new dependency is added by any package in this plan.

**G6. Stale roadmap rows** (the list in 1.2) are corrected in one documentation commit.

### Lane V-TRAJ (sven, worktree `v-traj`, 24 h, no GPU)

Order as in `training-instrumentation.md:26-44`.

- **V1. The seam.**
  - `CompletionRequest.sampling {temperature, top_p, top_k, seed}` and `want_logprobs`
    (`model/src/types.rs:505`).
  - `ResponseEvent::Tokens {ids, logprobs}` and `ResponseEvent::Served {model, seed}` (:548).
  - The OpenAI-compatible driver sends and parses `choices[].logprobs` and brain's `token_ids`
    extension (`model-drivers/src/openai_compat/mod.rs:485-494`, `stream.rs:296-303`).
  - The D-Bus driver passes the sampling fields and reads `out_meta` (`dbus/provider.rs:155-178,470`).
  - Mock providers emit deterministic ids and logprobs.
- **V2.**
  - `session-store` fills `prompt_token_ids`/`completion_token_ids`/`logprobs`,
    `TraceStep.model_name` and `extra.sven.sampling`.
  - Append-only writer: `ci/src/runner/mod.rs:728-762` uses `atif/src/persist/ndjson.rs:38` with
    a final atomic seal.
  - Wire `push_gate_decision`/`push_memory_retrieval` from `ThreadStore` (`thread_store.rs:228`)
    and from the memory tool.
- **V3.** `Agent::trajectory()` (`sdk/src/agent.rs:113`) carries the metrics and sampling.
- **TDD, all red today.**
  - `crates/session-store/tests/sampling_record.rs`: a mock provider's ids and logprobs reappear
    unchanged in the trajectory.
  - `crates/ci/tests/append_only_crash.rs`: kill mid-run, then the file parses and is flagged
    unsealed.
  - `crates/sdk/tests/trajectory_metrics.rs`.
  - `crates/executors/tests/gate_records_reach_trajectory.rs`.
- **Gate.** `make test check` (sven).
- Scrub: the date at `docs/technical/rlm-context-tools.md:5` becomes 2025; the dangling pointer
  at `research-upgrades.md:4-5` is replaced by inline evidence.
- **Prerequisite for live end-to-end:** B6. V1 and V2 are testable on mocks without it.

### Lane S-ARCH (splinter, worktree `s-arch`)

**S1. Architecture from the card, and the 27B budgets** (8 h, after B4 is pushed and re-pinned)
- **Arch.** `crates/model/src/train.rs:349-420,578-620` reads the architecture from the base
  directory's card or `config.json` `model_type`. The reading goes through a brain helper, so
  splinter does not parse brain formats. **OQ3:** whether `loader::model_dir` exposes the arch.
  Cheapest check: `rg -n 'fn arch|model_type' ~/code/brain/crates/loader/src`. The value is
  passed as `.arch(..)` to `ChatFineTune` and `PreferenceFineTune`.
- **Knobs.** The 27B knobs (`base_dtype`, `seq_len`, `recompute`) go into `TrainRequest`. The
  default for qwen35 is NF4, recompute on (decision D6).
- **Budget preflight** before any device is claimed. It refuses with numbers when any of these fail:
  - free disk at least (adapter size x expected candidates) + 2 GB;
  - host cap at least 16 GiB;
  - device free at least 48 GiB for NF4 (F1 peak 44.7 GiB at 8K) or at least 84 GiB for bf16.
- **TDD.** `crates/model/tests/arch_from_card.rs`: a tiny qwen35 directory yields `qwen35`, and
  a qwen3 one yields `qwen3`. Preflight unit tests with injected free-space numbers.
- **Gate.** `make build test check`.

**S2. Tiny-qwen35 end-to-end turn on CPU** (8 h, after B4, B5, B3)
- New `make test/e2e/flywheel-qwen35`, which needs `BRAIN_BIN` (the pattern of `test/e2e/brain-sven`):
  - a random-init tiny qwen35 (written by a brain CLI `brain qwen35 init --tiny`, or the SDK);
  - learn a 6-fact sheet;
  - gate, release, serve check, a second generation, rollback;
  - the manifest names the first release's digest.
- Exit 3 when a prerequisite is missing (the existing convention). It asserts the stages
  happened, not quality.
- This closes flywheel item 8 (`flywheel.md:55`) for qwen35.

### Lane S-GATE (splinter, worktree `s-gate`, 20 h, no GPU)

**S4. Anchor suite of at least 200 and a bound in questions.**
- **Suite.** `samples/generations/anchor-200.jsonl`: 200 general questions with short checkable
  references (arithmetic, units, capitals, definitions, code facts), frozen by
  `eval --suite anchor --freeze` with its digest in the prereg.
  - **Construction:** by code plus a reviewed list. No item may share an entity with any invented
    fact family (`generations.py make_facts`).
  - **Admission:** base Qwen3-1.7B, 4B and the 27B each answer at least 70% (recorded per model).
- **Bound.**
  - `crates/eval/src/gate.rs:99` gains `AnchorBound::NetLost(u32)` beside the rate bound. The
    decision is on `lost - gained > k`, counted from the paired items (`paired.rs`).
  - `k = 6` for n = 200 (3%).
  - The CLI exposes `--anchor-max-net-lost`.
- **Power note** (DERIVED, exact, computed during planning). A drift-free candidate whose answers
  flip independently at rate f each way gives net = X - Y, with X and Y Poisson(200 f).
  - False refusal, P(net >= 7): 1.1% at f = 2%, 3.0% at f = 3%. For k = 4 it would be 5.5% and
    9.5%, which is why k = 6 is chosen.
  - At n = 40 with zero allowed loss, P(net >= 1) is 32% at f = 2% and 36% at f = 3%. That is
    consistent with all five 0.6B candidates being refused (F31).
  - A true net loss of 15 answers (7.5%) is released with probability 4.7%; a true loss of 10
    (5%) with probability 23%. The prereg states this.
  - The script is kept with the paper (`scripts/anchor_power.py`).
- **TDD.** `crates/eval/src/gate/tests.rs`:
  - 200 items with 6 lost and 0 gained pass;
  - 7 lost and 0 gained refuse;
  - 9 lost and 3 gained pass;
  - the rate bound remains the default for old configs.

**S5. Retention that sees forgetting (knowledge 061).**
- `release/gating.rs:243-261` holds the candidate to each earlier release's *trained* claims
  under their stored paraphrase variants (the `recall_items` builder at :229-236), with the
  held-out families kept as a second component.
- TDD: a scripted lineage where the candidate forgets 8 of 15 trained facts but keeps every
  held-out one is refused on retention. Red today: it passes.

**S6. Group preflight and monitor degrade.**
- An intake check counts the independent groups and refuses before any rollout when a split is
  required and only one group exists. This addresses `flywheel.md:86-87`.
- `train/mod.rs:335-337` and `train/rehearsal.rs:108-113` degrade to "no monitoring" with a
  recorded reason on `NoGroupToHoldOut`.
- A fixed-side record sharing a group with free records is tested (`holdout.rs:170-190`).
- TDD: three tests, red today.

**S7. Sealed families everywhere, plan retention consumed.**
- The sealed families' tree ids are written to the state root by `experiment check`.
- `learn` source intake and `practice` without `--plan` refuse them.
- The gate's retention exam includes the plan's `retention` families when a plan is active.
- **Negative control.** Plant a sealed family's file content in a source; `learn` refuses, and
  a generator, teacher or rehearsal stage given it refuses.
- TDD: `crates/pipelines/tests/sealed_everywhere.rs` (red today) and
  `experiment_retention_consumed.rs`.

**S8a. Prereg of E5** (the repository-family experiment; "E3" in `tasks/e3-plan.toml` is renamed
E5 in the plan's comment, to avoid colliding with the paper's E3). This is written into
`prereg/protocol.md` before any E5 run:
- **Policy.** Qwen3-4B (qwen3 path, works today); the 27B only after R2 (decision D7).
- **Generations and exam.** Three generations by the plan. The final exam is mdrender and
  graphkit, sealed.
- **Primary.** Gen-3 candidate vs base on the sealed families' reserved tasks: the count solved,
  an exact one-sided sign test on discordant tasks, alpha 0.05.
- **Retention.** Each generation's candidate is at most 2 tasks below the previous release on
  every earlier family's exam.
- **Funnel precondition.** One model-generated funnel per family is measured and reported
  first; iniparse, ttlcache and exprcalc are re-measured on the current binary.
- **Rule.** A refused release is a result. No retry beyond one binary repair.

### Lane S-GRPO (splinter, worktree `s-grpo`)

**S8. GRPO stage, phase 1, brain's own sampler** (16 h; GPU shared for qwen3 0.6B/1.7B runs)
- **Adapter.** In `crates/model/src/rl.rs`, implement brain's `Environment`
  (`brain crates/promote/src/env.rs:95`) over a splinter task set. Prompts come through
  `ChatTokenizer`, which renders as serving does.
- **Verifier.** Implement `Verifier` (:142) by calling splinter's verifiers in `splinter-eval`.
  The adapter crate is allowed to depend on eval (AGENTS.md table).
- **Stage.** A stage `grpo` in `crates/pipelines/src/` chooses it, composed into `learn` behind
  `--objective grpo`. It runs `brain::Improve`, whose own gate is ignored; splinter's release
  gate decides.
- **TDD.**
  - `crates/pipelines/tests/grpo_stage.rs` with a scripted brain stand-in (as `release/serve`
    tests do): rewards come from splinter's verifier, and the stage records the cycle.
  - One real-model test, `#[ignore = "needs a model"]`: Qwen3-0.6B on 20 arithmetic tasks; the
    candidate's verified pass rate is recorded, with no threshold.
- **Prerequisite.** `Improve` qwen3 works today (F7, F16). qwen35 needs B7's F7 for any
  practical speed.

**S9. GRPO phase 2, sven trajectories** (16 h, after B6, V1, V2)
- Project ATIF steps with `completion_token_ids` and `logprobs` into brain GRPO rows. The
  behaviour logprobs feed the ratio basis (`grpo.rs:353`) through a `SyncRollout` that replays
  recorded completions.
- TDD: a recorded mock trajectory's logprobs equal the row's behaviour logprobs, and a
  trajectory without them is refused by name (no silent fallback to recompute).

### Lane S-PAPER (splinter, worktree `s-paper`, 10 h + after each run)

- **P1. Re-pin.** `data/commits.json` names `79a864f` or later (F30). `verify` refuses an
  unreachable commit: `scripts/verify_structure.py` runs `git merge-base --is-ancestor`
  against `origin/main`.
- **P2. Check kinds.**
  - Add `count` (a regex with an exact match count over a file set).
  - Add `test` (a named test that must exist and pass at the pinned commit, run through the
    repo's make target).
  - Re-point S01, A03, D01 to `count` and V02, V08, G01 to `test`.
- **P3.** `make papers/splinter-brain-sven-flywheel/verify` in the root Makefile, run by `make check`.
- **P4.** Every new run record carries the three commits, `git status --porcelain` emptiness,
  the binary sha256 and the GPU contexts. A `scripts/archive_run.sh` change refuses a dirty tree.
- **P5. Title.** "The Self-learning Flywheel" becomes a descriptive title until R2 releases
  twice: "A source-to-release loop for local models: Splinter + Brain + Sven" (decision D11).

### Runs (integrator, exclusive GPU)

**R1. Q6 rerun on the real 27B** (after B1, B2, B3, B4; GPU about 2.5 h exclusive; disk 2 GB).
Pre-registered as a brain knowledge entry before the run.
- **Arm T (treatment).**
  - Training set: 8 invented facts (two forms each, as in knowledge 280) plus rehearsal rows.
  - Rehearsal: the base's own greedy answers to 64 general prompts, at a share of 0.25.
  - Training: 50 steps, NF4, seq 256, `--warmup 4`.
  - Save, then reload in a fresh process.
  - Serve through `brain serve --adapter-manifest` (B3).
- **Arm C (quantified control).** The same run without rehearsal.
- **Probes.**
  - 8 trained questions.
  - 8 held-out rewordings that do not reuse the question's wording. They necessarily reuse the
    entity; that limitation is stated.
  - 50 control questions with the base's greedy answers frozen first. They include the France
    question.
- **Primary.**
  - Arm T answers at least 6 of 8 rewordings.
  - Arm T keeps at least 48 of 50 control answers unchanged.
  - The served answers equal the in-process answers on all 66 prompts.
- **Secondary.** Arm C keeps fewer control answers than arm T, reported as a count. No test:
  one run each.
- Unmet criteria are reported. Nothing is retried except after a binary repair.

**R2. Registered 27B flywheel turn** (after S1-S7, B1-B5, R1; GPU about 5 h exclusive; disk 10 GB)
- **E2-27B.** This is E2/E3 of the protocol with the 27B (NF4) as the policy and Qwen3-4B as
  generator and teacher. It uses anchor-200 (S4), retention on taught facts (S5), and two
  generations plus a rollback.
- **Time per generation** (DERIVED; OQ1 decides the generation term):
  - training about 200 steps x 1.04 s plus 34 s load, about 4 min;
  - gate probes about (200 anchor + 96 fact probes + retention) x 2 arms x about 24 tokens,
    which is about 15k tokens;
  - at 10 tokens/s that is about 25 min. The serve check adds about 10 min.
  - About 1 h per generation, so a full turn fits in one lease.
- If OQ1 shows decode slower than 5 tokens/s, the anchor probes run on the served engine
  batched (B3), not in-process.

---

## 5. Gaps found (additions beyond items 1-9)

| Gap | Why it is needed | Package |
|---|---|---|
| LoRA missing from qwen35 KV decode (F9) | Without it there is no fast adapter generation for the gate, serving or GRPO rollouts | B1 |
| No in-process qwen35 chat generation (F13) | Splinter's local policy provider uses `ChatPipeline`; the gate compares in-process and served agents | B5 |
| Brain servers refuse logprobs (F14) | Sven cannot record what brain will not return | B6 |
| No standing lease kit in any repo (F34) | Every node change lost the wrappers; GPU contention silently ruins throughput numbers | OPS |
| Lease restart procedure | Long runs cross lease ends | section 6.3 |
| Disk lifecycle | 134 GB free; sven's target alone is 70 GB; each lane target is 11-25 GB; adapters 445 MiB each | section 6.4, OPS `doctor` |
| Paper pin on an unreachable commit (F30) | Reproducibility | P1 |
| `verify` not in `make check`; string-only checks | "Every claim verified" must fail when a claim breaks | P2, P3 |
| Run records lack commits, dirty flag, binary digest, GPU contexts | Wave-2 runs were not rebuildable (`08-reproducibility.tex`) | P4 |
| `make test/full` omits `test/rl` and `test/gradcheck` (F17) | The release gate of brain does not run the GRPO correctness tests | G2 |
| Brain has no test-skip gate (F18) | 349 bare returns report a pass on a skipped check | G1 |
| No make target for the throughput instrument | Milestones measured by hand drift in their flags | T0 |
| Cross-repo integration test for qwen35 | Each repo is green alone while the turn is broken | S2 |
| Splinter local main behind `origin/main` (7 commits) | Lanes must branch from the current remote | section 6.1, step 0 |
| Release packaging of a 27B adapter: card base digest = FP8 directory digest | `ChatPipeline`'s `WeightsIdentity::of_path` (`sdk/src/chat.rs:440-466`) digests directories; the serve check must compare the same identity | B3, S1 |
| Security: sealed-family leakage across stages | Exam contamination is the one leak that silently inflates the headline | S7 negative control |
| Licence position on CUDA redistributables | Needed before any image ships | G5 |
| Observability | The PROBE lines and kernel tables are in logs only; the paper needs them as raw records | T0 writes them under the run directory; P4 archives them |

---

## 6. Schedule over 8-hour leases

### 6.1 Per lease

GPU column: **X** = exclusive (`gpu --exclusive`), **S** = short shared runs under the lock, **-** = none.

| Lease | Lanes in parallel (file-disjoint) | GPU | Done by the end of the lease |
|---|---|---|---|
| **L0** (now to 00:20) | step 0: rebase local splinter onto `origin/main`. Then OPS, G1-G6, S4+S6 (`s-gate`), V1 (`v-traj`), B1 (`b-q35`), T0 | S (B1 CUDA case, about 15 min) | lease kit in repo; gates and scrub; anchor-200 draft + bound code; preflight; sven seam on mocks; B1 red->green on CPU |
| **L1** | B2+B4 (`b-q35`), B3 (`b-srv`), T1 (`t-gdn`), S5+S7 (`s-gate`), V2+V3 (`v-traj`), B7 (`b-rl`) | X 2 h for T1 at the end; S otherwise | KV adapter sampling; SDK SFT/DPO for qwen35; bank refactor; fused GDN forward measured; retention and sealing; sven writer |
| **L2** | B5 (`b-q35`), B3 finish (`b-srv`), T2 (`t-gdn`), S1+S2 (`s-arch`), S8 (`s-grpo`), B6 (`b-srv` after B3), P1-P4 | X 2.5 h R1, X 2 h T2 | Q6 rerun (R1) recorded; tiny-qwen35 e2e green; GRPO phase 1 on qwen3 |
| **L3** | T3/T4 (`t-att`, `t-epi`), S9 (`s-grpo`), S8a prereg | X 5 h R2 | first registered 27B turn: two generations, rollback, or a reported refusal |
| **L4** | T5, B8, E5 run (S8a) | X 2 h T5, then B8 and E5 under the lock in turn | M4 measured (met or the gap table); powered GRPO ablation; E5 |
| **L5** | paper | - | the paper closes with what was measured |

### 6.2 Serialisation rules

- **One GPU process at a time.** Exclusive windows are announced in the lease log and run back
  to back: measurements first, while the card is cold and idle, then R-runs.
- **Brain `qwen35/src/model.rs`.** B1 lands before T4 starts. B5 uses only public functions.
- **Brain `kernels-cuda/src/lib.rs`.** T1, T2 and T3 append in their own blocks, and the
  integrator regenerates the registry.
- **Splinter `release/*`.** S4, S5 and S7 run in one lane, in order.
- **Lock pins.** Re-pin in this order: brain push, then `make deps/lock` in splinter, then the
  splinter lanes rebase. Sven pushes are independent of brain.

### 6.3 Lease restart procedure

1. `make -C ~/code/splinter lease/doctor` prints the new node, cap, disk, GPU, lock and worktrees.
2. Lanes resume from their last pushed or committed state. Uncommitted work in a worktree on the
   shared drive survives a node change; the scratchpad does not. Keep no plan state in it.
3. A training run interrupted by the lease end resumes with `finetune_from(.., resume=true)`
   (`qwen35/src/finetune.rs:50`, via `--resume`). A splinter learn interrupted mid-stage is
   re-run from the stage that failed, which the run record names (`flywheel.md:84`: full
   resume is open). A registered run interrupted mid-run is reported as interrupted, and
   rerun only under the protocol's one-rerun rule.

### 6.4 Disk lifecycle

- Before L1, delete `targets/sven` (70 GB; rebuildable), the qlora-era `*-scratch`
  directories, and this plan's `targets/plan`.
- Cap: 4 lane target directories at once, about 60 GB.
- Run state goes to `~/project/shared/containers/runs/<run-id>`. Adapters are kept for releases
  only. Rejected candidates' adapters are deleted after the paper archives their digest.
- The 27B FP8 base (29 GB) is read-only and shared.

---

## 7. Decisions made on the user's behalf

- **D1.** The OPS kit lives in splinter `scripts/lease/`. Brain and sven do not name it, and
  their lanes call it by path.
- **D2.** Phase 1 of on-policy RL uses brain's own sampler through `Environment`/`Verifier`.
  The sven-trajectory path is phase 2. The claim "on-policy data end to end through sven" waits
  for S9.
- **D3.** Q6 and the 27B flywheel use the NF4 tier by default. bf16 is reported where measured.
- **D4.** No 27B full fine-tune or consolidation in this campaign. Generations chain adapters
  (`continue_from`).
- **D5.** The served 27B base is the FP8 checkpoint at its training tier, not the Q8_0 GGUF.
  D5b is a single-sequence server if the bank is too large.
- **D6.** The splinter default for a qwen35 base is NF4, recompute on, seq 256-512 for fact
  learning (F4: about 1 s/step).
- **D7.** E5 (repository families) runs on Qwen3-4B. The 27B runs it only after R2.
- **D8.** Anchor: 200 questions, bound "net lost at most 6" (section S4 power note). The rate bound remains for old
  configs. Registered before any new run.
- **D9.** Throughput abandon order: GDN bit-identity first (tolerance gates), then T3 if not
  needed, then the bf16 strictness. 2,000 tokens/s is out of scope.
- **D10.** Every GPU process takes the lock, including short tests.
- **D11.** The paper title becomes descriptive until two releases and a rollback exist.
- **D12.** Licence-required attributions (`NOTICE.md:94`, the CC-BY dataset line) stay. Every
  other person attribution is replaced by title, year and arXiv id.
- **D13.** The repository-family plan is renamed E5 to avoid colliding with the paper's E3.
- **D14.** `top_logprobs` stays refused. Only chosen-token logprobs are needed for the GRPO ratio.

---

## 8. Paper: claims per experiment

| Now (supported by existing evidence) | After |
|---|---|
| Architecture and gate as described, with string-level checks and their limits stated | P2: claims backed by counts and tests |
| E1 funnels (both waves) | none |
| Registered E2/E3: no release; the anchor refused every 0.6B candidate; 13 of 30 facts learned | S4: anchor arithmetic explained with the power note (as a method fix, not a re-analysis) |
| Post hoc E2d: one 1.7B release kept the anchor; g1 refused; no rollback | R2: two releases, retention, rollback on the 27B (if it happens) |
| E4: 27B steps fit (NF4 8K about 44 GiB, bf16 16K about 78 GiB) | T5: 1,000 tokens/s met, or the measured rate with the component gap table |
| 27B adapter changes greedy answers but overwrote a control (knowledge 280) | R1: retention with rehearsal vs without, served |
| GRPO correction implemented and unit-verified; reduced ablation NOT MET | B8: powered ablation result, whichever way |
| Throughput: 622-635 tokens/s on the real checkpoint and probe at 8K, NF4, idle card (F1, F5) | T5 |
| none | S8: GRPO stage on qwen3 (pilot) |
| none | S9 + V2: on-policy data through sven, end to end |

Pipeline: `make papers/splinter-brain-sven-flywheel/verify` (P3) then `make papers/splinter-brain-sven-flywheel/pdf`. Every run is archived by `scripts/archive_run.sh` with P4's record. `commits.json` is re-pinned in the same commit as the text.

---

## 9. Open questions (with the cheapest check)

| # | Question | Cheapest check |
|---|---|---|
| OQ1 | Greedy tokens/s of the 27B NF4 KV decode with LoRA (after B1/B2) | 32 tokens through B2's sampler on the real checkpoint, 2 min GPU |
| OQ2 | Wall time of one seed of the GRPO ablation in a release build | run arm A, one seed, `--release`, timed |
| OQ3 | Whether brain's loader exposes the architecture of a base directory | `rg -n 'model_type\|fn arch' ~/code/brain/crates/loader/src` |
| OQ4 | Whether the 27B chat template renders tool calls as the 27B was trained | B5's round-trip test on the real tokenizer (CPU) |
| OQ5 | Base accuracy of the 27B on anchor-200, and the real per-direction flip rate f | one served base pass over anchor-200 (about 10 min GPU after B3) |
| OQ6 | The device-time composition of the "attention" share inside the library GEMM row | `--profile` already splits by kernel kind, not by caller; tag the library calls by caller in T0 |
| OQ7 | Whether `qwen35::serve::Engine` can be built from device-resident streamed weights without a host map | read `serve.rs:251-300` and `Qwen35::new_on_dt` in B3's first hour |
| OQ8 | The cause of the Vulkan race | G3 reproducer |

## 10. What may be unreachable, and the fallback claims

| Target | Status | Fallback claim |
|---|---|---|
| 1,000 tokens/s | depends on T1+T2 meeting their budget (risk H) | "N tokens/s on the real 27B checkpoint at 8192 tokens, NF4 and bf16, on one idle GH200; the remaining gap by component is ..." |
| 2,000 tokens/s | unreachable on this design (D9) | stated as out of scope with the arithmetic |
| A 27B release in R2 | depends on the base answering anchor-200 and on S5 | "the gate refused the 27B candidate on <check>", which is a valid result |
| On-policy data through sven | needs B6, V1-V3 and S9: two leases of work | "on-policy inside brain's sampler, used by splinter's GRPO stage" (S8) |
| E5 on the 27B | not planned | E5 on Qwen3-4B |
