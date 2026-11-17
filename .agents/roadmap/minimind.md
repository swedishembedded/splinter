<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# minimind: random weights to a small agent, in one resumable command

Scope: `samples/minimind`, one idempotent, resumable command that trains a tiny model from random
weights to an agent solving elementary workspace tool-use tasks: synthetic world -> tokenizer and
random-init checkpoint -> pretraining -> probes -> expert demonstrations through real sven -> full
chat fine-tune -> the model's own verified episodes -> expert iteration -> bench's sealed paired
verdict -> report. Brain is reached only over its HTTP API; sven runs the agent against an
OpenAI-compatible endpoint; bench examines. The sample keeps only what is specific to the demo.

Decisions (2026-11-15): architecture is configuration (`models.<label>.arch` plus its own config,
handed to brain unread), so the same sequence runs on several architectures over shared data
stages; Ouro first. Configuration is YAML (`minimind.yaml`). No downloads by default; a fetched
corpus is a pinned, checksummed, ledger stage. Demonstrations are the exact requests sven sends,
captured on the wire, and only verifier-accepted episodes become records.

## State

| Piece | Where | State |
|---|---|---|
| Stage ledger (key = hash of config slice, inputs, revisions; begin/commit/invalidate pack; attempt ids never reused; outputs re-hashed on reuse; one process per run dir) | `crates/store/src/stage_ledger.rs`, shared pack lock `pack.rs` | done, specs in `crates/store/tests/it/stage_ledger.rs` |
| Agent episode under sven's compact profile (`compact/1`, three tools, first request under 4 KB measured by its spec) and the policy endpoint (an OpenAI-compatible endpoint answered by a program, every request kept; also a recording proxy to a served model) | `crates/agent/src/workspace_episode.rs`, `policy_endpoint.rs` | done; sven pinned to 6f4b1716 |
| Training records from recorded model calls (`generic-messages-v2`, one record per call, the request as sven sent it as context, its reply the one supervised message); the same episode writes the same bytes (shell durations, server tool-call ids and the attempt number made canonical) | `crates/data/src/wire.rs` | done; spec `two_runs_of_one_episode_write_the_same_bytes` |
| Proportionate invalidation: a stage key holds its config slice, seed, input digests, its recipe version (`src/recipe.rs`, pinned by a spec over the generators' output) and the revisions of the services it consumes (sven for episodes, the brain binary digest for brain stages, bench for exams); this repository's revision is in no key, only in the manifest. An episode's digested outputs are its `outcome.json` and records, never sven's timestamped trajectory. A managed brain starts only when a stage needs it, so a finished run starts none | `samples/minimind/src/{recipe,stages,collect}.rs`, `crates/model/src/scratch.rs` | done; acceptance specs `a_change_recomputes_what_consumes_it_and_nothing_else`, `the_same_episodes_write_the_same_bytes_whatever_was_interrupted` |
| From-scratch model service seam (`ArchSpec` = arch name + its own config, `ChatFormat`, checkpoint, pretrain/full chat train, serve, raw-text completion) | `crates/model/src/scratch.rs` | done over `brain/train` `init_model`, `pretrain`, `full_train` |
| Bench examination run for real (smoke, 2026-11-16): 4 sittings of about 100 s each (59 exam + 20 anchor items, 3 attempts, 32 at once), all tiers 0.00 for pretrained, sft and improved; both verdicts Invalid on bench's `same_tasks` check (4 tasks differ in content or verifier between runs that ran one generated collection), signatures verified | `samples/minimind`, bench c883edf1 | open in bench (lane X) |
| Bench examination: each checkpoint sits the sealed exam + anchor once (`Examiner::sit`, exam and anchor from their own private seed files, bench's control profile for the spurious-reward arm, the endpoint's key passed), any two sittings paired into one signed verdict (`Examiner::verdict`, with `--control`); per-tier and anchor pass rates of both sides read back (aggregates only); only a verified Promote promotes | `crates/pipelines/src/bench/examine.rs`, `samples/minimind/src/exam.rs` | done; specs with a stand-in `bench` script and an examiner double |
| Run directory file count: a committed stage's episodes and Harbor task folders packed into one JSON-lines file each, reused stages read the pack (each file checked against its committed digest) | `samples/minimind/src/{pack,collect}.rs` | done; spec asserts the trees are gone |
| Terminal: a probe panel after every probe (passes by family, a solved and a failed transcript with the tool calls, the answer and the verdict) and a table across stages with the exam tiers | `samples/minimind/src/panel.rs` | done |
| Sample generators: YAML configuration with smoke/default/64m profiles and models as `arch` + config; deterministic world corpus; three training families (combine via `./calc`, registry lookup, log count) as Harbor folders with a trusted verifier; the expert policy reading every value from tool results; the compact chat template and its jinja twin (rendered identically, held by a spec) | `samples/minimind/src/{config,world,tasks,expert,template}.rs` | done |
| Stages over the ledger (init, fetch, world, tasks, demonstrate, then per model label: checkpoint, probe, pretrain, probe, sft, probe, scratch exam, explore, improve, probe, experience exam; report), the `run`/`status` program | `samples/minimind/src/{stages,collect,report,main}.rs` | done; acceptance specs (fresh run, rerun trains nothing, kill in SFT resumes, interrupted episode reruns under a new attempt and counts once, corrupt checkpoint caught, changed seed/setting, fetched corpus pinned, no promotion without a verified Promote) pass against a model-service double; against real brain the run stops at the checkpoint stage with `needs ...` |

## Measured (2026-11-16, smoke, seed 7, gpu3, brain d84390ff5)

| Stage | Result | Time | Throughput |
|---|---|---|---|
| world, tasks, demonstrations (1600 expert episodes through sven, 16 at once) | 13300 documents; 1600/1600 accepted, 5868 records | about 3 min | |
| init_model (host) | 2.1M-param Ouro, vocab 1024 | under 1 min | |
| pretrain 600 x 128 x 256 | loss 6.98 -> 0.34; probe 1/12 | 687 s | about 50k tok/s (batch sweep 8/32/128/512/2048: 27k/56k/69k/75k/77k tok/s; CPU 700 tok/s) |
| full_train 4000 steps, grad_accum 4 | loss 1.43 -> 0.11; probe 3/12 (4/12 on a rerun) | 834 s | |
| explore (96 own episodes) | 13 verified | | |
| improve 1000 steps on 40 records | probe 4/12 | | |
| bench exam | not run | | |

The fine-tuned model has the protocol; copying names and numbers from tool results is what it gets
wrong. Defects found on the way and fixed: brain refused records whose messages lacked `content`,
repeated tool-call ids or carried a server's `refusal` field (`data::wire` normalises them); a
managed brain with `--serve-adapters` refuses a store with no adapter base; seeds above 2^53;
`--device gpuN` alone leaves no budget for `init_model`'s host work; NVRTC must be on the brain
process's library path (prepended, or Vulkan enumeration loses the cards).

## Levers (2026-11-16, smoke, seed 7, gpu3, brain 16c2db09e; development probes of 12)

Measured one at a time, each on top of the one before, in one run directory (the ledger recomputes only
what a change reaches). Pretraining 600 x 64 x 512 unless said.

| Lever | pretrained | sft | improved | own episodes verified | Note |
|---|---|---|---|---|---|
| pass-2 settings, brain d84390ff5 (300 x 128 x 256 pretrain, 1500-step chat SFT) | 0 | 1 | - | 0/96 | baseline L0, 13.5 min |
| new brain, 1500 x 64 x 512 pretrain | 4 | 0 | - | 0/96 | 25 min: over the smoke budget |
| arguments spelled as the served template does (knowledge 159) | 1 | 0 | 0 | 3/96 | |
| tokenizer without digits (`tokenizer.split_digits`) | 1 | 0 | - | 0/96 | no change alone |
| copy drills in the world (`world.recall` 20000) | 3 | 0 | - | 0/96 | |
| chat SFT at lr 1e-4, 1000 steps | 3 | 0 | - | 0/96 | the chat objective, not its rate (knowledge 160) |
| SFT as text (`chat_training.as_text`, 400 x 32 x 1024) | 3 | 3 | 0 (chat improve) | 27/96 | |
| improvement as text too, own verified + 1:1 rehearsal, 40 x 8 x 1024 | 3 | 3 | 4 | 27/96 | 11.8 min end to end |

Retrieval probe (`src/retrieval.rs`): raw-text continuation of 12 held-out listings (`L1:name: 7145` ...
then `The code of NAME is`) and 12 request-to-path prompts, exact match of the copied value; run after
random init and after every pretraining segment (`pretrain_segments`, 6 in the default profile, each
continuing the last), so the run shows retrieval against pretraining tokens.

Retrieval against pretraining tokens (knowledge 161): smoke model on copy drills only (100000 docs),
0/0/0 of 24 after 16/33/49M tokens, 10 of 24 after 65M; the 4-pass vocabulary-4096 model on the mixed
world, 0/1/0/0 of 24 after 16/33/49/66M at 23k tok/s (stopped). The default profile is therefore the smoke
model pretrained on 197M tokens of a drill-dense world in six segments; smoke stays short and does not
reach the jump.

Default run of 2026-11-16 (brain 16c2db09e, 82 min): retrieval 0, 3, 9, 13, 13, 16, 19 of 24 after
0/33/66/98/131/164/197M tokens; probes pretrained 6, sft 6, improved 7 of 12 (lookup 3/4, count 4/4, combine
0/4); own episodes 146/384 verified; sealed exam 0.00 on every tier for every checkpoint, both verdicts
Reject (valid, signed, bench 421bebc2). On the exam the model makes no tool call at all: it answers in its
first turn. The three fixed-phrasing families taught templates, not the habit of acting with tools.
Next step (in progress): the skills generator (`src/skills.rs`, `src/formats.rs`): read, chain (2-4 hops),
compute with a helper whose usage comes from its own `--help` (four argument styles), edit a key, count;
seven file formats (two only in the held-out probe), a grammar of frames and synonyms with a disjoint
held-out set; the expert inspects first (`ls -R`), reads, discovers, acts, reads back, answers. Probes report
development and held-out pass rates and how often the first reply is a tool call. Chat stages trained as
text supervise every token, tool results included (brain's `pretrain` has no loss mask).

What it still cannot do: copy a file name or a code it has not seen often (combine and lookup 0 of 4
throughout). Pretraining throughput is about 50k tok/s at any batch from 8 x 1024 to 128 x 1024 on this
model (sweep through `brain/train pretrain`, 20 steps each).

## Toward the sealed exam (2026-11-17, default profile, seed 7, gpu3, brain 2784b15ab, sven 0d7d6643)

Each row a run from random init (v17 reuses v16 up to the fine-tune); probes of 12; exam = bench's sealed
workspace-tools exam (tiers a/b/c of 15/24/20 items, 3 attempts), pretrained / fine-tuned / improved;
every verdict valid and signed.

| Run | Change | probe pre/sft/imp | held-out pre/sft/imp | exam a/b/c pre | sft | improved | verdicts |
|---|---|---|---|---|---|---|---|
| v11 | episodes under the exam's conditions (knowledge 165) | 5/0/4 | 2/4/3 | 0/0/0 | 0/0/0 | - (control failed: block) | Reject |
| v13 | answer values numbers/words/codes, root files, files named by name | 2/2/3 | 3/1/1 | 0/0/0 | 0/0/0 | 0/0/0 | Reject x2 |
| v14 | half of all names random letters, 22 MB English | 4/1/2 | 1/3/1 | 0/.03/0 | .02/.01/0 | 0/.03/0 | Inconclusive, Reject |
| v15 | hidden 256 x 4 passes, 400 MB English, copy drills 600k->150k, teacher, SFT 5e-5 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | Reject x2 |
| v16 | v14 + names from the English corpus's words + teacher in SFT (6000 x 3e-4) | 6/1/4 | 5/3/3 | .13/.01/0 | 0/0/0 | .02/0/0 | Reject x2 |
| v17 | v16, fine-tune 1000 x 1e-4 | 6/6/5 | 5/3/4 | .13/.01/0 | .04/.03/0 | 0/.04/0 | Inconclusive, Reject |
| final | the default profile of e27eebdd8 (v17 without the teacher), from init, 81 min | 6/5/4 | 5/5/5 | .16/.01/0 | 0/.10/0 | 0/.08/0 | Inconclusive, Reject |

Findings: under the exam the model read a file the listing did not show in every one of its reads until
names stopped being only made-up words (v13: 0 of 1041 reads); a long fine-tune undoes pretraining
(v16 dev pass 0.50 at 1000 steps to 0.08 at 6000); low-rate (5e-5) fine-tuning under-fits at this size and
diluting the copy drills loses retrieval (v15: 5+12 of 24 against 12+12). Hidden 384 (v12) did not lower
the pretraining loss (0.850 against 0.855) at the same tokens. The teacher (Qwen3-4B served by brain,
no thinking, temperature 0.6) was admitted on 1269 of 10000 attempts, mostly count and read; its effect
alone is not measured (off by default).

## Open

- sven: before the compact profile, an embedded engine with `agent.system_prompt` set still sent
  sven's own 25 KB system prompt, including skills discovered in the host user's home directory
  (measured 2026-11-15, 25535 bytes); the compact profile replaces it. Whether the default
  profile of an embedded engine should read the host's skills is sven's to decide.
- `sven_sdk::rollout` does not exist on sven main; episodes are collected by splinter through
  `crates/agent` with durable attempt ids from the stage ledger (practice task folders: `episodes`,
  `splinter tasks harbor episodes`, since 2026-11-17).
- Done since: brain lane B landed init_model, pretrain and full_train; the bullet below is history.
- Brain (lane B): the from-scratch action (tokenizer + random-init checkpoint of a named architecture
  with the template's special tokens and chat template, installed as `local/<name>`), raw-text
  pretraining and a full chat fine-tune over `brain/train`, resumable by job identity, and serving an
  installed from-scratch checkpoint with tool calls. Then `ServedBrain` submits them and the model
  stages run for real; a second architecture (a tiny `qwen3`) is a configuration entry once brain
  can make it from scratch.
- Brain: `full_train` of an Ouro from scratch degrades context use (knowledge 160); the sample trains its
  chat stages as text until it is fixed. Served pretraining stays near 50k tok/s across batch sizes.
- Not yet run: GRPO as a selectable improvement method (brain's ouro score/policy_trace); the teacher's
  effect alone; a capacity step once the regime holds. No verdict has promoted: the fine-tune and the
  improvement do not yet beat the pretrained checkpoint on the exam.
