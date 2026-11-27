<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Integration of unlanded branches and worktrees (2026-11-27)

Scope: every branch, remote branch and worktree of this repository whose patches were not on `origin/main` on 2026-11-27 (`git cherry origin/main BRANCH`), with the decision taken for each: landed, or not landed and why. Rule applied: code lands only if it is sound (correct, tested, gated, consistent with main's current architecture) and adds something main does not have; superseded or experimental work stays out.

## Decisions

| Branch or worktree | Unlanded commits | Decision | Evidence |
|---|---|---|---|
| `b27-spl-1020` | 2 (`--base-precision` nf4; brain pin bc4a00ab4) | superseded | main's `BasePrecision` parses `nf4` and the served trainer takes `base_precision`; the brain pin is 5 weeks older than main's lock |
| `envrun` | 3 (factory host runner of task folders, `environment_run`, `env check`/`env run`) | superseded | episodes run through `episodes::SvenEpisodes` (`tasks harbor episodes`, 2026-11-17) and the factory's execution worker (`workers/harbor`); landing it would add a second runner of task folders. Its finding (tests laid beside the workspace) belonged to its own runner |
| `g-ce-train-report` | 2 (no patience without a monitoring set; tokens per second in the report) | superseded | both on main: `monitoring.rs` sets patience 0 for an uncarved set, `core::training` reports `tokens_per_second` |
| `lane-alg`, `vsft-ledger` (`vsft-ledger` contains `lane-alg`) | 51, 52; ledger and knowledge only | superseded | every entry (knowledge 191-203, items 21/24/25 of `experience-to-competence.md`) is on main in later form; `git diff origin/main vsft-ledger` adds no line main lacks |
| `lane-d-param-lua` | 1 (`deps/lua-param`: Lua 5.1.5 with eight run-time rule axes, a composable manual, identity test against stock) | not landed, open | a second Lua patch mechanism beside `deps/lua-dialects` (its `index_base=0` and `concat=plus` axes overlap the two dialects) with no consumer yet: pilot 1 of `rule-acquisition.md` is not started. It lands with pilot 1, which first decides whether it replaces `lua51-dialects` so one mechanism remains; the commit's author is a placeholder identity and is rewritten then. Kept on `origin/lane-d-param-lua` |
| `s-cap-20261017` | 2 (Landlock confinement; exposure audit of paths outside the tree) | superseded | `sandbox::landlock` and `eval::exposure_paths` are on main (da2349e3c, 8e9c1735b, later reworked) |
| `tts-lane-i`, `origin/tts-lane-i`, `origin/tts-lane-i-local-2026-11-27` | 2 (integer-arithmetic tasks; `jobs run-served`) | superseded | landed as c6b6f08f6 and 272a87bc0; the branch copies differ only by their base |
| `origin/corrections-stage`, `origin/corrections-stage-b` | 1, 2 (correction stage; `--no-compaction`) | superseded | `corrections_session`, `curriculum::corrections` are on main; `--no-compaction` landed (c62eed38a) and left with the legacy verbs (ce376ea07) |
| worktree `splinter-api3` (uncommitted, 106 files, on 2184ce842) | draft "training, dataset checks, scoring, devices" over brain's API | superseded | main landed the same programme as steps 1-5 of `api-only-brain.md` (2026-11-17/18), choosing character-based chunking where the draft calibrated a token ratio. Patch kept outside the repository: `~/project/shared/archive/splinter-integration-2026-11-27/splinter-api3-on-2184ce842.patch` |
| worktree `splinter-envfactory` (uncommitted, 23 files, on b3a6bb9ba) | defects of the first compile on real projects, and a `--for-sven` export | partly landed (66a7748e0) | landed: fault novelty from changed lines, drawn order numbers, two agreeing suite runs, bounded seed skipping (knowledge 247, 248; the 64 GiB address cap was already on main). Not landed: `env export --for-sven`, `Instance::sven_layout` and the `WORKSPACE_DIR` fallbacks in the plugins' scripts, made for a `sven rollout run` command sven does not have (`recursive-improvement.md`, 2026-11-17 correction); a document-questions premise format change with no recorded reason. Patch kept: `.../splinter-envfactory-on-b3a6bb9ba.patch` in the same directory |

Branches whose `git cherry` was empty although `git status` showed them ahead (`leftover-cal-1017`, `s1-signal`) hold patches already on main and were not touched.

## Open

- [ ] `lane-d-param-lua`: land with rule-acquisition pilot 1, unified with `deps/lua-dialects`.
