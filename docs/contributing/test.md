<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Test

```bash
make test                          # every test; no model, GPU or network needed
make test/crate/splinter-pipelines # one package; `make test/crate` lists them
make test/families                 # the repository task families: baselines, conventions, fault detection
make test/methods                  # one smoke per learning method on a toy model, with held-out scores before and after
make test/conformance              # every algorithm against its spec, with an evidence file; see validation
make test/conformance/integration  # the slower rows; every deliberate defect of the mutant table must fail the suite
make test/e2e                      # lists the end-to-end targets
```

`make test` loads no model. The end-to-end targets need built brain and sven binaries and a real model, and say so:

- `make test/e2e/brain-sven` has brain serve Qwen3-0.6B and sven complete a tool-using task with it, judged on a
  number only a file holds. It runs on the CPU backend and exits 3, never 0, when a binary or the model is missing, so a
  skipped run is never read as a pass. Name the binaries with `BRAIN_BIN`, `SVEN_BIN` and the store with `MODELS_DIR`.
- The learning factory's end-to-end scenarios run in `cargo test -p splinter-factory --test it`, against scripted
  models, a fake execution backend and a scripted trainer, with no experience database:

  | Scenario | Test |
  |---|---|
  | A: a document to a trained SFT candidate | `closed_book::scenario_a_a_document_becomes_a_trained_sft_candidate` |
  | B: a document to executable RLVR tasks | `qualify::scenario_b_a_document_becomes_executable_rlvr_tasks` |
  | C: a repository to agent trajectories, SFT and preference | `practice::scenario_c_a_repository_becomes_sft_and_preference_datasets` |
  | D: the full training loop with the release gate | `promotion::a_candidate_is_promoted_only_when_it_improves_transfers_and_retains` |
  | E: an interrupted, resumed and incrementally changed corpus | `foreach::a_crashed_fan_out_resumes_without_repeating_a_finished_item`, `foreach::one_changed_item_recomputes_only_itself` |

- A whole learning-factory run (attempts through the Harbor worker, training and evaluation on a served model) is a
  [run workspace](../guides/workspace.md)'s: `run.sh RECIPE DEVICE` runs a recipe against its own brain server, and
  `scripts/run-metrics.py RUN_DIR` reads what its evaluation arms did.

Write the test from the specification first and see it fail, then implement. An algorithm's test holds it to an
independent reference, never to a copy of its own arithmetic; [validation](validation.md) maps each algorithm to the
test that proves it and to the defects that test is known to catch. Tests assert observable behavior at a real
boundary: a test that returns early reports a pass, and a gate rejects a test that skips itself silently. The tests that
document a user-facing walk-through run the real binary against an empty home, so the documentation cannot drift from the
program; see [documentation](documentation.md).
