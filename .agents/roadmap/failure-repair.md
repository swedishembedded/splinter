<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Failure diagnosis and repair of practice attempts

Scope: the deterministic diagnosis of why an examined practice attempt failed (`splinter-eval::diagnosis`,
`experiences diagnose`) and the repair loop that turns a model failure into a verified corrected experience.

## Done

- 2026-11-18 Diagnosis. `FailureClass` gains the classes an examiner's evidence decides (gateway, model server,
  no answer, malformed artifact, wrong content, history probing, repetition, unsupported) and an attribution
  (model, infrastructure, task); every consumer that left environment failures out of the agent's denominators now
  leaves out every class that `exonerates_agent`. `DiagnosticResult` (core) carries the class, the rule, the
  decisive step and evidence spans; a model may only add labelled hypotheses. The repository-history rule moved to
  core so curation and diagnosis flag the same calls. `experiences diagnose` reads exports or practice run
  directories and counts classes per domain and attribution.
- Labelled fixtures: `crates/eval/tests/fixtures/diagnosis.jsonl`, 23 failed train-set practice attempts of the
  student Qwen3-4B (pilot A2 sanity runs of the base and the three A2 seeds, and lane R's first A2 rollouts),
  labelled by reading each trajectory. Precision and recall are 1.0 for every class present (wrong content 9, no
  answer 7, tool 5, malformed artifact 1, resource 1).

## Open

- Real infrastructure, repetition and history-probing failures in practice exports for the fixture set: the pilot's
  infrastructure failures (gateway 504s) and probing are in exam or eval-set runs, which never become fixtures.
- Model-assisted hypotheses: the type exists (`Hypothesis`); no producer asks a model yet.
- Gateway statuses per request are not read from the gateway's usage log; a model error is typed from the score
  record's cause.
- The repair loop (fork at the last good step, teacher continuation, re-grade by the task's verifier, lineage).
