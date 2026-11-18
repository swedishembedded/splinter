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

- 2026-11-18 Repair loop. `experiences repair RUN... --teacher REF --bench DIR` diagnoses the failed trials of
  practice runs and routes each by class (`route`): a model failure with a last good step goes to the fork
  correction; infrastructure, task, history-probing and undecided failures are skipped with the reason. The fork
  (`splinter-agent::fork_at`) lays out the pristine workspace through `bench task workspace`, keeps the student's
  steps through the last good one and has sven's fork replay them under the compact profile (every observation held
  to the recorded one; sven's `RecordedState::observed` for a step with no recorded tree digest), the shell confined
  to the tree; the teacher continues; `bench task grade` grades with the task's verifier outside any agent. Only a
  pass is stored: the student's stored prefix as copied context, the teacher's suffix, `extra.correction` lineage,
  the verifier's pass and a `revision_of` relation (the preference view pairs a repair by its copied prefix). Records are keyed by student, teacher and
  intervention: a rerun is a no-op. E2E spec: `crates/pipelines/tests/it/bench_repair.rs` (a real compact sven run
  of a scripted student, a scripted teacher through the fork, a scripted examiner).

- 2026-11-18 First repair smoke (teacher Qwen3.8-27B-FP8 on frozen brain e9044bbeb, gpu1; lane R's 4 finished A2
  practice runs, 31 trials, 13 failed): wrong content 10 failed, 9 routed, 1 accepted (legal, reward 1.0), 2 rejected
  (0.5, 0.73), 6 refused; malformed artifact 1 routed, refused; tool 1 routed, refused; history probing 1 and
  resource 1 not routed. Refusals: an absolute workspace path in the prefix (1), host tools differing from the image
  (python modules 1, git config under confinement 1), sven error text drift (2), no continuation after a final reply
  (1), a mid-run system turn (1), a support task's host-side service (1). Before the sven listing and duration
  normalisation and the request-log observations, 0 of 11 forks replayed.
- 2026-11-18 G=8 repair smoke (same teacher and brain, gpu2; lane R's support and compiled 8-attempt practice
  exports, 640 trials, 129 failed). Diagnosis: compiled wrong content 67, resource 14, history probing 3, gateway 2,
  repetition 1; support wrong content 40, resource 2. Compiled wrong content: 67 routed, 10 accepted (reward 1.0:
  lookup-fact 5, lookup-amended 2, defects-thresholds 2, lookup-definition 1), 50 rejected (the apply-* families,
  where the teacher too gets `outcome` wrong, reward 0.5), 7 refused at replay (host against sandbox output of
  `ls`, `date` and tool errors; one continuation prefix mismatch; one final reply); repetition 1 refused; support
  40 refused (no grade for service-backed tasks); resource, history probing and gateway not routed. The preference
  view over the repairs and the train trials made 129 records, 9 from repairs (9 of the 10 accepted). A run took its
  cases in trial order, so a bounded run piled onto the hard apply-* tasks: `--per-task` and `--seed` spread it.

## Open

- Real infrastructure, repetition and history-probing failures in practice exports for the fixture set: the pilot's
  infrastructure failures (gateway 504s) and probing are in exam or eval-set runs, which never become fixtures.
- Model-assisted hypotheses: the type exists (`Hypothesis`); no producer asks a model yet.
- Gateway statuses per request are not read from the gateway's usage log; a model error is typed from the score
  record's cause.
- The headless sven runner records a tool's raw output in ATIF, not what the model was shown (knowledge 173); the
  repair reads the request log instead, a fix belongs in sven's runner.
- A tool's error text differs between sven builds (the student's a0235c71 against the fork's main): such a fork is
  refused; replaying with the student's own sven build (or the examiner's sandbox) would remove that.
- A fork replays on the host, in the examiner's workspace laid out at a path of its own, so a student call whose
  observation names the sandbox's absolute workspace path (`pwd`, `cd /workspace/repo`) does not replay and the fork
  is refused; a namespace that mounts the tree at the recorded path would remove that.
- Tasks with a host-side service (support) are refused by `bench task grade`; their repairs are refused.
- Retry and decomposition interventions for practice tasks do not exist; only the fork correction is behind the
  `Intervention` interface.
