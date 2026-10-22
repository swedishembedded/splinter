<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Verification

A learning run is judged by measurements the policy under training cannot influence. This
page explains what counts as evidence that an answer is right, how strong that evidence is,
and where verification runs. The commands are `splinter verify`, `splinter judge` and
`splinter experiences replay`; the strength labels are what `dataset build --min-strength`
filters on.

## Verifiers


`verify` appends verdicts from each task kind's own verifiers: formal
(exact match, lenient), executable checks, mutation-validated tests,
agreement among answers, and - with `--judge REF` - a judge gated by
the calibration `judge calibrate` stored for it. A verifier either
establishes that an answer is right or only refutes it: grounding (every
number, name and quotation is in the source) is a constraint, whose fail
refutes whatever else says and whose pass decides nothing, so an answer that
invents nothing but says nothing right passes no check. A kind no other
verifier can pass - a conversation - is decided by a judge, and a `learn` on it
names one (`--judge`, else `--judge-model`, else the assistant). The
judge is another model than the teacher, the generator and the policy; it is
measured before any verdict counts, on controls made from the tasks' own
references (each task's reference is the right answer to it, and a task of
another family of sources lends the wrong one) unless a stored calibration
rests on at least `--min-calibration-controls` of them (16). A judge is
trusted for what its verdicts do. Where they admit answers to a training set
(`verify`, the `teach`, `critique` and `author` stages) its passes must be
precise (0.9): a fail only withholds an answer, so a strict judge whose fails
were measured less precise has them abstain and admits nothing wrong. Where two
models are compared on its verdicts (the release gate, the exam) its passes and
its fails must both be precise, or the comparison would be left with the ties.
A judge not precise enough for its use is refused with its numbers; `splinter
judge measure` says which uses a judge is fit for. A labelled file is JSON
Lines, `{"experience": "<id>", "label": "pass" | "fail"}`.

## Where verification runs

- Closed-book and runtime tasks are graded by the verifier of the task's kind inside Splinter,
  from the task's reference and evidence. The policy never sees the reference.
- An executable check passes only when it is seen to run to its end, so a solution that exits
  before its check cannot pass. `splinter experiences replay` runs an experience's code calls
  again in the environment it recorded and compares each result with the one observed.
- A repository task is graded by applying the agent's patch to a clean copy of the snapshot,
  laying the hidden tests over it and running the project's own commands, which must go from
  failing to passing. A patch that touches a protected or hidden path is refused.
- An **exposure audit** reads the agent's trajectory for a hidden path, the hidden tests' text
  or the state root's path and fails the attempt whatever its patch did. It counts what it
  found by kind. Where no container is configured the audit is detection, not prevention; see
  [limitations](limitations.md#security-of-agent-generated-code).
- A judge is another model, measured before its verdicts count, on controls made from the tasks'
  own references. Its calibration is stored and `verify --judge` gates its verdicts by it.

Because a verifier either establishes that an answer is right or only refutes it, a pass from
a refuting check is never taken as proof. Training admits answers by the strength of the verdicts
behind them; the release gate additionally re-measures them on questions the candidate did not train
on. See [release policy](release-policy.md).
