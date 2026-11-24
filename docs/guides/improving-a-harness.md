<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Improve a sven harness

The agent loop is not the only way to solve a repository task: a sven harness
is a programmable sequence of model turns, tool calls, verifications and
retries, written as a YAML document. `solve TASKSET --harness FILE` runs such a
document on each workspace task in place of the agent loop - in the same fresh
copy of the task's snapshot, with the same confinement - and stores the attempts
as experiences, so `verify` grades them and `experiences compare` sets them
against another arm exactly as it does for the agent loop. The document may
declare an input `task` (a string) to receive the task's instruction.

The `harness-search` template automates the comparison across many candidates:

```sh
splinter init --template harness-search my-repo
cd my-repo
splinter validate recipes/harness-search.yaml
splinter run recipes/harness-search.yaml --out runs/harness-search
```

Its `harness.search` stage runs the baseline harness (`harness/baseline.yaml`) on the
pinned tasks as an arm of its own, then each of its candidates: one mutation of the baseline
(`strategy: best-of-n`) or of the best found so far (`hill-climb`), drawn by the recipe's seed
from the mutation space (`harness/space.yaml`) and kept only when sven's compiler accepts it.

The space lists knobs, each a `path` into the document and the `choices` it may
take (`[execution, limits, model_turns]` with `[4, 6, 8]`, or a task's prompt,
`[workflow, "0", task]`, with alternatives; a number, quoted, indexes a list); a candidate is the
baseline with a few knobs changed. Each candidate runs as an arm with the same agent, model,
environment, attempts and attempt seeds, and is compared with the baseline by the evaluation's
paired sign test at `alpha / candidates`: the best of several is chosen, so the split keeps the
chance that any candidate no better than the baseline wins within `alpha`. The product holds `harness.yaml` (the winner, else the baseline), each
candidate under `candidates/` and `search.json` with every candidate's decision; every arm is a
committed product, so a stopped search resumes without running one again. The [learning
factory](learning-factory.md#searching-for-a-harness) describes the stage's parameters. How a decision
is made, and the choice between acceptance gates, is in [Choose an acceptance
gate](acceptance-gates.md).
