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

`harness optimise` automates the comparison across many candidates. The search seed has no default, because the same seed draws the same candidates and so reproduces a search:

```sh
splinter harness optimise TASKSET-ID --baseline harness.yaml --space space.yaml \
    --evaluations 12 --attempts 2 --strategy annealing --search-seed 7
```

The space lists knobs, each a `path` into the document and the `choices` it may
take (`[execution, limits, model_turns]` with `[4, 6, 8]`, or a task's prompt,
`[workflow, "0", task]`, with alternatives; a number, quoted, indexes a list); a candidate is the baseline with a few knobs changed. The search
is sven's (best-of-N or simulated annealing) and follows the share of tasks the
candidate solved, which guides and never decides. Each candidate is run on the
whole task set, its attempts verified by the tasks' own verifiers, and compared
with the baseline's attempts, a family counted once. A candidate is promoted
only if no family got worse and the one-sided sign test over families is below
`--alpha`; one that regressed a family is rejected whatever its totals, and one
that merely did not regress is inconclusive. The first evaluation is the
baseline against its own arm, a replicate that shows how far two identical runs
differ. A promotion stores the candidate document with its evidence and the
experience sets behind it and moves the `harness/champion` pointer, whose
history is the lineage of champions. `--search-seed` has no default: the same
seed draws the same candidates.

`--proposer model --proposer-model REF` has a model propose the candidates instead of
the mutation space: it is shown the parent document and how that document did when
last evaluated, and answers with edits (a path into the YAML and the value to set).
An edit counts only once the document it makes compiles under sven's compiler; a reply
that does not compile is sent back with the compiler's message, up to
`--proposer-repairs`, and a proposal never fixed is replaced by a mutation of `--space`
when there is one (counted in the report) and stops the run when there is not.
`--parallel N` evaluates N candidates at once (best-of-N only; annealing's next
candidate depends on the last): each is a whole arm, so the model server sees N arms'
attempts together on top of `--remote-concurrency`.
