<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Protocol, fixed before the first measured run (2026-10-13)

Question: can a local agentic model be tuned to a git repository, repeatedly and
cumulatively, so that it resolves more of that repository's reserved task families than
the same base model under the same harness, without losing earlier repositories.

Frozen rules:

1. Unit: the family (tasks derived from one mutated path or source entity). Families are
   assigned to training or exam before any rollout used for data.
2. Arms: base and adapter, differing only in the attached adapter. Same harness profile
   (agent revision, system prompt, tool schemas, budgets, sampling), same seeds per
   (task, attempt), same environment image.
3. Primary decision: one-sided exact sign test over non-tied exam families, alpha 0.05.
   Not significant means "not shown", never equivalence. Fewer than 5 non-tied families
   cannot decide.
4. Retention: one-sided 95% lower bound (family-cluster bootstrap) on the change in
   resolve rate on earlier repositories' exam families exceeds minus 0.05 absolute.
5. Exclusions: environment failures are excluded from denominators and counted;
   exposed experiences are excluded from training and gates and counted; no other
   exclusion without a dated amendment.
6. Controls: A1 trains on all trajectories regardless of verdict; A2 omits replay; E3
   off-diagonal cells separate repository learning from protocol learning.
7. Anything computed under a rule changed after the first measured run is post hoc.

Known limits: isolation level none in the current environment; mutation-seeded tasks only
until historical-fix tasks exist; single harness; exam size per repository may be
underpowered.
