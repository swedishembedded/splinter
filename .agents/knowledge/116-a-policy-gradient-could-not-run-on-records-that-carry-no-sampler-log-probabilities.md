<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 116. A policy gradient could not run on records that carry no sampler log-probabilities

brain refuses a policy-gradient run when a supervised token has no recorded behaviour log-probability,
and no record Splinter writes carries one, so `train` on a rollout dataset stopped on its first step
("1389 of the 1389 supervised tokens ... have no recorded behaviour log-probability"). Splinter never
passed brain's `start_of_run_proxy`, which is exact for rollouts sampled from the adapter the run starts
from. `PolicyGradientOptions::behaviour` and `--behaviour recorded|start` now carry it; on-policy rounds
use `start`, and step 0 of such a run has importance ratio exactly 1.000 (measured).

Rule: a run on rollouts drawn from a different policy than the one it starts from must not use the
proxy; the plan records which source was used. The smokes are `grpo.rs` and `selfplay.rs` in
`tests/methods`.
