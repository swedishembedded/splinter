<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 131. Bench-ingested rollouts carry no behaviour log-probabilities; policy-gradient training needs --behaviour start

Training the `policy-gradient` dataset built from `ingest-bench` trials on the 27B through a served job refused:
"77997 of the 77997 supervised tokens of the records with an advantage have no recorded behaviour
log-probability, so the ratio would not be taken against the policy that sampled them". The benchmark's trial
artifacts hold the request log and the trajectory, not the sampler's log-probabilities. `train --behaviour start`
takes the start-of-run policy as the behaviour proxy; with it a 40-step run on 368 records (24 graded groups)
completed, ratio 1.000 to 1.007, clip fraction 0.003 to 0.014, KL to the reference 0.0008 to 0.0038.

What this establishes: the proxy is exact only for the first step of a run on rollouts from the starting policy;
later steps are slightly off-policy by construction (ratio drift is the measured size of it above). What it does
not: that recorded log-probabilities would change the outcome. Recording them with the rollouts (sven request log
or brain reply) is the open fix.
