<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 130. A base exam measured under another job's load scored 0.854; the same base alone scored 0.938

Accounting eval split (80 tasks, 2 attempts, partial-credit reward, Qwen3.8-27B-FP8 through sven, four brain
servers). The first base run shared the servers with a long coding-benchmark arm and read 0.854 [0.786, 0.913].
A replicate on an idle server read 0.938 [0.899, 0.968]; paired against the first run it is +0.083
[+0.033, +0.144], 11W-2L-67T, the size of a real effect, with no change to model, tasks or sven. Two adapters
trained by policy gradient then read 0.924 and 0.932 against the original base (+0.069, +0.078, both CIs
excluding 0) and -0.014 [-0.039, +0.009] and -0.005 [-0.028, +0.015] against the idle replicate.

What this establishes: a paired comparison is only a measurement of the adapter if the base arm ran under the
same load as the adapter arm; a base that ran slow-and-contended manufactures a positive "effect". What it does
not: the mechanism (step timeouts, truncated tool turns) was not isolated. Rule: every learning claim carries a
base replicate taken at the same time and load as the trained arm, and the report prints both pairings.
