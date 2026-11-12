<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 115. Plain DPO collapses a small model without an anchor on the chosen answer

Preference optimisation lowers the likelihood of both answers of a pair as readily as it separates
them. On the toy checkpoint (a three-letter cipher, held-out exact match, CPU), plain DPO collapsed
the model to 0.01 within a hundred steps; the same pairs with an anchor term on the chosen answer
took the held-out score from 0.39 to 0.57 over 3 on-policy rounds.

Rule: run DPO with the chosen-answer anchor (the smoke does); a DPO number without it says little
about the method. Not measured on a real model. The smoke is `make test/methods` (splinter bfc2d17fa, `tests/methods/dpo.rs`).
