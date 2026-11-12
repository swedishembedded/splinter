<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 128. A paraphrase chosen by document position never fired with few units

The hard world paraphrased the odd-numbered live chunk of each topic. With 6 units and half of them retired there is one live chunk (index 0), so the first real-model runs with `--units 6` contained no paraphrase at all and their "hard" numbers (selective recall 1.00) were an easy world's. Noticed because lexical and fused selection gave identical numbers. Fix: paraphrase by a seeded coin per document, and 8 units for the real runs. Rule: when two variants of a method give identical numbers, check that the feature separating them was present in the data. Evidence: splinter 230c14039; runs moved to `hard-v1-seed2000-uncalibrated`.
