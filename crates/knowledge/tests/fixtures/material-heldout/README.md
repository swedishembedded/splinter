<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Held-out mini corpus

Five fictional documents (a tenancy act chapter, its amendment, a tenant forum post, a clinic intake protocol, a behaviour sheet) and `gold.json`, the
objects a careful reader must extract from them, in pack form. It was written after the proposer's first brief and **before any model was run on it**,
and no brief, example or repair message may be taken from it: it measures whether an improvement made on `../material` holds on text the proposer was
not tuned on. `gold.json` for `../material` lists what that corpus's scripted proposer states.

Score an extracted pack with the consumer's `python -m compiled score PACK GOLD`: recall and precision per class (one-to-one matching by class,
document, quote overlap and numbers), and the tasks the families yield per document.
