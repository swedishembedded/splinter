<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 234. A stage computed again in another run is not the product its downstream stages were keyed on

**Lesson:** a fork reuses a stage exactly when the product store holds a product under its stage identity, and a
stage's identity names its inputs by the digest of their product *manifests*. A manifest records when it was
committed (`created_at`), so the same stage computed in two runs (each pinning the tasks for itself) has one identity
but two products, and every stage downstream of it has a different identity in each run. Joining two such runs can
reuse the downstream stages of only the one whose upstream product was taken.

## How it showed (2026-11-27)

The Vim loop's generation-2 material was produced in a run of its own; merging that run's `products/*.json` and
`blobs/` into the lineage run did not make the next generation's fork reuse the material: the material's attempt
stage was computed again. The multi-source fork's test reproduced it: two runs that each pinned the same tasks and ran
one attempt stage, forked together, reused the first run's attempts and recomputed the second's, whose stage identity
differed only in the digest of its `tasks` input.

## What changed

`splinter fork RUN [RUN ...]` reuses the products of several runs (first match in the order given) and warns, per
source, about a stage it holds another product of than the one taken, since that source's downstream products cannot
be reused. The runs must share their upstream products: start a material run as a fork of a run the lineage descends
from. Spec: `crates/factory/tests/it/reuse.rs`,
`a_fork_of_several_runs_reuses_what_each_committed_first_match_first_and_computes_the_rest`.

Not changed: the manifest digest still includes `created_at`. Keying downstream identities on the product's files
alone would let independently computed runs join, but it changes every stage identity, so every existing run would be
recomputed by a fork.
