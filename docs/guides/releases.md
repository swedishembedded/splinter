<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Release, serve and roll back

A candidate becomes the policy only through a release decision, and there are two places a
decision lives.

A recipe of the [learning factory](learning-factory.md) decides inside its run: its `evaluation`
stages measure each candidate, and its `promotion` stage records every decision and the gates each
candidate met and, with `promote: true`, promotes the best candidate that meets them all. The
decision is the run's release product (`release.json`, with the champion's adapter copied beside
it); a later recipe trains from it by naming `champion:RUN`
([learning factory](learning-factory.md)). It is not written to the release store below.

[Knowledge absorption](../reference/cli.md#splinter-absorb) (`absorb`, `consolidate`) releases through the release
store of the state root: its gate writes a release manifest and moves an alias, and `release list`
and `rollback` work on that store. The checks themselves are in the
[evaluation reference](../reference/evaluation.md#the-release-gate) and the reasoning in
[release policy](../concepts/release-policy.md).

```bash
splinter report runs/knowledge   # a factory run's evaluations and its release decision
splinter release list            # the release store: every release with its parent and aliases
splinter rollback default        # move an alias back (below)
```

A factory candidate that does not meet its gates is not promoted, and `report` names each gate it
failed. An absorption that does not pass its gate makes no release, and the alias stays.

## What a release is


A release is written once. Its manifest is kept in the experience database
(`<state>/expdb`) in canonical JSON - the base model and
its digest, the adapter digest, the parent release, the candidate, the new
datasets, the replay sample and the rehearsal with their digests, the training record
(brain's included), every gate number with the anchor suite's version and
digest, and `created_at` - and names the adapter file, which stays where the candidate
was kept, by content address under `<state>/artifacts/`. The manifest's
digest is the release id. An alias is a pointer in the same database that moves only
from the champion the gate measured against; `<state>/serving/<name>.json` is the manifest a
server follows for it, rewritten whenever it moves. `release
list` shows every release with its parent and aliases; `rollback <ALIAS>`
points the alias at the release its current one was trained from, and
refuses when there is none.

## Serve a release without Splinter

An adapter belongs to one base model. To serve a release without
Splinter, name that base - the manifest's `base_model` and `base_digest`
say which checkpoint it is - and the release's adapter, as the serve check
does:

```bash
brain serve --openai --checkpoint <base checkpoint> --adapter-manifest <state>/serving/<alias>.json
```

brain serves the adapter on that one Qwen3 base, as the model
`brain/qwen3`, and refuses to start when the base's digest is not the one
the adapter records.

To keep a running server on whatever an alias points at, give it the
manifest the release store rewrites whenever the alias moves (a release, a
rollback) instead of an adapter file:

```bash
brain serve --openai --checkpoint <base checkpoint> --adapter-manifest <state>/serving/default.json
```

The manifest names the adapter the alias points at and its digest, and is
replaced atomically; the server swaps onto the new adapter once the file
hashes to the digest and binds to its base, and keeps the one it has when it
does not. An alias at a full checkpoint names no adapter, so its manifest is
withdrawn.
