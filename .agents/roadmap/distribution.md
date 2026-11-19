<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Distribution: immutable bundles on the Hugging Face Hub

Scope: publishing experience sets, datasets, candidates and signed aggregate verdicts as immutable, versioned bundles of private Hugging Face dataset repositories (`splinter publish`, `splinter_pipelines::hub`, `splinter-hub`) and ingesting them back at a pinned revision (`splinter experiences ingest-hf`).

## Done (2026-11-19, lane H)

| Row | What | Evidence |
|---|---|---|
| D1 | `splinter-hub`: a client of the Hub's dataset-repository endpoints (create private, info, revision, refs, tag, paged recursive tree, verified streaming download, pre-upload, LFS batch with single-`PUT` and multipart transfers and verification, JSON-lines commit on a named parent); token only to the configured endpoint; no new third-party crate (reqwest, rustls, tokio, sha2, base64 already pinned in the lock) | unit specs in `crates/hub` (names, link paging) |
| D2 | `publish`: one tagged commit per bundle (`splinter-<20 hex of the contents' SHA-256>`), the previous revision's files replaced, every revision kept; republishing the same contents is a no-op; a candidate's datasets come along | `hub_bundle::a_published_bundle_ingests_back_under_the_same_ids_and_once` |
| D3 | Repository created private and its `private` flag read back before any upload; a missing flag or `false` refuses | `hub_bundle::a_repository_not_stated_private_is_refused_before_any_upload` (both quirks) |
| D4 | Typed `Withheld` screen before staging: `bench-private` paths (never opened), item files, files beside `sealed-export.json`, exam-split or eval-registered experiences, sealed families, unsigned / foreign-schema / item-level verdicts | `hub_bundle::sealed_and_item_level_material_is_refused_before_anything_is_sent` (the emulator sees no request) |
| D5 | `ingest-hf REPO@REVISION`: branch or tag resolved to its commit and recorded as it; manifest identity and tree sizes checked; every file hashed while downloaded; stores written only through their APIs, each under its content address (sets, experiences, datasets byte for byte, candidates, artifacts, verdicts re-verified); a `hub_ingest` provenance record; re-ingest stores nothing | `hub_bundle` round trip and `a_tampered_file_is_refused_and_nothing_of_the_bundle_is_stored` |
| D6 | Mutation check: dropping the privacy check or the republish no-op turns the privacy and round-trip specs red | run 2026-11-19 |

## Open

- [ ] No bundle has been published to the real Hub from this code: the LFS transfer and commit calls are verified
      against the local emulator only (the Hub's own client of version 2.4.0 was the reference for the wire format).
      The first real publish is the check.
- [ ] A candidate that continues a release is refused: a bundle does not carry release lineage (releases, aliases,
      replayed records) yet.
- [ ] A basic LFS transfer reads the whole file into memory; the Hub answers large files with a multipart transfer,
      whose parts are read one at a time.
- [ ] The manifest is not signed: a manifest rewritten consistently with its files is not detected; trust rests on
      the private repository and the pinned commit.
