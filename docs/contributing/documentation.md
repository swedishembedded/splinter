<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Documentation

This manual is the `docs/` tree. `docs/manifest.yaml` lists every published page and is the one source of order for the
website sidebar and the PDF. The sample and example READMEs that stay next to their code are included from there. The
internal notes, roadmaps and papers are not published.

## Targets

```bash
make docs          # = docs/build
make docs/build    # build the publication into build/docs (no Rust build, no network)
make docs/check    # build twice and compare, check links and coverage, check the generated reference and every command shown
make docs/pdf      # the manual as one PDF, in manifest order (needs pandoc and xelatex)
make docs/clean    # remove the generated documentation
```

The publisher is the pinned `tools/docs-publisher/docs-publisher.mjs`, verified against its committed sha256 before it runs
(override it with `DOCS_PUBLISHER=<path>` while developing against a newer one). It needs Node 20 or later. Its rules for the
manifest, links and assets are in the README of `tools/docs-publisher` in the website repository.

## Rules for pages

- Every page starts with the two-line CC-BY-4.0 SPDX and copyright comments, then a single `#` title.
- Link other pages by relative path; the publisher rewrites them and fails on a broken one or a missing anchor. Link the
  sibling projects by their canonical roots, `https://swedishembedded.com/docs/sven` and `https://swedishembedded.com/docs/brain`.
  Code, sample and crate READMEs link to a manual page by its URL, `https://swedishembedded.com/docs/splinter/<route>`, because
  the no-doc-citations gate keeps code from naming `docs/` paths; `docs/check` verifies every such URL names a published route.
- A command in a `bash` block must parse against the real command-line grammar. Write a placeholder as a shell variable
  (`$CANDIDATE`), not as `<ID>`.
- State what a command does, then its options by name; do not repeat the reference. A statement about behavior is sourced from
  the code or a test, and a number is not published without a measurement that backs it.

## Generated reference

`docs/reference/cli.md` is generated from the `clap` grammar and must not be edited:

```bash
splinter cli-reference > docs/reference/cli.md
splinter cli-reference --check docs/reference/cli.md
splinter cli-reference --check-invocations docs/getting-started/first-learning-run.md
```

`make docs/check` runs the last two with the built binary (`SPLINTER_BIN`, default `target/release/splinter`; build it with
`make build/crate/splinter` first). `--check-invocations` takes Markdown files and fails, naming file and line, on every
`splinter` command in a `bash`, `sh`, `shell` or `console` block that the grammar rejects.

## The first-run walk-through is a test

`getting-started/first-learning-run.md` is read by the `first_run` integration test of the `splinter` crate, which runs the
commands the page shows against an empty home and asserts the page quotes what they print. Change the page and the test
together.
