<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Provenance

Every artifact Splinter makes can be traced back to the bytes it came from, and forward to what
was made from it. That is what lets you ask where a released model's knowledge came from, and
what to do when a source turns out to be wrong.

## Identity

Everything is addressed by its content: a source and its parts, a task, an experience, a dataset,
a candidate, a release, an answer. The same bytes are the same source, so intake is repeatable and
an id names the same thing wherever it is read. An id is the full `blake3:<hex>`, the hex alone, or a
prefix of at least four hex digits that names exactly one stored object.

A source records where it was captured: a file's path, a repository's revision, a command's argv
and exit code, a session's removals. A `--usage-policy` on `source add` records the terms the
content is under, and a release records the terms of everything it was made from.


## Lineage

`lineage <ID>` takes any artifact's id - a source, a source part's
content, a task or task set, an experience or experience set, a dataset, a
candidate, a replay sample, a release, an adapter digest, an answer - as
the full id, its hex, or a prefix naming exactly one; a prefix naming
artifacts in more than one store is refused with every one it names. It
prints two trees: where the artifact came from (`--up`), down to the source
part and byte range each task is grounded in with those bytes quoted, and
what came from it (`--down`): tasks, experiences, datasets, candidates,
releases and answers. Both by default; `--depth N` stops N edges out. The
graph is derived from what the stores already record, on every call:

| From | Relation | To |
|---|---|---|
| content | `part_of` | the source holding it as a part |
| span | `span_of` | the source it names a part of, else the content it indexes |
| task | `evidence` | each span it is grounded in |
| task | `taught_by` | the claim it was made from (`absorb`) |
| claim | `evidence` | each span of the person's words it rests on |
| task set, experience set | `member` | each task, experience |
| experience | `attempts`, `ran_in`, `solved_by` | its task, environment snapshot, solver model |
| experience | `critique_of`, `retry_of`, `revision_of`, `preferred_over`, `variant_of` | the experience its relation names |
| verdict | `verdict_on`, `produced_by` | the experience it grades, the verifier that gave it |
| dataset | `projected_from` | each experience, task and source content its manifest names |
| candidate, release | `trained_on`, `trained_from`, `replayed`, `rehearsed`, `adapter` | its datasets, parent release, replay sample, rehearsal dataset, adapter digest |
| replay sample | `sampled_from` | each earlier release it drew from |
| release | `release_of` | its candidate |
| answer | `answered_with`, `answered_by`, `open_book` | the release, the model, the source shown |

With `--json`: `{"nodes": [{"id", "kind", "label"}], "edges": [{"from",
"to", "relation"}]}`. The artifact asked about is the first node; every
node reached follows once, in walk order; an edge means `from` was derived
from `to`, whichever way it was walked. `kind` is one of `source`,
`content`, `span`, `claim`, `task`, `task_set`, `experience`, `experience_set`,
`environment`, `model`, `verdict`, `producer`, `dataset`, `candidate`,
`replay`, `release`, `adapter`, `answer`. Artifacts without an address of
their own have ids of their kind: `span:<content-hex>:<start>-<end>`,
`model:<identity>`, `verdict:<experience-hex>:<n>`,
`producer:<name>@<version>`.

## Answers are provenance too

`splinter ask` records every answer under the state with the model that gave it and, when asked
through `policy:<alias>`, the release the alias resolved to. An open-book or retrieved answer also
records the sources and passages it was shown, so `lineage` links an answer to the sources it
quoted. See [Learn from documents](../guides/documents.md#ask-the-learned-model).
