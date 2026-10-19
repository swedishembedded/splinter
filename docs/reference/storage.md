<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Storage reference

Where Splinter keeps what it knows, how to keep it small, and how to back it up and recover it. The state
root is `~/.splinter` by default (`--state DIR` or `SPLINTER_STATE` change it); see
[configuration](../getting-started/configuration.md).

## Runs


Every command that writes pipeline state records a run in the experience
database: the command and its arguments, each stage as it finishes, its
status (`running`, `completed`, `failed`, `cancelled`) and what it
produced. `runs cancel <ID>` asks a run in progress to stop, by raising a
signal any process can see: it stops the model run in progress and the stage
at its next check, and training at its next optimizer step. A run whose
process died stays `running`.

## State

Everything Splinter knows lives in one experience database under
`<state>/expdb`: sources, tasks, experiences, annotations, sets, runs, and the
metadata of datasets, candidates, releases, answers, suites, calibrations and
the curriculum queue. The mutable names (an alias, the anchor suite in force)
are pointers whose whole history is kept; two processes moving one from the same
value cannot both win. Every experience is also an attempt in the database's
graph, a verdict is evidence about that attempt, and a dataset, a training run
and a release record where they came from, so `lineage` traces a release back to
what it learned from.

The bulk files a tool needs as real files (adapters, dataset JSONL) are kept
under `<state>/artifacts/<aa>/<digest>` and tracked by the database: a file is
written first, then one commit makes it official, so a crash leaves at most an
unrecorded file that `state maintain --collect` removes. `<state>/work` holds
scratch that exists only while a command runs.

`state status` reports what the database holds as files and what was written
off: `{"state", "storage": {"segments", "blob_packs", "index_runs", "pins",
"history", "artifacts", "losses"}, "losses", "pins"}`. A dataset keeps the
database as it was alive under a name (`pins` lists them); `state unpin <holder>`
lets one go so `state maintain --collect` can free its files. Every commit leaves small files
behind, and a command that opens the database reads them all, so `state
maintain` merges them, indexes what is not indexed and retires finished
writers; nothing stored changes, and it is safe beside a running command.
`state maintain --collect` also deletes files nothing reaches that are past
their grace period (a snapshot a dataset pinned is never touched) and artifact
files no commit made official. Its report is `{"run", "before", "after",
"segment_groups", "blob_groups", "writers_retired", "removed",
"orphan_artifacts"}` with the last two `null` unless `--collect` was given.
State written before the experience database is not read: regenerate it.

### Archive and recovery

`state archive FILE` packs the database and the artifacts it tracks into one
`.tar.zst`: the same state always gives the same bytes. `--no-artifacts` leaves
the files out, and `--since PREV` carries only what an earlier archive lacks (such an
archive restores together with it). `state restore FILE [BASE...]` unpacks into
an empty state root, checks every member against its digest, verifies the whole
state deeply next to the root and moves it into place only if it passes; a
restore that fails leaves the root as it was.

`state verify [--deep]` reports every missing or damaged file of the database and
of the artifacts without stopping at the first, and exits 1 if there is
one. `state repair [--from PATH]... [--accept-loss]` fills holes from copies (an
archive, or another state root): every file is named by the hash of its bytes, so
a copy is used only if it is exactly right. Index files are rebuilt from the
records. What no copy has stays reported unless `--accept-loss` is given, which
withdraws damaged database files and writes lost artifacts off in a ledger that
`state status` lists; a lost artifact is from then on refused with its digest
wherever it would have been used.

## Related

- The commands are `state status`, `state maintain`, `state unpin`, `state verify`, `state repair`,
  `state archive` and `state restore` in the [command reference](cli.md#splinter-state).
- The JSON each prints is in [Output and exit status](output.md).
- The experience database is the `splinter-expdb` crate; its specification is in the repository under
  `crates/expdb`.
