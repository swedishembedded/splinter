<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Storage reference

Where Splinter keeps what it knows, how to keep it small, and how to back it up and recover it. The state
root is `~/.splinter` by default (`--state DIR` changes it); see
[configuration](../getting-started/configuration.md).

## Runs


Every command that writes pipeline state records a run in the experience
database: the command and its arguments, each stage as it finishes, its
status (`running`, `completed`, `failed`, `cancelled`, `lost`) and what it
produced. `runs cancel <ID>` asks a run in progress to stop, by raising a
signal any process can see: it stops the model run in progress and the stage
at its next check, and training at its next optimizer step. A run whose
process died is reported `lost` once its heartbeat is stale (a run recorded
before heartbeats were kept has none and keeps `running`).

## Jobs

A job submitted with `jobs submit` is a content-addressed specification and a
record in the experience database under the pointer `job/<spec digest>`: its
state, attempt, launcher, attempt directory, exit code and the controller's
lease. The specification is the job's identity, so submitting the same one twice
names the same job and starts nothing.

Its workers run detached, each in its own session, in
`<state>/jobs/<digest>/attempt-<n>`: `launch.json` names the attempt and the
worker processes, `rank-<r>.out` and `rank-<r>.err` hold their output, and
`rank-<r>.exit` holds the exit code recorded after the worker finished, so a
process that did not start the job reads how it ended. A worker that outlives
the job's `--wall-clock` is stopped with its process group and recorded as 124.
`--job-attempt-root DIR` writes the attempt directories elsewhere.

A job submitted with `--distributed` runs under a generated plan: before any
worker starts, `job-plan.json` is written into its attempt directory, naming
the job by its specification digest, its world size and the coordinator every
rank registers with. Each worker is given `--job-plan job-plan.json` and its
own `--job-rank`, so every rank reads the same bytes, and a retry writes the
next attempt's plan with the same job identity.

`jobs list` shows each job's last recorded state, `jobs reconcile` records what
the launcher finds (completed, failed, running, or lost when nothing accounts
for the workers), `jobs retry` starts the next attempt where the retry policy
allows one, and `jobs cancel` stops the workers and records the job cancelled. Both `jobs list` and `jobs show` read the record only; nothing is
observed until `jobs reconcile` runs.

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

A continuous campaign (the campaign controller, `splinter_sdk::campaign`) keeps its
state and its experiment ledger in one append-only file, `<state>/campaigns/<name>.pack`:
a line per state, a small head line naming the current state, a line per ledger entry.
It is a single file however many arms the campaign runs, because the project disk's
file quota, not its size, is what a long run exhausts; saving a state is a
compare-and-swap on the head, so a second controller on the same campaign is refused.

`state status` reports what the database holds and what was written
off: `{"state", "storage": {"segments", "blob_packs", "index_runs", "pins",
"history", "artifacts", "losses"}, "losses", "pins"}`. A dataset keeps the
database as it was alive under a name (`pins` lists them); `state unpin <holder>`
lets one go so `state maintain --collect` can free its files. Every commit leaves small objects
behind, and a command that opens the database reads them all, so `state
maintain` merges them, indexes what is not indexed and retires finished
writers; nothing stored changes, and it is safe beside a running command.
`state maintain --collect` also deletes files nothing reaches that are past
their grace period (a snapshot a dataset pinned is never touched) and artifact
files no commit made official. Its report is `{"run", "before", "after",
"segment_groups", "blob_groups", "writers_retired", "removed",
"orphan_artifacts"}` with the last two `null` unless `--collect` was given.
State written before the experience database is not read: regenerate it.

### Layout

The database's objects (segments of records, blob packs, manifests, index
runs, signals) are immutable and named by the hash of their bytes. A state
root made by this version keeps them **packed**: appended to
`<state>/expdb/packs/<n>.pack` files of up to 1 GiB, with the append-only
`packs/journal` naming where each one lies. The few mutable pointers stay
small files: `refs/` (`refs/format` names the layout), `pins/` and
`overlay/`. A state therefore holds a handful of files whatever number of
experiences it holds, where the former **loose** layout kept one file per
object, three or four per ingested experience, which exhausts a
file-counting quota long before its bytes do.

Every write takes an exclusive lock on the journal, appends the object's
bytes to the newest pack and syncs them, then appends one journal entry
naming them and syncs it: an object exists once its entry is whole. A process
killed mid-write leaves bytes no entry names, or a torn last entry; readers
never see either and the next writer cuts both off. Writers in several
processes take turns on the lock; a filesystem without locks refuses the
write. Damage inside the journal is reported, never cut away. Removing or
collecting an object only records that it is gone; its bytes stay in the
pack until the state is packed again.

A state in the loose layout is read and written as it is. `state pack`
rewrites it into the packed layout: the new database is built beside the old
one (`expdb.packing`), every object is checked against its name and read
back, and only then is it moved into place and the old one deleted. A pack
that dies before the move leaves the old database in use; one that dies
during it is finished by the next command that opens the state. Stop every
process using the state first: a write to it while it is copied is detected
and the pack refused. On a packed state, `state pack` compacts: only live
objects are copied. A binary older than the packed layout refuses a packed
state instead of writing beside it.

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
  `state archive`, `state restore` and `state pack` in the [command reference](cli.md#splinter-state).
- The JSON each prints is in [Output and exit status](output.md).
- The experience database is the `splinter-expdb` crate; its specification is in the repository under
  `crates/expdb`.
