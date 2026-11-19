<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 178. A store that writes a file per object runs out of inodes, not bytes

**Lesson:** on a filesystem with a file quota, the number of files a store makes is a resource as real as its
bytes, and a content-addressed store that writes one file per immutable object spends it per record. Count the files a
design makes per unit of work before it ships, and pack immutable objects into a few append-only files with an index.

Found 2026-11-19 (lane S). The experience database's loose layout (`splinter-expdb 2`) wrote every segment, blob
pack, manifest and signal as its own file, three or four per ingested experience: one lane state held 17,400 database
files (5,920 manifests, 5,773 segments, 2,997 signals, 2,695 blob packs) for about 1.9k experiences, and the shared
project disk's 500k-file quota hit 100 % several times in a day while its bytes were far from full. `state maintain`
merges segments but never ran on those states, and merging still leaves manifests and signals per commit.

The packed layout (`splinter-expdb 3`, `crates/expdb/src/backend/packed.rs`) keeps content addressing, write-once,
deduplication and the database format above the backend unchanged: only where an object's bytes lie changes. Objects
are appended to `packs/<n>.pack` (rolled at 1 GiB) and named by entries in the append-only `packs/journal`; every
change holds an exclusive `flock` on the journal and writes bytes, syncs, then appends and syncs the entry, so an
object exists once its entry is whole. Readers stop at the first entry that is not whole; the next writer cuts a torn
tail of the journal and unnamed bytes past the last named object of the newest pack, and refuses (rather than cuts)
damage followed by whole entries. Refs, pins and overlays stay files, so `refs/format` still names the layout and a
binary that knows only layout 2 refuses a packed root instead of writing loose files beside it.

Measured on a fixture of 2,000 experiences, each stored, graded, committed and signalled through `ExperienceStore`
(the `ingest` of `crates/store/tests/it/state_layout.rs`, run once at 2,000 on the project's Lustre disk): the loose
layout made 12,064 database files, the packed one 4; `state pack` rewrote the loose one into 4 files in 5.8 s (12,063
objects, 28 MB, every one checked against its name and read back). Ingesting was slower packed, 80 s against 65 s
(a lock and two syncs per object instead of one file, a link and a directory sync). Copies of two real lane-d states
packed from 432 files to 4 and from 5,445 to 5 (4.7 s), and listed the same experience sets and verified deep before
and after. The packed count does not grow with the experience count
(`a_new_state_keeps_a_bounded_number_of_database_files`). The same
bench export builds the same dataset id from a loose state, a packed one and the loose one after `state pack`
(`crates/pipelines/tests/it/state_layout.rs`).

Not done: a removed or collected object's bytes stay in its pack until `state pack` copies only live objects;
artifacts stay files because brain opens adapters and datasets by path, and their count follows datasets and adapters,
not experiences.
