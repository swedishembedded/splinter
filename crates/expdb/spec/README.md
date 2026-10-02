<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Model checking the storage protocol

`ExpDB.tla` is a TLA+ model of the protocol `splinter-expdb` uses to keep its
files consistent when many processes write, compact, collect and pin at once.
It is checked with TLC. The model found real errors in the design, each fixed
in the code and pinned by a spec in `../tests/hardening.rs`.

## What is modelled

- **Objects** (segments, blob packs, index runs) are the set of records they
  hold, so they are content-addressed: equal contents are the same file.
- **Manifests** add and remove files and name their parents. A manifest is its
  own content, so identical manifests are one file. A removal is permanent: a
  file is live when it is added by some ancestor and removed by none. A
  checkpoint ends the walk because it carries the whole state.
- **Writers** seal a segment, then publish a manifest on a ref of their own.
  They can crash in between.
- **Compactors** (several, overlapping) read the live set, write the union of
  some inputs and publish a manifest that adds it and removes the inputs.
- **The catalog** is a set of immutable head files, merged and trimmed.
  **Absorbing** retires an idle writer ref, folds its head into the catalog,
  and discards it. **Checkpoints** squash the history.
- **Pins** name a snapshot and are validated after they are written.
- **Time**: a file leaves its grace period at some point.
- **Collection** reads the refs and pins, sets candidates aside, reads them
  again, restores what is needed, and only then discards.

Safety is stated for when no collection is mid-way (a needed file may be
briefly set aside and is restored before the collection ends):

| Invariant | Meaning |
|---|---|
| `ManifestsExist` | every manifest reachable from a ref exists |
| `ObjectsExist` | every file live under the refs exists |
| `NoLostRecord` | every record a writer published is still in a live file |
| `PinnedReadable` | a validated pin keeps its whole snapshot readable |
| `NoSelfRemoval` | no manifest adds and removes the same file |

## Assumptions the model makes explicit

`GraceHolds` is an assumption about time, not about code: the grace period is
longer than the time between a writer sealing a file and publishing it, longer
than one collection, and longer than the gap between a process writing a merge
manifest and pointing a ref at it. Without it, TLC finds a writer whose sealed
file is collected before it is published (`Core_noGrace`).

Each job ref has a single publisher; maintenance tasks publish on refs of
their own. Several publishers on one ref name can overwrite each other's head,
which is why the code gives every writer and every maintenance run its own.

## Results

Each safeguard of the implementation is a switch. With every switch on the
model is safe over the whole bounded state space; with one off, TLC returns a
counterexample.

| Config | Switch off | Outcome |
|---|---|---|
| `Core` | none: writers, two compactors, time and collection, two records | no error, 5,409 states |
| `Core_noTouch` | rewriting an existing file does not restart its grace period | violated: a compaction crashes and leaves an orphan, it ages, the same compaction runs again and finds the file already there, a collection removes it, the compaction publishes it |
| `Core_noGrace` | `GraceHolds` | violated: a sealed file ages and is collected before its writer publishes |
| `Compaction` | none: three records, two overlapping compactors, three compactions, no collection | no error, 8,733 states |
| `Compaction_noExclude` | a merge that equals one of its inputs still removes it | violated: two overlapping compactions leave a segment and one of its subsets live; merging them reproduces the larger, which is then removed with its inputs, and every record is lost |
| `Pins` | none | no error, 12,615 states |
| `Pins_noTrash` | collection deletes without setting aside and re-reading | violated: a pin is validated while a collection that began earlier is running, and the collection deletes a manifest the pin needs |
| `Checkpoint` | none | no error, 240,987 states |
| `Absorb` | none (catalog as a set of heads) | no error, 3,127,085 states |
| `Absorb_unsafe` | refs are checked, then deleted later | violated: a writer publishes between the check and the delete and its newest manifest is lost |
| `Full` | none: every feature together, two records, two writers, a compactor, at most five manifests | no error, 20,163,841 states |

`Compaction` and `Full` need `-deadlock`, because they reach states with
nothing left to do.

An earlier design overwrote the catalog ref when absorbing. TLC found that two
processes folding at once could drop history that a pruned ref had been the
only other copy of; the catalog is now a set of head files that is never
overwritten.

## Running it

TLC needs a Java runtime and `tla2tools.jar` from the TLA+ releases (checked
with version 1.8.0):

```text
java -XX:+UseParallelGC -cp tla2tools.jar tlc2.TLC -deadlock -workers 8 -config Core.cfg ExpDB.tla
```

The configurations differ only in the constants. The bounds (two or three
records, two writers, two compactors, at most a handful of manifests) are small
by design: the errors above need at most three records and two compactors, and
a clean run means no violation within those bounds, not for all sizes.

## What is not modelled

Readers other than pins, the contents of records, edges and indexes (which
are projections rebuilt from segments), blob chunking, file systems that
reorder metadata operations across a power loss, and processes that are slow
for longer than the grace period. The model is of the protocol; the code is
checked against it by the specs under `../tests`.
