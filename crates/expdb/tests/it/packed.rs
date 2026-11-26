// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in keeping
// file counts bounded on quota-limited parallel filesystems, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: a new database packs its objects, so the files it holds do not grow
//! with the experience it holds; it names, deduplicates and reads objects
//! exactly as the loose layout does; a crash mid-append leaves it readable;
//! concurrent writers take turns; and a loose database repacks into a packed
//! one that is the same database.
#![allow(clippy::unwrap_used)]

use crate::common;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use common::Scratch;
use splinter_expdb::backend::{
    repack, Key, Kind, Layout, PackedBackend, PosixBackend, StorageBackend,
};
use splinter_expdb::model::{Body, Entity, Skill};
use splinter_expdb::{Config, Database, StepClock, WriterIdentity};

fn skill(name: String) -> Body {
    Body::Skill(Skill {
        name,
        description: String::new(),
        trigger: String::new(),
        action_pattern: String::new(),
        expected_effect: String::new(),
        parents: vec![],
        prerequisites: vec![],
    })
}

/// Bytes no chunker can deduplicate against another experience's.
fn noise(len: usize, seed: u64) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect()
}

/// Experiences `from..to` as one ingesting process writes them: each its own
/// records, a blob and a commit, and a signal that it is done.
fn ingest(db: &Database, from: u64, to: u64) {
    let mut c = db
        .collector(&WriterIdentity::new("exp", "ingest", "node", 0))
        .unwrap();
    for n in from..to {
        c.put_blob(&noise(20_000, n)).unwrap();
        c.record(skill(format!("experience {n}"))).unwrap();
        c.record(skill(format!("step of {n}"))).unwrap();
        c.flush().unwrap();
        db.signal(&format!("done/{n}"), "ok").unwrap();
    }
    // A finished writer hands its history to the catalog, as a workspace
    // does when it closes, so no ref per process is left behind.
    db.retire_job(c.job()).unwrap();
}

fn files_under(root: &Path) -> usize {
    let mut count = 0;
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                pending.push(entry.path());
            } else {
                count += 1;
            }
        }
    }
    count
}

#[test]
fn a_new_database_holds_a_bounded_number_of_files_however_much_is_ingested() {
    let scratch = Scratch::new();
    let db = scratch.open();
    assert_eq!(db.backend().layout(), Layout::Packed);
    ingest(&db, 0, 40);
    let after_few = files_under(scratch.dir.path());
    ingest(&db, 40, 160);
    let after_many = files_under(scratch.dir.path());
    assert_eq!(
        db.snapshot().unwrap().records().unwrap().len(),
        320,
        "everything ingested is there"
    );
    assert_eq!(
        after_few, after_many,
        "four times the experience, the same files"
    );
    assert!(after_many <= 8, "{after_many} files");

    let loose = Scratch::new();
    ingest(&loose.open_loose(), 0, 40);
    assert!(
        files_under(loose.dir.path()) > 40 * 3,
        "the loose layout pays files per experience, which is the point"
    );
}

#[test]
fn packs_roll_over_at_their_size_and_every_object_still_reads() {
    let scratch = Scratch::new();
    let backend = PackedBackend::with_pack_file_bytes(scratch.dir.path(), 64 * 1024);
    let mut written = BTreeMap::new();
    for n in 0..40u64 {
        let bytes = noise(10_000, n);
        let key = Key::new(Kind::BlobPack, &format!("{n:04}.pack")).unwrap();
        assert!(backend.write_once(&key, &bytes).unwrap());
        written.insert(key, bytes);
    }
    let packs = backend.pack_files().unwrap();
    assert!(
        (6..=9).contains(&packs.len()),
        "about 400 kB in 64 kB packs plus the journal: {packs:?}"
    );
    let reopened = PackedBackend::new(scratch.dir.path());
    for (key, bytes) in &written {
        assert_eq!(&reopened.read(key).unwrap(), bytes);
        assert_eq!(
            &reopened.read_range(key, 100, 50).unwrap(),
            &bytes[100..150]
        );
    }
    assert!(reopened
        .read_range(written.keys().next().unwrap(), 9_990, 20)
        .is_err());
}

/// The same inputs get the same content ids in either layout, and read back
/// as the same bytes: addressing is the format's, not the layout's.
#[test]
fn both_layouts_name_the_same_inputs_with_the_same_ids() {
    let ids = |layout: Layout| {
        let scratch = Scratch::new();
        let db = Database::open_as(scratch.dir.path(), layout, Config::default()).unwrap();
        let mut c = db
            .collector(&WriterIdentity::new("exp", "ingest", "node", 0))
            .unwrap();
        let mut ids = Vec::new();
        for n in 0..12 {
            let blob = c.put_blob(&noise(30_000, n)).unwrap();
            let entity = Entity::new("experience", serde_json::json!({ "n": n }));
            ids.push((blob.id, c.put_entity(&entity).unwrap()));
        }
        c.flush().unwrap();
        for (n, (blob, _)) in ids.iter().enumerate() {
            assert_eq!(c.read_blob(blob).unwrap(), noise(30_000, n as u64));
        }
        ids
    };
    assert_eq!(ids(Layout::Packed), ids(Layout::Loose));
}

#[test]
fn writing_what_is_there_again_adds_nothing_and_says_so() {
    let scratch = Scratch::new();
    let db = scratch.open();
    ingest(&db, 0, 5);
    let backend = db.backend();
    let segments = backend.list(Kind::Segment).unwrap();
    let packs_before: Vec<u64> = pack_sizes(scratch.dir.path());
    for key in &segments {
        let bytes = backend.read(key).unwrap();
        assert!(!backend.write_once(key, &bytes).unwrap(), "already there");
    }
    assert_eq!(
        pack_sizes(scratch.dir.path()),
        packs_before,
        "no byte twice"
    );
    assert_eq!(backend.list(Kind::Segment).unwrap(), segments);
    assert!(
        !db.signal("done/0", "again").unwrap(),
        "a signal is raised once"
    );
    assert_eq!(db.signal_note("done/0").unwrap().unwrap(), "ok");
}

fn pack_sizes(root: &Path) -> Vec<u64> {
    let mut sizes: Vec<(String, u64)> = std::fs::read_dir(root.join("packs"))
        .unwrap()
        .map(|e| e.unwrap())
        .filter(|e| e.file_name().to_string_lossy().ends_with(".pack"))
        .map(|e| {
            (
                e.file_name().to_string_lossy().into_owned(),
                e.metadata().unwrap().len(),
            )
        })
        .collect();
    sizes.sort();
    sizes.into_iter().map(|(_, len)| len).collect()
}

/// A process killed mid-append leaves bytes no entry names in the pack and a
/// torn entry at the end of the journal. Both are invisible to readers and
/// cut off by the next writer, and nothing written before is lost.
#[test]
fn a_crash_mid_append_leaves_a_readable_store_that_the_next_writer_repairs() {
    let scratch = Scratch::new();
    let db = scratch.open();
    ingest(&db, 0, 6);
    let records = db.snapshot().unwrap().records().unwrap().len();
    drop(db);

    let packs = scratch.dir.path().join("packs");
    let journal = packs.join("journal");
    let pack = packs.join("00000000.pack");
    let (journal_len, pack_len) = (
        std::fs::metadata(&journal).unwrap().len(),
        std::fs::metadata(&pack).unwrap().len(),
    );
    let append = |path: &Path, bytes: &[u8]| {
        use std::io::Write;
        std::fs::OpenOptions::new()
            .append(true)
            .open(path)
            .unwrap()
            .write_all(bytes)
            .unwrap();
    };
    append(&pack, &noise(5_000, 99));
    // The first bytes of an entry: a length that promises more than follows.
    append(&journal, &[60, 0, 0, 0, 1, 1, 7]);

    let reader = scratch.open();
    assert_eq!(
        reader.snapshot().unwrap().records().unwrap().len(),
        records,
        "a reader sees every whole entry and nothing torn"
    );
    ingest(&reader, 6, 8);
    assert_eq!(
        reader.snapshot().unwrap().records().unwrap().len(),
        records + 4
    );
    let reopened = scratch.open();
    assert_eq!(
        reopened.snapshot().unwrap().records().unwrap().len(),
        records + 4,
        "what was written after the crash is there for every later open"
    );
    assert!(reopened.verify(true).unwrap().problems.is_empty());
    assert!(std::fs::metadata(&pack).unwrap().len() > pack_len);
    assert!(std::fs::metadata(&journal).unwrap().len() > journal_len);
}

/// Damage in the middle of the journal is not a torn write: the next writer
/// refuses rather than cut away the entries after it.
#[test]
fn damage_inside_the_journal_is_refused_not_cut_away() {
    let scratch = Scratch::new();
    let db = scratch.open();
    ingest(&db, 0, 3);
    drop(db);
    let journal = scratch.dir.path().join("packs/journal");
    let mut bytes = std::fs::read(&journal).unwrap();
    bytes[30] ^= 0xff;
    std::fs::write(&journal, &bytes).unwrap();
    let damaged = PackedBackend::new(scratch.dir.path());
    let key = Key::new(Kind::Signal, "after").unwrap();
    assert!(damaged.write_once(&key, b"x").is_err());
    assert_eq!(std::fs::read(&journal).unwrap(), bytes, "nothing was cut");
}

/// Two handles on one root, as two processes are, never interleave their
/// appends: one waits while the other holds the journal's lock, and every
/// object either wrote is there afterwards.
#[test]
fn concurrent_writers_take_turns_on_the_journal_lock() {
    let scratch = Scratch::new();
    let root = scratch.dir.path().to_path_buf();
    let first = PackedBackend::new(&root);
    first
        .write_once(&Key::new(Kind::Signal, "start").unwrap(), b"")
        .unwrap();

    // Another process holds the lock.
    let held = std::fs::File::open(root.join("packs/journal")).unwrap();
    held.lock().unwrap();
    let (sent, done) = mpsc::channel();
    let writer = {
        let root = root.clone();
        thread::spawn(move || {
            let second = PackedBackend::new(&root);
            second
                .write_once(&Key::new(Kind::Signal, "waited").unwrap(), b"w")
                .unwrap();
            sent.send(()).unwrap();
        })
    };
    assert!(
        done.recv_timeout(Duration::from_millis(300)).is_err(),
        "the second writer waits for the lock"
    );
    held.unlock().unwrap();
    done.recv_timeout(Duration::from_secs(30)).unwrap();
    writer.join().unwrap();

    let writers: Vec<_> = (0..8u64)
        .map(|w| {
            let root = root.clone();
            thread::spawn(move || {
                let backend = PackedBackend::new(&root);
                let mut wins = 0;
                for n in 0..25u64 {
                    let own = Key::new(Kind::Segment, &format!("w{w}-{n}.seg")).unwrap();
                    backend
                        .write_once(&own, &noise(3_000, w * 100 + n))
                        .unwrap();
                    let shared = Key::new(Kind::Signal, &format!("shared/{n}")).unwrap();
                    wins += usize::from(backend.write_once(&shared, &[w as u8]).unwrap());
                }
                wins
            })
        })
        .collect();
    let wins: usize = writers.into_iter().map(|h| h.join().unwrap()).sum();
    assert_eq!(wins, 25, "exactly one writer raised each shared signal");
    let reader = PackedBackend::new(&root);
    assert_eq!(reader.list(Kind::Segment).unwrap().len(), 200);
    for w in 0..8u64 {
        for n in 0..25u64 {
            let key = Key::new(Kind::Segment, &format!("w{w}-{n}.seg")).unwrap();
            assert_eq!(reader.read(&key).unwrap(), noise(3_000, w * 100 + n));
        }
    }
}

/// A loose database repacks into a packed one that is the same database:
/// the same snapshot, records, signals and object bytes, and the same ages
/// for collection to judge by.
#[test]
fn a_loose_database_repacks_into_the_same_database_in_a_few_files() {
    let scratch = Scratch::new();
    let loose = scratch.open_loose();
    ingest(&loose, 0, 30);
    let snapshot = loose.snapshot().unwrap().persist().unwrap();
    let records = loose.snapshot().unwrap().records().unwrap();
    let target = tempfile::TempDir::new().unwrap();
    let packed_root = target.path().join("expdb");

    let report = repack(scratch.dir.path(), &packed_root).unwrap();
    assert_eq!(
        report.files_before as usize,
        files_under(scratch.dir.path())
    );
    assert!(report.files_after <= 8, "{report:?}");
    assert!(report.files_before > 90, "{report:?}");

    let packed = Database::open(&packed_root, Config::default()).unwrap();
    assert_eq!(packed.backend().layout(), Layout::Packed);
    assert_eq!(packed.snapshot().unwrap().persist().unwrap(), snapshot);
    assert_eq!(packed.snapshot().unwrap().records().unwrap(), records);
    assert_eq!(packed.signals("done/").unwrap().len(), 30);
    let source = PosixBackend::new(scratch.dir.path());
    for kind in [Kind::Segment, Kind::BlobPack, Kind::Manifest] {
        for key in source.list(kind).unwrap() {
            assert_eq!(
                packed.backend().read(&key).unwrap(),
                source.read(&key).unwrap()
            );
            assert_eq!(
                packed.backend().modified(&key).unwrap(),
                source.modified(&key).unwrap()
            );
        }
    }
    assert!(packed.verify(true).unwrap().problems.is_empty());

    // Packing a packed database again copies it whole, live objects only.
    let again = target.path().join("again");
    repack(&packed_root, &again).unwrap();
    assert_eq!(
        Database::open(&again, Config::default())
            .unwrap()
            .snapshot()
            .unwrap()
            .records()
            .unwrap(),
        records
    );
}

#[test]
fn a_repack_refuses_an_object_that_does_not_hash_to_its_name() {
    let scratch = Scratch::new();
    ingest(&scratch.open_loose(), 0, 3);
    let source = PosixBackend::new(scratch.dir.path());
    let victim = source.list(Kind::Segment).unwrap().remove(0);
    let path = scratch.dir.path().join(victim.relative_path());
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[20] ^= 0x01;
    std::fs::write(&path, bytes).unwrap();
    let target = tempfile::TempDir::new().unwrap();
    let err = repack(scratch.dir.path(), &target.path().join("expdb")).unwrap_err();
    assert!(err.to_string().contains("hash to its name"), "{err}");
}

/// A binary that knows only the loose layout must not mistake a packed root
/// for an empty loose one and write beside the packs.
#[test]
fn a_packed_root_is_refused_by_the_loose_layout_and_the_reverse() {
    let packed = Scratch::new();
    packed.open();
    assert!(Database::with_backend(
        Arc::new(PosixBackend::new(packed.dir.path())),
        Config::default(),
        Arc::new(StepClock::new(0, 1)),
    )
    .is_err());
    let loose = Scratch::new();
    loose.open_loose();
    assert!(Database::with_backend(
        Arc::new(PackedBackend::new(loose.dir.path())),
        Config::default(),
        Arc::new(StepClock::new(0, 1)),
    )
    .is_err());
    assert_eq!(
        loose.open().backend().layout(),
        Layout::Loose,
        "an existing root keeps its layout"
    );
}

/// Damage to a pack's bytes is found by a deep check, as it is in a loose
/// file, and a hole filled from a copy reads as the copy's bytes.
#[test]
fn damage_in_a_pack_is_found_deeply_and_filled_from_a_copy() {
    let scratch = Scratch::new();
    let db = scratch.open();
    ingest(&db, 0, 4);
    let copy = Scratch::new();
    let copied = repack(scratch.dir.path(), &copy.dir.path().join("expdb")).unwrap();
    assert!(copied.objects > 0);
    drop(db);

    let pack = scratch.dir.path().join("packs/00000000.pack");
    let mut bytes = std::fs::read(&pack).unwrap();
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0x01;
    std::fs::write(&pack, bytes).unwrap();

    let db = scratch.open();
    assert!(
        db.verify(false).unwrap().problems.is_empty(),
        "sizes still match"
    );
    let report = db.verify(true).unwrap();
    assert_eq!(report.problems.len(), 1, "{report:?}");
    let problem = &report.problems[0];
    let source = Database::open(copy.dir.path().join("expdb"), Config::default()).unwrap();
    let key = problem.object.as_ref().unwrap().key().unwrap();
    let good = source.backend().read(&key).unwrap();
    assert!(db.fill(&problem.path, &good).unwrap());
    assert!(db.verify(true).unwrap().problems.is_empty());
    assert_eq!(db.snapshot().unwrap().records().unwrap().len(), 8);
}

/// Make every file and directory under `root` read-only (`writable: false`)
/// or writable again, as a committed product directory is made read-only.
fn set_writable(root: &Path, writable: bool) {
    use std::os::unix::fs::PermissionsExt;
    let mut pending = vec![root.to_path_buf()];
    let mut dirs = Vec::new();
    while let Some(path) = pending.pop() {
        let meta = std::fs::symlink_metadata(&path).unwrap();
        if meta.is_dir() {
            pending.extend(std::fs::read_dir(&path).unwrap().map(|e| e.unwrap().path()));
            dirs.push(path);
        } else {
            let mode = if writable { 0o644 } else { 0o444 };
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
        }
    }
    for dir in dirs {
        let mode = if writable { 0o755 } else { 0o555 };
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(mode)).unwrap();
    }
}

#[test]
fn a_database_in_a_read_only_directory_opens_and_reads() {
    // A stage reads the database an earlier stage committed, and a committed
    // product directory is read-only: opening it must write nothing.
    let scratch = Scratch::new();
    ingest(&scratch.open(), 0, 4);
    set_writable(scratch.dir.path(), false);
    let opened = Database::open(scratch.dir.path(), Config::default())
        .and_then(|db| db.snapshot()?.records().map(|r| r.len()));
    set_writable(scratch.dir.path(), true);
    assert_eq!(
        opened.unwrap(),
        8,
        "every record of the committed database reads"
    );
}
