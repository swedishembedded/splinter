// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements experience databases proven by killing them
// between every write they make, for its clients. If your team needs
// expertise in crash-safe storage for machine-learning data, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec, by enumeration (plan 10B): the database's writes killed at every
//! durability boundary. Every object a backend stores is published by one
//! atomic call (a synced temporary linked or renamed into place), so a crash
//! between two calls is a crash at a boundary of the database's own
//! protocol. Over one scenario (records of one writer committed before; two
//! more writers flush, then segments and blobs are compacted, the jobs'
//! refs folded into the catalog and retired, and unused files collected,
//! with no grace period), a child process is ended after each storage call that
//! changes anything, in turn, in both layouts. After each crash:
//!
//! * the database opens, and a deep verification finds nothing missing or
//!   damaged;
//! * every record committed before reads back;
//! * every record it shows is one an uninterrupted run ends with: nothing
//!   partial or invented appears;
//! * running the scenario again ends with exactly the records an
//!   uninterrupted run ends with.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use splinter_expdb::backend::{Key, Kind, Layout, StorageBackend};
use splinter_expdb::model::{Body, Skill};
use splinter_expdb::{Config, Database, Result, SystemClock, WriterIdentity};

const PROBE: &str = "crash::probe";
const DIR_ARG: &str = "crash-probe-dir=";
const AT_ARG: &str = "crash-probe-at=";
const LAYOUT_ARG: &str = "crash-probe-layout=";
/// The exit status of a child ended at a boundary.
const CRASHED: i32 = 86;

/// A backend that counts the calls that change what is stored, and ends the
/// process right after the one it is armed at, as a crash would.
struct Crashing {
    inner: Arc<dyn StorageBackend>,
    at: Option<usize>,
    done: AtomicUsize,
    seen: Mutex<Vec<String>>,
}

impl Crashing {
    fn new(inner: Arc<dyn StorageBackend>, at: Option<usize>) -> Self {
        Self {
            inner,
            at,
            done: AtomicUsize::new(0),
            seen: Mutex::new(Vec::new()),
        }
    }

    fn step(&self, what: String) {
        let n = self.done.fetch_add(1, Ordering::SeqCst);
        self.seen.lock().unwrap().push(what);
        if self.at == Some(n) {
            std::process::exit(CRASHED);
        }
    }
}

impl StorageBackend for Crashing {
    fn write_once(&self, key: &Key, bytes: &[u8]) -> Result<bool> {
        let published = self.inner.write_once(key, bytes)?;
        self.step(format!("write_once {}", key.relative_path()));
        Ok(published)
    }
    fn replace(&self, key: &Key, bytes: &[u8]) -> Result<()> {
        self.inner.replace(key, bytes)?;
        self.step(format!("replace {}", key.relative_path()));
        Ok(())
    }
    fn rename(&self, from: &Key, to: &Key) -> Result<bool> {
        let moved = self.inner.rename(from, to)?;
        self.step(format!(
            "rename {} -> {}",
            from.relative_path(),
            to.relative_path()
        ));
        Ok(moved)
    }
    fn read(&self, key: &Key) -> Result<Vec<u8>> {
        self.inner.read(key)
    }
    fn read_range(&self, key: &Key, offset: u64, len: usize) -> Result<Vec<u8>> {
        self.inner.read_range(key, offset, len)
    }
    fn len(&self, key: &Key) -> Result<u64> {
        self.inner.len(key)
    }
    fn exists(&self, key: &Key) -> Result<bool> {
        self.inner.exists(key)
    }
    fn list(&self, kind: Kind) -> Result<Vec<Key>> {
        self.inner.list(kind)
    }
    fn remove(&self, key: &Key) -> Result<()> {
        self.inner.remove(key)?;
        self.step(format!("remove {}", key.relative_path()));
        Ok(())
    }
    fn modified(&self, key: &Key) -> Result<SystemTime> {
        self.inner.modified(key)
    }
    fn touch(&self, key: &Key) -> Result<()> {
        self.inner.touch(key)?;
        self.step(format!("touch {}", key.relative_path()));
        Ok(())
    }
    fn preferred_write_bytes(&self) -> usize {
        self.inner.preferred_write_bytes()
    }
    fn layout(&self) -> Layout {
        self.inner.layout()
    }
}

fn layout(name: &str) -> Layout {
    if name == "loose" {
        Layout::Loose
    } else {
        Layout::Packed
    }
}

fn open(root: &Path, layout: Layout) -> Database {
    Database::with_backend(layout.backend(root), config(), Arc::new(SystemClock)).unwrap()
}

/// No grace period: the collection removes and retires everything unused,
/// so its every step is enumerated too.
fn config() -> Config {
    Config {
        orphan_grace: std::time::Duration::ZERO,
        ..Config::default()
    }
}

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

fn write(db: &Database, rank: u32) {
    let mut c = db
        .collector(&WriterIdentity::new("exp", "job", "node", rank))
        .unwrap();
    for n in 0..10 {
        c.record(skill(format!("{rank}-{n}"))).unwrap();
    }
    c.flush().unwrap();
}

/// The records committed before the scenario.
fn before(root: &Path, layout: Layout) {
    write(&open(root, layout), 0);
}

/// The writes a crash interrupts.
fn scenario(db: &Database) {
    write(db, 1);
    write(db, 2);
    db.compact_segments().unwrap();
    db.compact_blobs().unwrap();
    db.absorb_jobs().unwrap();
    db.gc().unwrap();
}

/// The records the database shows, by what they hold (a record's id names
/// the writer that made it, which differs from one database to another), and
/// how many of each writer's rank it shows: a flush is ten records, so a
/// count that is not a multiple of ten is a flush shown in part.
fn records(db: &Database) -> (BTreeSet<String>, BTreeMap<String, usize>) {
    let all = db.snapshot().unwrap().records().unwrap();
    let mut names = BTreeSet::new();
    let mut per_rank = BTreeMap::new();
    for record in &all {
        let Body::Skill(skill) = &record.body else {
            panic!("only skills are written, got {:?}", record.body)
        };
        names.insert(skill.name.clone());
        let rank = skill.name.split('-').next().unwrap().to_string();
        *per_rank.entry(rank).or_default() += 1;
    }
    (names, per_rank)
}

/// Whether every flush `per_rank` counts is whole.
fn whole(per_rank: &BTreeMap<String, usize>) -> bool {
    per_rank.values().all(|n| n % 10 == 0)
}

/// Run as a child: crash after the storage call the arguments name.
#[test]
#[ignore = "a child process of the crash enumeration, run by it"]
fn probe() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |prefix: &str| {
        args.iter()
            .find_map(|a| a.strip_prefix(prefix))
            .unwrap_or_else(|| panic!("the crash enumeration passes {prefix}"))
            .to_owned()
    };
    let root = PathBuf::from(arg(DIR_ARG));
    let layout = layout(&arg(LAYOUT_ARG));
    let backend = Arc::new(Crashing::new(
        layout.backend(&root),
        Some(arg(AT_ARG).parse().unwrap()),
    ));
    let db = Database::with_backend(backend, config(), Arc::new(SystemClock)).unwrap();
    scenario(&db);
}

fn enumerate(layout_name: &str) {
    let layout = layout(layout_name);
    // The calls the scenario makes, and where it ends uninterrupted.
    let reference = tempfile::tempdir().unwrap();
    before(reference.path(), layout);
    let (first, _) = records(&open(reference.path(), layout));
    let counting = Arc::new(Crashing::new(layout.backend(reference.path()), None));
    let db = Database::with_backend(counting.clone(), config(), Arc::new(SystemClock)).unwrap();
    scenario(&db);
    let calls = counting.seen.lock().unwrap().clone();
    let (expected, per_rank) = records(&open(reference.path(), layout));
    assert_eq!(first.len(), 10);
    assert_eq!(expected.len(), 30);
    assert!(per_rank.values().all(|n| *n == 10), "{per_rank:?}");
    assert!(
        calls.len() > 10,
        "{layout_name}: the scenario makes {} calls",
        calls.len()
    );

    for (at, call) in calls.iter().enumerate() {
        let crash = format!("{layout_name}: crashed after call {at} ({call})");
        let dir = tempfile::tempdir().unwrap();
        before(dir.path(), layout);
        let out = Command::new(std::env::current_exe().unwrap())
            .args(["--ignored", "--exact", PROBE, "--test-threads=1", "--quiet"])
            .arg(format!("{DIR_ARG}{}", dir.path().display()))
            .arg(format!("{AT_ARG}{at}"))
            .arg(format!("{LAYOUT_ARG}{layout_name}"))
            .output()
            .unwrap();
        assert_eq!(
            out.status.code(),
            Some(CRASHED),
            "{crash}: the child did not end there: {}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        let db = open(dir.path(), layout);
        let report = db.verify(true).unwrap();
        assert!(report.problems.is_empty(), "{crash}: {:?}", report.problems);
        let (shown, per_rank) = records(&db);
        assert!(
            shown.is_superset(&first),
            "{crash}: a committed record is gone"
        );
        assert!(
            shown.is_subset(&expected),
            "{crash}: records no run ends with"
        );
        assert!(
            per_rank.values().all(|n| *n == 10),
            "{crash}: a flush shown in part or twice: {per_rank:?}"
        );
        let snapshot = db.snapshot().unwrap();
        for record in snapshot.records().unwrap() {
            assert!(
                snapshot.get(record.id).unwrap().is_some(),
                "{crash}: a record does not read"
            );
        }
        scenario(&db);
        // A restarted writer records again what it records: the database is
        // append-only, so a flush the crash had committed is there twice,
        // whole, and nothing else is.
        let (resumed, per_rank) = records(&db);
        assert_eq!(resumed, expected, "{crash}: resumed elsewhere");
        assert!(
            whole(&per_rank),
            "{crash}: a flush shown in part: {per_rank:?}"
        );
        let report = db.verify(true).unwrap();
        assert!(
            report.problems.is_empty(),
            "{crash}: after resuming: {:?}",
            report.problems
        );
    }
}

#[test]
fn a_crash_between_any_two_writes_of_the_packed_layout_loses_nothing_committed() {
    enumerate("packed");
}

#[test]
fn a_crash_between_any_two_writes_of_the_loose_layout_loses_nothing_committed() {
    enumerate("loose");
}
