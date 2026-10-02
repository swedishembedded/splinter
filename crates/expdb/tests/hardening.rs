// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: failures and odd inputs never lose or silently change data. Each
//! case here was found by an independent review or by model checking the
//! storage protocol, and is pinned so it cannot return.
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use common::{attempt, rid, Scratch};
use splinter_expdb::backend::{Key, Kind, PosixBackend, StorageBackend};
use splinter_expdb::blob::Tree;
use splinter_expdb::model::{
    Action, Body, ClockDomain, ClockMapping, Decision, Evaluation, EvaluatorRef, Outcome,
    PolicyRef, Target,
};
use splinter_expdb::train::{DataRef, Recipe, Rng};
use splinter_expdb::{Clock, Config, ContentId, Database, StepClock, WriterIdentity};

/// A backend whose next ref replacement fails, as a full disk would.
struct Flaky {
    inner: PosixBackend,
    fail_replace: AtomicBool,
}

impl StorageBackend for Flaky {
    fn write_once(&self, key: &Key, bytes: &[u8]) -> splinter_expdb::Result<bool> {
        self.inner.write_once(key, bytes)
    }
    fn replace(&self, key: &Key, bytes: &[u8]) -> splinter_expdb::Result<()> {
        if self.fail_replace.swap(false, Ordering::SeqCst)
            && key.kind() == Kind::Ref
            && key.name() != "format"
        {
            return Err(splinter_expdb::Error::Io {
                path: key.name().into(),
                source: std::io::Error::other("no space left on device"),
            });
        }
        self.inner.replace(key, bytes)
    }
    fn read(&self, key: &Key) -> splinter_expdb::Result<Vec<u8>> {
        self.inner.read(key)
    }
    fn read_range(&self, key: &Key, offset: u64, len: usize) -> splinter_expdb::Result<Vec<u8>> {
        self.inner.read_range(key, offset, len)
    }
    fn len(&self, key: &Key) -> splinter_expdb::Result<u64> {
        self.inner.len(key)
    }
    fn exists(&self, key: &Key) -> splinter_expdb::Result<bool> {
        self.inner.exists(key)
    }
    fn list(&self, kind: Kind) -> splinter_expdb::Result<Vec<Key>> {
        self.inner.list(kind)
    }
    fn remove(&self, key: &Key) -> splinter_expdb::Result<()> {
        self.inner.remove(key)
    }
    fn rename(&self, from: &Key, to: &Key) -> splinter_expdb::Result<bool> {
        self.inner.rename(from, to)
    }
    fn modified(&self, key: &Key) -> splinter_expdb::Result<SystemTime> {
        self.inner.modified(key)
    }
    fn preferred_write_bytes(&self) -> usize {
        self.inner.preferred_write_bytes()
    }
}

fn skill(name: &str) -> Body {
    Body::Skill(common::skill(name))
}

#[test]
fn a_flush_that_fails_to_publish_keeps_the_records_so_the_retry_loses_nothing() {
    let scratch = Scratch::new();
    let flaky = Arc::new(Flaky {
        inner: PosixBackend::new(scratch.dir.path()),
        fail_replace: AtomicBool::new(false),
    });
    let db = Database::with_backend(
        flaky.clone(),
        Config::default(),
        Arc::new(splinter_expdb::SystemClock),
    )
    .unwrap();
    let mut c = db
        .collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap();
    for n in 0..5 {
        c.record(skill(&format!("s{n}"))).unwrap();
    }
    flaky.fail_replace.store(true, Ordering::SeqCst);
    assert!(c.flush().is_err(), "the full disk is reported");
    assert_eq!(c.buffered(), 5, "and nothing was thrown away");

    c.flush().unwrap();
    assert_eq!(db.snapshot().unwrap().records().unwrap().len(), 5);
}

#[test]
fn rewriting_an_existing_file_restarts_its_grace_period() {
    let scratch = Scratch::new();
    let backend = PosixBackend::new(scratch.dir.path());
    let key = Key::new(Kind::Segment, "aaaa.seg").unwrap();
    backend.write_once(&key, b"x").unwrap();
    let path = scratch.dir.path().join("segments/aa/aaaa.seg");
    let old = SystemTime::now() - Duration::from_secs(10 * 24 * 3600);
    std::fs::File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_modified(old)
        .unwrap();

    assert!(
        !backend.write_once(&key, b"x").unwrap(),
        "it was already there"
    );
    let age = SystemTime::now()
        .duration_since(backend.modified(&key).unwrap())
        .unwrap();
    assert!(
        age < Duration::from_secs(3600),
        "a writer about to publish it must not find it collectable"
    );
}

#[test]
fn pinning_a_snapshot_of_an_empty_database_works_and_does_not_break_collection() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let snapshot = db.snapshot().unwrap();
    snapshot.pin("run").unwrap();
    db.gc().unwrap();
    assert!(db
        .snapshot_at(snapshot.id())
        .unwrap()
        .records()
        .unwrap()
        .is_empty());
}

#[test]
fn the_snapshot_a_plan_was_compiled_from_can_always_be_reopened() {
    let scratch = Scratch::new();
    let db = scratch.open();
    for rank in 0..3 {
        let mut c = db
            .collector(&WriterIdentity::new("exp", "job", "node", rank))
            .unwrap();
        attempt(&mut c, u64::from(rank), "p", 2, Outcome::Pass);
        c.flush().unwrap();
    }
    let snapshot = db.snapshot().unwrap();
    let plan = snapshot.compile(&Recipe::sft()).unwrap();
    let reopened = db.snapshot_at(plan.snapshot).unwrap();
    assert_eq!(
        reopened.compile(&Recipe::sft()).unwrap().id().unwrap(),
        plan.id().unwrap()
    );
}

#[test]
fn a_value_json_cannot_represent_is_refused_instead_of_silently_becoming_nothing() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db
        .collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap();
    let bad = |score: f64| {
        Body::Evaluation(Evaluation::new(
            Target::Record(rid(1, 1)),
            EvaluatorRef::new("e", "1"),
            "c",
            score,
            1.0,
        ))
    };
    assert!(c.record(bad(f64::NAN)).is_err());
    assert!(c.record(bad(f64::INFINITY)).is_err());
    assert!(c.record(bad(0.5)).is_ok());
    let decision = |logprob: f64| {
        Body::Decision(Decision {
            state: ContentId::of(b"s"),
            observation: None,
            policy: PolicyRef::new("p", "1"),
            context: None,
            action: Action::new("a", serde_json::Value::Null),
            old_logprob: Some(logprob),
            value_estimate: None,
        })
    };
    assert!(
        c.record(decision(f64::NEG_INFINITY)).is_err(),
        "a zero-probability choice is not representable"
    );
    assert!(c.record(decision(-0.7)).is_ok());
    let clock = ContentId::of(b"c");
    let mapping = |slope: f64| {
        Body::ClockMapping(ClockMapping {
            source: clock,
            destination: ContentId::of(b"d"),
            src_ref_ns: 0,
            dst_ref_ns: 0,
            slope,
            uncertainty_ns: 1,
        })
    };
    assert!(c.record(mapping(f64::NAN)).is_err());
    assert!(
        c.record(mapping(0.0)).is_err(),
        "a clock that does not advance cannot be mapped"
    );
    assert!(c.record(mapping(-1.0)).is_err());
    // Nothing bad reached the segment, so it still reads: the two good records.
    c.flush().unwrap();
    assert_eq!(db.snapshot().unwrap().records().unwrap().len(), 2);
}

#[test]
fn epoch_nanosecond_timestamps_convert_without_losing_a_single_nanosecond() {
    let epoch = 1_700_000_000_000_000_000i64;
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db
        .collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap();
    let wall = c
        .register_clock(&ClockDomain {
            name: "wall".into(),
            description: String::new(),
        })
        .unwrap();
    c.flush().unwrap();
    let snapshot = db.snapshot().unwrap();
    // A clock read against itself is the identity, exactly.
    let identity = snapshot.resolve_clock(wall, wall).unwrap().unwrap();
    for t in [epoch, epoch + 1, epoch + 125_000, i64::MAX, i64::MIN] {
        assert_eq!(identity.apply(t), t);
    }
    // A mapping that is only an offset is exact too.
    let mapping = ClockMapping {
        source: wall,
        destination: ContentId::of(b"other"),
        src_ref_ns: epoch,
        dst_ref_ns: 0,
        slope: 1.0,
        uncertainty_ns: 1,
    };
    assert_eq!(mapping.to_destination(epoch + 125_000), 125_000);
    assert_eq!(mapping.to_source(125_000), epoch + 125_000);
    // Extreme times saturate instead of wrapping.
    assert_eq!(mapping.to_destination(i64::MAX), i64::MAX - epoch);
}

#[test]
fn absorbing_a_ref_whose_newest_manifest_the_catalog_has_not_seen_keeps_that_manifest() {
    let scratch = Scratch::new();
    let db = scratch.open_with(Config {
        orphan_grace: Duration::ZERO,
        ..Config::default()
    });
    let mut c = db
        .collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap();
    c.record(skill("early")).unwrap();
    c.flush().unwrap();
    db.merge_catalog().unwrap();
    // The writer publishes again after the catalog merged.
    c.record(skill("late")).unwrap();
    c.flush().unwrap();

    let report = db.absorb_jobs().unwrap();
    assert_eq!(report.pruned, 1);
    assert!(db.job_heads().unwrap().is_empty());
    let names: Vec<_> = db
        .snapshot()
        .unwrap()
        .records()
        .unwrap()
        .into_iter()
        .map(|r| match r.body {
            Body::Skill(s) => s.name,
            _ => String::new(),
        })
        .collect();
    assert_eq!(
        names.len(),
        2,
        "the late record is in the catalog, not lost with its ref: {names:?}"
    );
}

#[test]
fn merging_the_catalog_after_a_checkpoint_does_not_pull_old_history_back_in() {
    let scratch = Scratch::new();
    let db = scratch.open();
    for rank in 0..3 {
        let mut c = db
            .collector(&WriterIdentity::new("exp", "job", "node", rank))
            .unwrap();
        c.record(skill(&format!("s{rank}"))).unwrap();
        c.flush().unwrap();
    }
    let checkpoint = db.checkpoint().unwrap();
    assert_eq!(
        db.merge_catalog().unwrap(),
        checkpoint,
        "the checkpoint already holds every head"
    );
}

#[test]
fn a_zero_priority_means_never_and_a_non_finite_one_is_refused() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut overlay = splinter_expdb::overlay::ReplayOverlay::load(&db, "t").unwrap();
    let ids: Vec<_> = (0..4).map(|n| rid(1, n)).collect();
    overlay.set_priority(ids[0], 0.0).unwrap();
    for id in &ids[1..] {
        overlay.set_priority(*id, 1.0).unwrap();
    }
    for seed in 0..40 {
        assert!(
            !overlay.sample(&ids, 3, seed).contains(&ids[0]),
            "a record with priority zero is never drawn"
        );
    }
    assert!(overlay.set_priority(ids[1], f64::NAN).is_err());
    assert!(overlay.set_priority(ids[1], -1.0).is_err());
}

#[test]
fn an_episode_without_an_end_still_renders_what_happened() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db
        .collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap();
    let (definition, instance, initial) = common::coding_task(1);
    let mut run = c
        .start_attempt(
            &definition,
            &instance,
            &initial,
            &PolicyRef::new("p", "1"),
            None,
        )
        .unwrap();
    let attempt_id = run.attempt();
    run.decision()
        .commit(Action::new("grep", serde_json::json!({ "q": "x" })))
        .unwrap();
    // An evaluation is the newest record of the attempt, but it is not part of the path.
    run.evaluate(Evaluation::new(
        Target::Record(attempt_id),
        EvaluatorRef::new("v", "1"),
        "task_completion",
        1.0,
        1.0,
    ))
    .unwrap();
    // The attempt never finished: the run is dropped.
    c.flush().unwrap();

    let snapshot = db.snapshot().unwrap();
    let text = snapshot
        .render(&DataRef::Episode {
            attempt: attempt_id,
        })
        .unwrap();
    assert!(
        text.contains("grep"),
        "the episode shows the action: {text:?}"
    );
}

#[test]
fn a_path_that_is_both_a_file_and_a_directory_is_an_error() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut blobs = splinter_expdb::blob::BlobStore::open(&db).unwrap();
    let files = std::collections::BTreeMap::from([
        ("a".to_owned(), b"file".to_vec()),
        ("a/b".to_owned(), b"nested".to_vec()),
    ]);
    assert!(Tree::write(&mut blobs, &files).is_err());
}

#[test]
fn consecutive_flow_matching_steps_draw_from_independent_streams_even_with_seed_zero() {
    let first: Vec<u64> = {
        let mut r = Rng::for_step(0, 0);
        (0..8).map(|_| r.next_u64()).collect()
    };
    let second: Vec<u64> = {
        let mut r = Rng::for_step(0, 1);
        (0..8).map(|_| r.next_u64()).collect()
    };
    assert!(
        first.iter().all(|v| !second.contains(v)),
        "step 1 must not replay step 0 shifted by one"
    );
}

#[test]
fn a_clock_that_ticks_is_never_mistaken_for_one_that_stands_still() {
    // A deterministic clock cannot make two writers of one identity collide.
    let scratch = Scratch::new();
    let backend = Arc::new(PosixBackend::new(scratch.dir.path()));
    let clock: Arc<dyn Clock> = Arc::new(StepClock::new(0, 0));
    let db = Database::with_backend(backend, Config::default(), clock).unwrap();
    let identity = WriterIdentity::new("exp", "job", "node", 0);
    let a = db.collector(&identity).unwrap().writer_id();
    let b = db.collector(&identity).unwrap().writer_id();
    assert_ne!(
        a, b,
        "two incarnations of one identity on a frozen clock still get distinct writer ids"
    );
}

#[test]
fn a_blob_reference_with_an_absurd_length_is_an_error_not_an_abort() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut blobs = splinter_expdb::blob::BlobStore::open(&db).unwrap();
    let real = blobs.put(&vec![7u8; 100_000]).unwrap();
    let lying = splinter_expdb::blob::BlobRef {
        id: real.id,
        len: u64::MAX,
    };
    assert!(blobs.get(&lying).is_err());
}

#[test]
fn processes_absorbing_at_the_same_moment_cannot_lose_each_others_history() {
    // Six finished writers, and three processes folding their refs into the
    // catalog together. The catalog is never overwritten, so whatever each
    // folded is still in it when the dust settles, and a collection that
    // follows finds every record still needed.
    let scratch = Scratch::new();
    let config = Config {
        orphan_grace: Duration::ZERO,
        ..Config::default()
    };
    let db = scratch.open_with(config.clone());
    for rank in 0..6 {
        let mut c = db
            .collector(&WriterIdentity::new("exp", "job", "node", rank))
            .unwrap();
        for n in 0..10 {
            c.record(skill(&format!("{rank}-{n}"))).unwrap();
        }
        c.flush().unwrap();
    }
    let root = scratch.dir.path().to_path_buf();
    let workers: Vec<_> = (0..3)
        .map(|_| {
            let (root, config) = (root.clone(), config.clone());
            std::thread::spawn(move || {
                Database::open(&root, config)
                    .unwrap()
                    .absorb_jobs()
                    .unwrap()
            })
        })
        .collect();
    for w in workers {
        w.join().unwrap();
    }
    db.merge_catalog().unwrap();
    db.gc().unwrap();
    assert_eq!(db.snapshot().unwrap().records().unwrap().len(), 60);
    assert_eq!(
        db.catalog_heads().unwrap().len(),
        1,
        "the heads were joined into one"
    );
}

#[test]
fn a_collection_that_died_midway_is_finished_by_the_next_one() {
    let scratch = Scratch::new();
    let db = scratch.open_with(Config {
        orphan_grace: Duration::ZERO,
        ..Config::default()
    });
    let mut c = db
        .collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap();
    for n in 0..4 {
        c.record(skill(&format!("s{n}"))).unwrap();
    }
    c.flush().unwrap();
    let needed = db.snapshot().unwrap().segments()[0].clone();
    let orphan = {
        let (records, _) = common::mixed(9, 3);
        let id =
            splinter_expdb::format::seal_segment(db.backend(), &records, &[], db.config()).unwrap();
        splinter_expdb::format::Segment::open(db.backend_arc(), id)
            .unwrap()
            .object_ref()
            .unwrap()
    };
    // The earlier collection moved both aside and then died.
    for object in [&needed, &orphan] {
        let key = object.key().unwrap();
        let aside = Key::new(Kind::Trash, &format!("segments.{}", key.name())).unwrap();
        assert!(db.backend().rename(&key, &aside).unwrap());
    }
    assert_eq!(db.backend().list(Kind::Trash).unwrap().len(), 2);

    db.gc().unwrap();
    assert!(
        db.backend().list(Kind::Trash).unwrap().is_empty(),
        "nothing is left aside"
    );
    assert_eq!(
        db.snapshot().unwrap().records().unwrap().len(),
        4,
        "what was needed is back"
    );
    assert!(
        !db.backend().exists(&orphan.key().unwrap()).unwrap(),
        "what was not is gone"
    );
}

#[test]
fn a_segment_whose_leading_magic_is_damaged_is_refused() {
    use splinter_expdb::format::{encode_segment, Segment};
    let scratch = Scratch::new();
    let db = scratch.open();
    let (records, edges) = common::mixed(1, 3);
    let (_, mut bytes) = encode_segment(&records, &edges, db.config()).unwrap();
    bytes[0] ^= 1;
    let damaged = ContentId::of(&bytes);
    db.backend()
        .write_once(
            &Key::new(Kind::Segment, &format!("{damaged}.seg")).unwrap(),
            &bytes,
        )
        .unwrap();
    assert!(
        Segment::open(db.backend_arc(), damaged)
            .and_then(|s| s.verify())
            .is_err(),
        "only the tail's magic was checked, so a damaged head passed"
    );
}

#[test]
fn a_blob_pack_whose_leading_magic_is_damaged_is_refused() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut collector = db
        .collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap();
    let blob = collector.put_blob(b"some bytes worth keeping").unwrap();
    collector.flush().unwrap();
    let packs = db.backend().list(Kind::BlobPack).unwrap();
    assert_eq!(packs.len(), 1);
    let key = packs[0].clone();
    let mut bytes = db.backend().read(&key).unwrap();
    bytes[0] ^= 1;
    db.backend().replace(&key, &bytes).unwrap();

    let read = db.snapshot().and_then(|s| s.read_blob(&blob.id));
    assert!(read.is_err(), "a pack with a damaged head was read");
}
