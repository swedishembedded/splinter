// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Properties that must hold for any input, not only the examples: formats
//! round-trip, damage is always noticed, history merges in any order, and
//! clock conversions and flow targets obey their formulae.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::BTreeSet;

use common::{rid, Scratch};
use proptest::prelude::*;
use splinter_expdb::backend::{Key, Kind};
use splinter_expdb::blob::BlobStore;
use splinter_expdb::format::{encode_segment, Segment};
use splinter_expdb::manifest::{ObjectKind, ObjectRef};
use splinter_expdb::model::{Body, ClockMapping, Record, Skill};
use splinter_expdb::train::flow_matching;
use splinter_expdb::{Compression, Config, ContentId};

fn config() -> ProptestConfig {
    // Each case touches the filesystem, so fewer; no regression files are
    // written into the repository.
    ProptestConfig {
        cases: 16,
        failure_persistence: None,
        ..ProptestConfig::default()
    }
}

fn skill_record(writer: u64, seq: u64, name: String, ts: u64, parent: Option<u64>) -> Record {
    let mut record = Record::new(
        rid(writer, seq),
        ts,
        Body::Skill(Skill {
            name,
            description: "d".into(),
            trigger: String::new(),
            action_pattern: String::new(),
            expected_effect: String::new(),
            parents: vec![],
            prerequisites: vec![],
        }),
    );
    record.parent = parent.map(|p| rid(writer, p));
    record
}

fn records() -> impl Strategy<Value = Vec<Record>> {
    prop::collection::vec(
        ("[a-z ]{0,20}", any::<u64>(), prop::option::of(0u64..50)),
        1..40,
    )
    .prop_map(|items| {
        items
            .into_iter()
            .enumerate()
            .map(|(i, (name, ts, parent))| skill_record(7, i as u64, name, ts, parent))
            .collect()
    })
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn a_segment_returns_exactly_the_records_it_was_given(
        records in records(),
        block in 1usize..12,
        zstd in any::<bool>(),
    ) {
        let scratch = Scratch::new();
        let compression = if zstd { Compression::Zstd(3) } else { Compression::None };
        let db = scratch.open_with(Config { block_records: block, compression, ..Config::default() });
        let (id, bytes) = encode_segment(&records, &[], db.config()).unwrap();
        db.backend().write_once(&Key::new(Kind::Segment, &format!("{id}.seg")).unwrap(), &bytes).unwrap();
        let segment = Segment::open(db.backend_arc(), id).unwrap();
        prop_assert_eq!(segment.records().unwrap(), records);
        prop_assert!(segment.verify().is_ok());
    }

    #[test]
    fn no_single_damaged_byte_in_a_segment_goes_unnoticed(
        records in records(),
        at in any::<prop::sample::Index>(),
        flip in 1u8..=255,
    ) {
        let scratch = Scratch::new();
        let db = scratch.open();
        let (_, mut bytes) = encode_segment(&records, &[], db.config()).unwrap();
        let position = at.index(bytes.len());
        bytes[position] ^= flip;
        let damaged = ContentId::of(&bytes);
        db.backend().write_once(&Key::new(Kind::Segment, &format!("{damaged}.seg")).unwrap(), &bytes).unwrap();
        // Either the segment will not open, or reading it fails; it is never
        // accepted as sound.
        let sound = Segment::open(db.backend_arc(), damaged).and_then(|s| s.verify()).is_ok();
        prop_assert!(!sound, "a flipped byte at {position} was accepted");
    }

    #[test]
    fn any_bytes_stored_in_the_blob_store_come_back_and_are_stored_once(
        data in prop::collection::vec(any::<u8>(), 0..40_000),
    ) {
        let scratch = Scratch::new();
        let db = scratch.open_with(Config { chunk_min: 64, chunk_avg: 256, chunk_max: 1024, ..Config::default() });
        let mut store = BlobStore::open(&db).unwrap();
        let blob = store.put(&data).unwrap();
        prop_assert_eq!(store.get(&blob).unwrap(), data.clone());
        let written = store.stats().chunks_written;
        store.put(&data).unwrap();
        prop_assert_eq!(store.stats().chunks_written, written);
    }

    #[test]
    fn what_is_visible_does_not_depend_on_the_order_things_were_published(
        ops in prop::collection::vec((0usize..3, 0u8..8, any::<bool>()), 1..12),
        seed in any::<u64>(),
    ) {
        let object = |n: u8| ObjectRef { kind: ObjectKind::Segment, id: ContentId::of(&[n]), bytes: 1, records: 1 };
        let live = |ops: &[(usize, u8, bool)]| {
            let scratch = Scratch::new();
            let db = scratch.open();
            for (job, n, remove) in ops {
                let (add, rm) = if *remove { (vec![], vec![object(*n)]) } else { (vec![object(*n)], vec![]) };
                db.publish(&format!("job-{job}"), add, rm).unwrap();
            }
            db.snapshot().unwrap().segments().into_iter().map(|o| o.id).collect::<BTreeSet<_>>()
        };
        let mut shuffled = ops.clone();
        let mut state = seed | 1;
        for i in (1..shuffled.len()).rev() {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            shuffled.swap(i, (state % (i as u64 + 1)) as usize);
        }
        // Adds and removals only accumulate, and a removal is permanent, so
        // the visible set is the added files minus the removed ones whatever
        // the order.
        let expected: BTreeSet<ContentId> = ops
            .iter()
            .filter(|(_, _, remove)| !remove)
            .map(|(_, n, _)| ContentId::of(&[*n]))
            .filter(|id| !ops.iter().any(|(_, m, remove)| *remove && ContentId::of(&[*m]) == *id))
            .collect();
        prop_assert_eq!(live(&ops), expected.clone());
        prop_assert_eq!(live(&shuffled), expected);
    }

    #[test]
    fn merging_the_catalog_again_changes_nothing(jobs in 1usize..5) {
        let scratch = Scratch::new();
        let db = scratch.open();
        for job in 0..jobs {
            db.publish(&format!("job-{job}"), vec![ObjectRef { kind: ObjectKind::Segment, id: ContentId::of(&[job as u8]), bytes: 1, records: 1 }], vec![]).unwrap();
        }
        let first = db.merge_catalog().unwrap();
        prop_assert_eq!(db.merge_catalog().unwrap(), first);
    }

    #[test]
    fn a_clock_mapping_round_trips_to_within_a_nanosecond(
        slope in 0.999f64..1.001,
        src_ref in -1_000_000_000_000i64..1_000_000_000_000,
        dst_ref in -1_000_000_000_000i64..1_000_000_000_000,
        delta in -3_600_000_000_000i64..3_600_000_000_000,
    ) {
        let mapping = ClockMapping {
            source: ContentId::of(b"a"),
            destination: ContentId::of(b"b"),
            src_ref_ns: src_ref,
            dst_ref_ns: dst_ref,
            slope,
            uncertainty_ns: 1,
        };
        let t = src_ref + delta;
        prop_assert!((mapping.to_source(mapping.to_destination(t)) - t).abs() <= 1);
    }

    #[test]
    fn flow_matching_always_satisfies_its_defining_equations(
        clean in prop::collection::vec(-10.0f32..10.0, 1..64),
        seed in any::<u64>(),
        step in any::<u64>(),
    ) {
        let flow = flow_matching(&clean, seed, step);
        prop_assert!(flow.t > 0.0 && flow.t < 1.0);
        for (i, action) in clean.iter().enumerate() {
            prop_assert_eq!(flow.noisy[i], (1.0 - flow.t) * flow.noise[i] + flow.t * action);
            prop_assert_eq!(flow.target[i], action - flow.noise[i]);
        }
    }
}
