// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a sealed segment holds records and edges in column blocks with zone
//! maps, is identified by its content, and refuses to be read if any byte of
//! it is damaged or missing.
#![allow(clippy::unwrap_used)]

mod common;

use common::{decision, family, mixed, rid, Scratch};
use splinter_expdb::backend::{Key, Kind};
use splinter_expdb::format::{encode_segment, seal_segment, Segment};
use splinter_expdb::model::RecordKind;
use splinter_expdb::{Compression, Config};

#[test]
fn records_and_edges_round_trip_through_a_sealed_segment() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let (records, edges) = mixed(7, 40);
    let id = seal_segment(db.backend(), &records, &edges, db.config()).unwrap();

    let segment = Segment::open(db.backend_arc(), id).unwrap();
    assert_eq!(segment.records().unwrap(), records);
    assert_eq!(segment.edges().unwrap(), edges);
    assert_eq!(segment.info().records, 40);
    assert_eq!(segment.info().edges, 39);
    segment.verify().unwrap();
}

#[test]
fn a_segment_is_split_into_blocks_of_the_configured_size() {
    let scratch = Scratch::new();
    let config = Config {
        block_records: 10,
        ..Config::default()
    };
    let db = scratch.open_with(config);
    let (records, edges) = mixed(1, 35);
    let id = seal_segment(db.backend(), &records, &edges, db.config()).unwrap();
    let segment = Segment::open(db.backend_arc(), id).unwrap();
    assert_eq!(segment.info().blocks.len(), 4);
    assert_eq!(segment.read_block(3).unwrap().len(), 5);
}

#[test]
fn compressed_and_uncompressed_segments_hold_the_same_records() {
    let scratch = Scratch::new();
    let (records, edges) = mixed(2, 64);
    let plain = Config {
        compression: Compression::None,
        ..Config::default()
    };
    let zstd = Config {
        compression: Compression::Zstd(3),
        ..Config::default()
    };
    let (plain_id, plain_bytes) = encode_segment(&records, &edges, &plain).unwrap();
    let (zstd_id, zstd_bytes) = encode_segment(&records, &edges, &zstd).unwrap();
    assert_ne!(plain_id, zstd_id);
    assert!(zstd_bytes.len() < plain_bytes.len());

    let db = scratch.open();
    for (id, bytes) in [(plain_id, plain_bytes), (zstd_id, zstd_bytes)] {
        db.backend()
            .write_once(
                &Key::new(Kind::Segment, &format!("{id}.seg")).unwrap(),
                &bytes,
            )
            .unwrap();
        assert_eq!(
            Segment::open(db.backend_arc(), id)
                .unwrap()
                .records()
                .unwrap(),
            records
        );
    }
}

#[test]
fn sealing_the_same_records_twice_is_one_file() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let (records, edges) = mixed(3, 12);
    let a = seal_segment(db.backend(), &records, &edges, db.config()).unwrap();
    let b = seal_segment(db.backend(), &records, &edges, db.config()).unwrap();
    assert_eq!(a, b);
    assert_eq!(db.backend().list(Kind::Segment).unwrap().len(), 1);
}

#[test]
fn an_empty_segment_is_refused() {
    let scratch = Scratch::new();
    let db = scratch.open();
    assert!(seal_segment(db.backend(), &[], &[], db.config()).is_err());
}

fn damaged(
    mutate: impl FnOnce(&mut Vec<u8>),
) -> (Scratch, splinter_expdb::Database, splinter_expdb::ContentId) {
    let scratch = Scratch::new();
    let db = scratch.open();
    let (records, edges) = mixed(4, 30);
    let (_, mut bytes) = encode_segment(&records, &edges, db.config()).unwrap();
    mutate(&mut bytes);
    let id = splinter_expdb::ContentId::of(&bytes);
    db.backend()
        .write_once(
            &Key::new(Kind::Segment, &format!("{id}.seg")).unwrap(),
            &bytes,
        )
        .unwrap();
    (scratch, db, id)
}

#[test]
fn a_truncated_segment_is_rejected_when_opened() {
    let (_s, db, id) = damaged(|bytes| bytes.truncate(bytes.len() - 10));
    assert!(Segment::open(db.backend_arc(), id).is_err());
}

#[test]
fn a_segment_whose_footer_is_damaged_is_rejected_when_opened() {
    let (_s, db, id) = damaged(|bytes| {
        let at = bytes.len() - 40;
        bytes[at] ^= 0x55;
    });
    assert!(Segment::open(db.backend_arc(), id).is_err());
}

#[test]
fn a_flipped_byte_in_a_block_is_an_error_when_the_block_is_read() {
    let (_s, db, id) = damaged(|bytes| bytes[40] ^= 0x01);
    let segment = Segment::open(db.backend_arc(), id).unwrap();
    assert!(segment.read_block(0).is_err());
    assert!(segment.verify().is_err());
}

#[test]
fn zone_maps_let_a_reader_skip_blocks_that_cannot_match() {
    let scratch = Scratch::new();
    let config = Config {
        block_records: 10,
        ..Config::default()
    };
    let db = scratch.open_with(config);
    // Four blocks of ten decisions; every block belongs to its own family.
    let records: Vec<_> = (0..40u64)
        .map(|seq| decision(9, seq, family(seq / 10), "act"))
        .collect();
    let id = seal_segment(db.backend(), &records, &[], db.config()).unwrap();
    let segment = Segment::open(db.backend_arc(), id).unwrap();

    for block in 0..4u64 {
        let wanted = family(block);
        let hits: Vec<usize> = (0..4)
            .filter(|i| segment.info().blocks[*i].zone.may_contain_family(&wanted))
            .collect();
        assert_eq!(hits, [block as usize]);
    }
    assert!(!segment.info().blocks[0]
        .zone
        .may_contain_family(&family(99)));
}

#[test]
fn a_blocks_zone_knows_which_kinds_and_times_it_holds() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let (records, edges) = mixed(5, 8);
    let id = seal_segment(db.backend(), &records, &edges, db.config()).unwrap();
    let zone = Segment::open(db.backend_arc(), id).unwrap().info().blocks[0]
        .zone
        .clone();
    assert!(zone.has_kind(RecordKind::Decision));
    assert!(zone.has_kind(RecordKind::AttemptEnd));
    assert!(!zone.has_kind(RecordKind::Skill));
    assert_eq!((zone.ts_min, zone.ts_max), (0, 7));
    assert!(zone.may_contain_id(rid(5, 3)));
}
