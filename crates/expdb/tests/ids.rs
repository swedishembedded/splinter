// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: ids are made without a coordinator and never collide, and content
//! ids name exactly the bytes they were made from.
#![allow(clippy::unwrap_used)]

use std::collections::HashSet;

use splinter_expdb::{ContentId, RecordId, WriterId, WriterIdentity};

#[test]
fn the_same_bytes_have_the_same_content_id_and_other_bytes_do_not() {
    assert_eq!(ContentId::of(b"abc"), ContentId::of(b"abc"));
    assert_ne!(ContentId::of(b"abc"), ContentId::of(b"abd"));
}

#[test]
fn a_content_id_round_trips_through_its_text_form() {
    let id = ContentId::of(b"round trip");
    assert_eq!(ContentId::parse(&id.to_string()).unwrap(), id);
    assert!(ContentId::parse("not-hex").is_err());
    assert!(ContentId::parse(&"ab".repeat(31)).is_err());
}

#[test]
fn writers_with_different_identities_get_different_ids() {
    let ids: HashSet<WriterId> = (0..64)
        .map(|rank| WriterIdentity::new("exp", "job-1", "node-a", rank).writer_id(0))
        .collect();
    assert_eq!(ids.len(), 64);
}

#[test]
fn the_same_identity_started_twice_is_told_apart_by_its_incarnation() {
    let identity = WriterIdentity::new("exp", "job-1", "node-a", 0);
    assert_ne!(identity.writer_id(1), identity.writer_id(2));
    assert_eq!(identity.writer_id(1), identity.writer_id(1));
}

#[test]
fn record_ids_order_by_writer_then_sequence() {
    let w = WriterId::from_raw(5);
    assert!(RecordId::new(w, 1) < RecordId::new(w, 2));
    assert!(RecordId::new(WriterId::from_raw(4), 9) < RecordId::new(w, 0));
}

#[test]
fn a_record_id_round_trips_through_its_text_form() {
    let id = RecordId::new(WriterId::from_raw(0xdead_beef), 42);
    assert_eq!(id.to_string().parse::<RecordId>().unwrap(), id);
}
