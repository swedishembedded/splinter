// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: the storage backend writes immutable objects once, replaces refs
//! atomically, and never exposes a half-written file.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;
use std::thread;

use splinter_expdb::backend::{Key, Kind, PosixBackend, StorageBackend};
use tempfile::TempDir;

fn backend() -> (TempDir, PosixBackend) {
    let dir = TempDir::new().unwrap();
    let backend = PosixBackend::new(dir.path());
    (dir, backend)
}

#[test]
fn an_object_is_written_once_and_the_second_writer_is_told_so() {
    let (_dir, b) = backend();
    let key = Key::new(Kind::Segment, "abcdef.seg").unwrap();
    assert!(b.write_once(&key, b"first").unwrap());
    assert!(!b.write_once(&key, b"second").unwrap());
    assert_eq!(b.read(&key).unwrap(), b"first");
}

#[test]
fn exactly_one_of_many_racing_writers_wins() {
    let (_dir, b) = backend();
    let b = Arc::new(b);
    let key = Key::new(Kind::BlobPack, "racepack.pack").unwrap();
    let wins: usize = (0..16)
        .map(|n| {
            let b = Arc::clone(&b);
            let key = key.clone();
            thread::spawn(move || {
                b.write_once(&key, format!("writer {n}").as_bytes())
                    .unwrap()
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|h| usize::from(h.join().unwrap()))
        .sum();
    assert_eq!(wins, 1);
}

#[test]
fn a_ref_is_replaced_atomically_and_reads_see_one_whole_value() {
    let (_dir, b) = backend();
    let key = Key::new(Kind::Ref, "jobs/7").unwrap();
    b.replace(&key, b"one").unwrap();
    b.replace(&key, b"two").unwrap();
    assert_eq!(b.read(&key).unwrap(), b"two");
}

#[test]
fn a_byte_range_is_read_without_the_whole_file() {
    let (_dir, b) = backend();
    let key = Key::new(Kind::Segment, "ranged.seg").unwrap();
    b.write_once(&key, b"0123456789").unwrap();
    assert_eq!(b.read_range(&key, 3, 4).unwrap(), b"3456");
    assert_eq!(b.len(&key).unwrap(), 10);
}

#[test]
fn a_range_past_the_end_is_an_error_not_a_short_read() {
    let (_dir, b) = backend();
    let key = Key::new(Kind::Segment, "short.seg").unwrap();
    b.write_once(&key, b"abc").unwrap();
    assert!(b.read_range(&key, 2, 5).is_err());
}

#[test]
fn listing_shows_published_objects_and_never_temporary_files() {
    let (dir, b) = backend();
    for name in ["aa1.seg", "bb2.seg", "cc3.seg"] {
        b.write_once(&Key::new(Kind::Segment, name).unwrap(), b"x")
            .unwrap();
    }
    std::fs::write(dir.path().join("segments/aa/.aa9.seg.1.0.tmp"), b"half").unwrap();
    let mut names: Vec<String> = b
        .list(Kind::Segment)
        .unwrap()
        .into_iter()
        .map(|k| k.name().to_owned())
        .collect();
    names.sort();
    assert_eq!(names, ["aa1.seg", "bb2.seg", "cc3.seg"]);
}

#[test]
fn a_key_cannot_climb_out_of_the_root() {
    assert!(Key::new(Kind::Ref, "../escape").is_err());
    assert!(Key::new(Kind::Ref, "/absolute").is_err());
    assert!(Key::new(Kind::Segment, "a/b.seg").is_err());
    assert!(Key::new(Kind::Segment, "").is_err());
}

#[test]
fn a_missing_object_is_not_found_not_empty() {
    let (_dir, b) = backend();
    let key = Key::new(Kind::Manifest, "nothere").unwrap();
    assert!(!b.exists(&key).unwrap());
    assert!(b.read(&key).is_err());
}

#[test]
fn removing_an_object_makes_it_unreadable() {
    let (_dir, b) = backend();
    let key = Key::new(Kind::Segment, "gone.seg").unwrap();
    b.write_once(&key, b"x").unwrap();
    b.remove(&key).unwrap();
    assert!(!b.exists(&key).unwrap());
}
