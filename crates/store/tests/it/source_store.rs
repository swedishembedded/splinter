// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements content-addressed source stores that every
// learned fact can be traced back to, for its clients. If your team needs
// expertise in training-data provenance or durable learning state, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a source is stored once under an address of its origin and part
//! contents, its bytes are stored once per content and shared across
//! sources, every read is verified against its address, and a span that
//! names a part resolves to exactly the bytes it covers.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

use splinter_core::clock::FixedClock;
use splinter_core::digest::Digest;
use splinter_core::experience::Span;
use splinter_core::source::{CapturedSource, Origin, PartContent, PartRef, SourceError};
use splinter_store::experiences::StoreError;
use splinter_store::sources::SourceStore;
use splinter_store::StateRoot;

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        static N: AtomicU32 = AtomicU32::new(0);
        let path = std::env::temp_dir().join(format!(
            "splinter-src-{name}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        Self(path)
    }
    fn store(&self) -> SourceStore {
        SourceStore::new(&splinter_store::workspace::Workspace::at(&StateRoot::new(
            &self.0,
        )))
    }
    /// How many distinct contents the stored sources' parts hold.
    fn blobs(&self) -> usize {
        let store = self.store();
        let mut contents = std::collections::BTreeSet::new();
        for id in store.list().unwrap() {
            for part in store.get_source(&id).unwrap().parts {
                contents.insert(part.content);
            }
        }
        contents.len()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

const README: &str = "# Tool\n\nThe tool prints its version with --version.\n";
const NOTES: &str = "Plain notes.\n\nSecond paragraph.\n";

fn part(name: &str, media_type: &str, text: &str) -> PartContent {
    PartContent {
        name: name.into(),
        media_type: media_type.into(),
        bytes: text.as_bytes().to_vec(),
    }
}

fn repository(at: &str, notes: &str) -> CapturedSource {
    CapturedSource::new(
        Origin::Repository {
            path: "/src/tool".into(),
            revision: None,
            skipped: Vec::new(),
        },
        vec![
            part("notes.txt", "text/plain", notes),
            part("README.md", "text/markdown", README),
        ],
        &FixedClock::new(at),
    )
    .unwrap()
}

#[test]
fn a_source_round_trips_and_its_spans_resolve_to_exact_bytes() {
    let scratch = Scratch::new("roundtrip");
    let store = scratch.store();
    let captured = repository("2026-09-30T08:00:00.000Z", NOTES);
    let id = store.put_source(&captured).unwrap();
    assert_eq!(&id, &captured.source().id);

    let source = store.get_source(&id).unwrap();
    assert_eq!(&source, captured.source());
    assert_eq!(source.kind(), "repository");
    let names: Vec<&str> = source.parts.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(
        names,
        ["README.md", "notes.txt"],
        "parts are ordered by name"
    );
    let readme = source.part("README.md").unwrap();
    assert_eq!(readme.content, Digest::of(README.as_bytes()));
    assert_eq!(readme.bytes, README.len() as u64);
    assert_eq!(
        store.read_part(&id, "README.md").unwrap(),
        README.as_bytes()
    );

    let start = README.find("The tool").unwrap() as u64;
    let end = README.find(" with").unwrap() as u64;
    let span = Span::in_part(
        PartRef {
            source: id.clone(),
            name: "README.md".into(),
        },
        readme.content.clone(),
        start,
        end,
    )
    .unwrap();
    assert_eq!(
        store.read_span(&span).unwrap(),
        b"The tool prints its version"
    );

    // A span past the end of its part, or naming a part whose content is
    // not the span's, does not resolve.
    let past = Span::in_part(span.part.clone().unwrap(), readme.content.clone(), 0, 999).unwrap();
    assert!(matches!(
        store.read_span(&past),
        Err(StoreError::SpanOutOfRange { .. })
    ));
    let mismatched = Span::in_part(
        PartRef {
            source: id.clone(),
            name: "notes.txt".into(),
        },
        readme.content.clone(),
        0,
        4,
    )
    .unwrap();
    assert!(store.read_span(&mismatched).is_err());
    assert_eq!(store.list().unwrap(), vec![id]);
}

#[test]
fn identical_content_is_one_source_and_blobs_are_shared_across_sources() {
    let scratch = Scratch::new("dedup");
    let store = scratch.store();
    let first = store
        .put_source(&repository("2026-09-30T08:00:00.000Z", NOTES))
        .unwrap();
    // The capture time is not part of the address: recapturing unchanged
    // content later is the same source, and the first capture time stays.
    let again = store
        .put_source(&repository("2026-09-30T09:00:00.000Z", NOTES))
        .unwrap();
    assert_eq!(first, again);
    assert_eq!(
        store.get_source(&first).unwrap().captured_at,
        "2026-09-30T08:00:00.000Z"
    );
    assert_eq!(scratch.blobs(), 2);

    // A changed file is a new source that shares the unchanged blob.
    let changed = store
        .put_source(&repository("2026-09-30T09:00:00.000Z", "edited\n"))
        .unwrap();
    assert_ne!(changed, first);
    assert_eq!(scratch.blobs(), 3);
    assert_eq!(store.list().unwrap().len(), 2);
}

#[test]
fn damaged_storage_is_refused_on_read() {
    let scratch = Scratch::new("corrupt");
    let store = scratch.store();
    let id = store
        .put_source(&repository("2026-09-30T08:00:00.000Z", NOTES))
        .unwrap();
    assert!(matches!(
        store.read_part(&id, "missing.txt"),
        Err(StoreError::UnknownPart { .. })
    ));
    assert!(matches!(
        store.read_blob(&Digest::of(b"never stored")),
        Err(StoreError::UnknownBlob(_))
    ));

    for pack in files_under(&scratch.0.join("expdb/blobs")) {
        let mut bytes = fs::read(&pack).unwrap();
        for byte in &mut bytes {
            *byte ^= 0xff;
        }
        fs::write(&pack, bytes).unwrap();
    }
    assert!(
        scratch.store().read_part(&id, "notes.txt").is_err(),
        "damaged content is an error, never different bytes"
    );
}

fn files_under(dir: &std::path::Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for entry in fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() {
            found.extend(files_under(&path));
        } else {
            found.push(path);
        }
    }
    found
}

#[test]
fn a_part_name_appears_once() {
    let twice = CapturedSource::new(
        Origin::Document {
            path: "/doc.md".into(),
        },
        vec![
            part("doc.md", "text/markdown", "a"),
            part("doc.md", "text/markdown", "b"),
        ],
        &FixedClock::new("2026-09-30T08:00:00.000Z"),
    );
    assert!(matches!(twice, Err(SourceError::DuplicatePart(_))));
}
