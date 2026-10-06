// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable, content-addressed experience
// stores for learning agents, for its clients. If your team needs expertise
// in training-data lineage or crash-safe storage, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: a document is stored once under the digest of its canonical form,
//! read back checked against it, and listed by class.
#![allow(clippy::unwrap_used)]

use serde::{Deserialize, Serialize};
use splinter_core::digest::Digest;
use splinter_store::workspace::Workspace;
use splinter_store::StateRoot;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Note {
    text: String,
    n: u32,
}

fn workspace(name: &str) -> (Workspace, StateRoot) {
    let root = StateRoot::new(
        std::env::temp_dir().join(format!("splinter-doc-{name}-{}", std::process::id())),
    );
    let _ = std::fs::remove_dir_all(root.path());
    (Workspace::at(&root), root)
}

#[test]
fn a_document_is_stored_once_and_read_back_by_its_address() {
    let (ws, root) = workspace("once");
    let note = Note {
        text: "hello".into(),
        n: 7,
    };
    let id = ws.put_document("note", &note).unwrap();
    assert_eq!(ws.put_document("note", &note).unwrap(), id);
    assert!(id.as_str().starts_with("blake3:"));
    assert_eq!(
        ws.get_document::<Note>("note", &id).unwrap(),
        Some(note.clone())
    );
    assert_eq!(
        ws.get_document::<Note>("note", &Digest::of(b"nope"))
            .unwrap(),
        None
    );
    assert_eq!(
        ws.get_document::<Note>("other", &id).unwrap(),
        None,
        "classes are separate"
    );
    assert_eq!(ws.document_ids("note").unwrap(), vec![id.clone()]);
    assert!(ws.has_document("note", &id).unwrap());

    // Another process reads the same document.
    assert_eq!(
        Workspace::at(&root)
            .get_document::<Note>("note", &id)
            .unwrap(),
        Some(note)
    );
    let _ = std::fs::remove_dir_all(root.path());
}

#[test]
fn documents_are_listed_in_the_order_they_were_stored() {
    let (ws, root) = workspace("order");
    let ids: Vec<Digest> = (0..5)
        .map(|n| {
            ws.put_document(
                "note",
                &Note {
                    text: format!("note {n}"),
                    n,
                },
            )
            .unwrap()
        })
        .collect();
    assert_eq!(ws.documents_in_order("note").unwrap(), ids);
    let mut sorted = ids.clone();
    sorted.sort();
    assert_eq!(ws.document_ids("note").unwrap(), sorted);
    let _ = std::fs::remove_dir_all(root.path());
}

#[test]
fn measured_numbers_read_back_verified() {
    // A release records measured metrics: awkward floats, not tidy decimals.
    // Parsing must give back the very same number, or the address check on
    // read refuses a document that was stored correctly.
    let (workspace, root) = workspace("float-round-trip");
    let mut x = 0.123_456_789_f64;
    let measured: Vec<f64> = (0..500)
        .map(|_| {
            x = (x * 7.918_273_645 + 0.318_209_886).fract() * 3.0 - 1.0;
            x * 1e-3_f64.powi((x * 7.0) as i32)
        })
        .collect();
    let id = workspace.put_document("measurements", &measured).unwrap();
    workspace.commit().unwrap();
    // Read by another process's workspace: from the files, not from memory.
    let back: Vec<f64> = Workspace::at(&root)
        .get_document("measurements", &id)
        .unwrap()
        .unwrap();
    assert_eq!(back, measured);
}
