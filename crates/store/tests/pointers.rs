// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable, content-addressed experience
// stores for learning agents, for its clients. If your team needs expertise
// in training-data lineage or crash-safe storage, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: a pointer is its history. A move succeeds only from the value the
//! caller saw, exactly one of several racing moves wins, nothing is ever
//! overwritten, and moving back is one more move.
#![allow(clippy::unwrap_used)]

use splinter_store::experiences::StoreError;
use splinter_store::workspace::Workspace;
use splinter_store::StateRoot;

fn root(name: &str) -> StateRoot {
    let root = StateRoot::new(
        std::env::temp_dir().join(format!("splinter-ptr-{name}-{}", std::process::id())),
    );
    let _ = std::fs::remove_dir_all(root.path());
    root
}

#[test]
fn a_pointer_moves_from_the_value_the_caller_saw_and_keeps_its_history() {
    let root = root("history");
    let ws = Workspace::at(&root);
    assert_eq!(ws.pointer("default").unwrap(), None);
    assert_eq!(
        ws.move_pointer("default", None, "release-1", "t1").unwrap(),
        1
    );
    assert_eq!(
        ws.move_pointer("default", Some("release-1"), "release-2", "t2")
            .unwrap(),
        2
    );
    // Rolling back is one more move, to an older value.
    assert_eq!(
        ws.move_pointer("default", Some("release-2"), "release-1", "t3")
            .unwrap(),
        3
    );

    let other = Workspace::at(&root);
    assert_eq!(
        other.pointer("default").unwrap(),
        Some((3, "release-1".into()))
    );
    let history = other.pointer_history("default").unwrap();
    assert_eq!(
        history
            .iter()
            .map(|m| (m.value.as_str(), m.previous.as_deref(), m.at.as_str()))
            .collect::<Vec<_>>(),
        vec![
            ("release-1", None, "t1"),
            ("release-2", Some("release-1"), "t2"),
            ("release-1", Some("release-2"), "t3"),
        ]
    );
    assert_eq!(
        other.pointer("other-alias").unwrap(),
        None,
        "pointers are separate"
    );
    let _ = std::fs::remove_dir_all(root.path());
}

#[test]
fn a_move_from_a_stale_value_is_refused_and_changes_nothing() {
    let root = root("stale");
    let ws = Workspace::at(&root);
    ws.move_pointer("default", None, "a", "t").unwrap();
    ws.move_pointer("default", Some("a"), "b", "t").unwrap();
    match ws.move_pointer("default", Some("a"), "c", "t") {
        Err(StoreError::PointerConflict {
            expected, found, ..
        }) => {
            assert_eq!(expected.as_deref(), Some("a"));
            assert_eq!(found.as_deref(), Some("b"));
        }
        other => panic!("{other:?}"),
    }
    assert!(matches!(
        ws.move_pointer("fresh", Some("x"), "y", "t"),
        Err(StoreError::PointerConflict { .. })
    ));
    assert_eq!(ws.pointer("default").unwrap(), Some((2, "b".into())));
    let _ = std::fs::remove_dir_all(root.path());
}

#[test]
fn of_several_processes_moving_one_pointer_at_once_exactly_one_wins() {
    let root = root("race");
    Workspace::at(&root)
        .move_pointer("default", None, "start", "t")
        .unwrap();
    let handles: Vec<_> = (0..8)
        .map(|n| {
            let root = root.clone();
            std::thread::spawn(move || {
                let ws = Workspace::at(&root);
                ws.move_pointer("default", Some("start"), &format!("winner-{n}"), "t")
            })
        })
        .collect();
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(
        results.iter().filter(|r| r.is_ok()).count(),
        1,
        "{results:?}"
    );
    assert!(results
        .iter()
        .filter(|r| r.is_err())
        .all(|r| matches!(r, Err(StoreError::PointerConflict { .. }))));
    let history = Workspace::at(&root).pointer_history("default").unwrap();
    assert_eq!(
        history.len(),
        2,
        "the start and the one winner, nothing lost or doubled"
    );
    let _ = std::fs::remove_dir_all(root.path());
}

#[test]
fn pointers_are_listed_by_prefix_with_their_current_values() {
    let root = root("list");
    let ws = Workspace::at(&root);
    ws.move_pointer("queue-a", None, "one", "t").unwrap();
    ws.move_pointer("queue-a", Some("one"), "two", "t").unwrap();
    ws.move_pointer("queue-b", None, "x", "t").unwrap();
    ws.move_pointer("alias", None, "y", "t").unwrap();
    assert_eq!(
        ws.pointers("queue-").unwrap(),
        vec![
            ("queue-a".to_owned(), "two".to_owned()),
            ("queue-b".to_owned(), "x".to_owned())
        ]
    );
    let _ = std::fs::remove_dir_all(root.path());
}
