// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements content-addressed task stores that every
// training example traces back to, for its clients. If your team needs
// expertise in training-data lineage or durable learning state, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a generated task is stored once under its own address and read
//! back verified; a task set names stored tasks with how each was
//! generated, is content-addressed like an experience set, and refuses a
//! task the store does not hold.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::fs;
use std::path::PathBuf;

use splinter_store::digest::Digest;
use splinter_store::experience::{Environment, Privileged, PrivilegedKind, Task};
use splinter_store::experiences::StoreError;
use splinter_store::tasks::{TaskEntry, TaskSet, TaskStore};
use splinter_store::StateRoot;

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("splinter-tasks-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        Self(path)
    }

    fn store(&self) -> TaskStore {
        TaskStore::open(&StateRoot::new(&self.0))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn task(instruction: &str) -> Task {
    Task::new(
        "recall",
        vec![],
        Environment::closed_book(),
        instruction,
        vec![Privileged {
            kind: PrivilegedKind::Reference,
            content: "115200 baud".into(),
            span: None,
        }],
    )
    .unwrap()
}

fn entry(task: &Task) -> TaskEntry {
    TaskEntry {
        task: task.task.id.clone(),
        generator: Some("splinter-knowledge/model-tasks@1:scripted".into()),
        prompt: Some(Digest::of(b"prompt")),
        variant_of: None,
        subject: None,
    }
}

#[test]
fn a_task_is_stored_once_and_read_back_verified() {
    let scratch = Scratch::new("once");
    let store = scratch.store();
    let baud = task("At what baud rate does the console run?");
    let id = store.put(&baud).unwrap();
    assert_eq!(id, baud.task.id);
    assert_eq!(store.put(&baud).unwrap(), id, "storing it again is a no-op");
    assert_eq!(store.get(&id).unwrap(), baud);
    assert_eq!(store.list().unwrap(), vec![id.clone()]);

    // A stored task whose content no longer addresses it is corruption.
    let path = scratch
        .0
        .join("tasks/objects")
        .join(format!("{}.json", id.hex()));
    let tampered = fs::read_to_string(&path)
        .unwrap()
        .replace("console", "modem");
    fs::write(&path, tampered).unwrap();
    assert!(store.get(&id).is_err(), "a tampered task is refused");

    let unknown = Digest::of(b"no such task");
    assert!(matches!(
        store.get(&unknown),
        Err(StoreError::UnknownTask(_))
    ));
}

#[test]
fn a_task_set_is_content_addressed_and_names_only_stored_tasks() {
    let scratch = Scratch::new("sets");
    let store = scratch.store();
    let first = task("At what baud rate does the console run?");
    let second = task("How many data bits does the console use?");
    store.put(&first).unwrap();
    store.put(&second).unwrap();

    let set = TaskSet {
        name: "recall from the manual".into(),
        members: vec![entry(&first), entry(&second)],
    };
    let id = store.put_set(&set).unwrap();
    assert_eq!(store.put_set(&set).unwrap(), id, "same content, same id");
    assert_eq!(store.get_set(&id).unwrap(), set);
    assert_eq!(store.list_sets().unwrap(), vec![id]);

    let unstored = task("Which pin carries the console transmit line?");
    let refused = store
        .put_set(&TaskSet {
            name: "dangling".into(),
            members: vec![entry(&unstored)],
        })
        .unwrap_err();
    assert!(matches!(refused, StoreError::UnknownTask(_)), "{refused}");
    let twice = store.put_set(&TaskSet {
        name: "twice".into(),
        members: vec![entry(&first), entry(&first)],
    });
    assert!(twice.is_err(), "a task is listed once");
}

/// A task may declare the concepts it exercises: the declaration is part
/// of what the task is (its address), survives the store, and is carried by
/// every experience of it; a task that declares none keeps the address it
/// had before tasks could declare any.
#[test]
fn declared_concepts_are_part_of_the_task() {
    let scratch = Scratch::new("concepts");
    let store = scratch.store();
    let plain = task("At what baud rate does the console run?");
    let declared = plain
        .with_concepts(["console", "serial", "console"])
        .unwrap();
    assert_eq!(declared.concepts, ["console", "serial"], "each once");
    assert_ne!(declared.task.id, plain.task.id);
    assert_eq!(
        plain.with_concepts(Vec::<String>::new()).unwrap().task.id,
        plain.task.id,
        "declaring nothing is the undeclared task"
    );
    assert!(plain.with_concepts([" "]).is_err(), "a concept has a name");
    let text = serde_json::to_string(&plain).unwrap();
    assert!(!text.contains("concepts"), "{text}");

    store.put(&declared).unwrap();
    assert_eq!(store.get(&declared.task.id).unwrap(), declared);
    let reviewed = declared
        .with_privileged(Privileged {
            kind: PrivilegedKind::Critique,
            content: "check the baud rate".into(),
            span: None,
        })
        .unwrap();
    assert_eq!(reviewed.concepts, declared.concepts);
    assert!(reviewed.same_apart_from_critiques(&declared));
    assert!(!plain.same_apart_from_critiques(&declared));
}

/// A variant is the same fact asked in other words: everything but the
/// instruction is the original's, and the set records which task it varies
/// without changing the address of a set that records none.
#[test]
fn a_variant_asks_in_other_words_and_a_set_records_what_it_varies() {
    let scratch = Scratch::new("variants");
    let store = scratch.store();
    let original = task("At what baud rate does the console run?")
        .with_concepts(["console"])
        .unwrap();
    let variant = original
        .with_instruction("Which baud rate is the console set to?")
        .unwrap();
    assert_ne!(variant.task.id, original.task.id);
    assert_eq!(variant.task.kind, original.task.kind);
    assert_eq!(variant.privileged, original.privileged);
    assert_eq!(variant.concepts, original.concepts);
    assert!(variant.with_instruction(" ").is_err(), "a blank question");

    store.put(&original).unwrap();
    store.put(&variant).unwrap();
    let plain = TaskSet {
        name: "originals".into(),
        members: vec![entry(&original)],
    };
    let varied = TaskSet {
        name: "originals".into(),
        members: vec![TaskEntry {
            variant_of: Some(original.task.id.clone()),
            subject: Some("console".into()),
            ..entry(&variant)
        }],
    };
    let id = store.put_set(&varied).unwrap();
    assert_eq!(store.get_set(&id).unwrap(), varied);
    let canonical = |set: &TaskSet| serde_json::to_string(set).unwrap();
    assert!(!canonical(&plain).contains("variant_of"));
    assert!(!canonical(&plain).contains("subject"));
    assert!(canonical(&varied).contains("variant_of"));
}

#[test]
fn a_retry_task_returns_to_the_task_it_retried_without_its_critiques() {
    let original = task("At what baud rate does the console run?");
    let retried = original
        .with_privileged(Privileged {
            kind: PrivilegedKind::Critique,
            content: "The rate is wrong.".into(),
            span: None,
        })
        .unwrap();
    assert_ne!(retried.task.id, original.task.id);
    assert_eq!(retried.without_critiques().unwrap(), original);
    assert_eq!(original.without_critiques().unwrap(), original);
}
