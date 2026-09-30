// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Fixtures the verifier specs share: tasks, experiences, a python
//! environment in the process sandbox, and reading a verdict back.

// Each spec binary uses a subset of these helpers.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use splinter_lab::verifiers::{annotation, Verifier};
use splinter_sandbox::{
    Limits, ProcessSandbox, ResolvedEnvironment, RuntimeEnvironment, RuntimeRegistry,
};
use splinter_store::annotation::{AnnotationBody, Outcome, Strength};
use splinter_store::clock::FixedClock;
use splinter_store::experience::{
    Environment, Experience, Privileged, PrivilegedKind, Provenance, Task,
};
use sven_sdk::atif::{AgentProfile, Trajectory};

/// A scratch directory removed when dropped.
pub struct Scratch(pub PathBuf);

impl Scratch {
    /// A fresh scratch directory named for `test`.
    pub fn new(test: &str) -> Self {
        let path = std::env::temp_dir().join(format!("splinter-{test}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// python3 in the process sandbox under `limits`.
pub fn python(scratch: &Scratch, limits: Limits) -> RuntimeEnvironment {
    let sandbox = ProcessSandbox::new(&scratch.0, BTreeMap::new(), &[], limits);
    RuntimeEnvironment::new(&RuntimeRegistry::builtin(), "python3", Arc::new(sandbox)).unwrap()
}

/// The record a task keeps of `env`.
pub fn record(env: &RuntimeEnvironment) -> Environment {
    ResolvedEnvironment::Runtime(env.clone()).record().unwrap()
}

/// A task of `kind` in `environment` carrying `privileged`.
pub fn task(kind: &str, environment: Environment, privileged: Vec<Privileged>) -> Task {
    Task::new(kind, vec![], environment, "Solve the task.", privileged).unwrap()
}

/// A reference answer, as a privileged item.
pub fn reference(text: &str) -> Privileged {
    Privileged {
        kind: PrivilegedKind::Reference,
        content: text.into(),
        span: None,
    }
}

/// An experience of `task` answered with `output` by `solver`.
pub fn experience(task: &Task, output: Option<&str>, solver: &str) -> Experience {
    let profile = AgentProfile {
        name: "t".into(),
        version: "1".into(),
        model_name: None,
        tool_definitions: None,
        extra: None,
    };
    Experience::new(
        task.clone(),
        Trajectory::new("ATIF-v1.7", profile),
        output.map(str::to_string),
        Provenance::new(solver, &FixedClock::new("2026-09-30T00:00:00.000Z")),
    )
    .unwrap()
}

/// The verdict `verifier` gives `exp`, read back from its annotation.
pub fn verdict(
    verifier: &dyn Verifier,
    task: &Task,
    exp: &Experience,
) -> (Outcome, Strength, serde_json::Value) {
    let note = annotation(verifier, task, exp).unwrap();
    assert_eq!(note.experience, exp.id().unwrap());
    assert_eq!(note.producer, verifier.producer());
    match note.body {
        AnnotationBody::Verdict {
            outcome,
            strength,
            evidence,
        } => (outcome, strength, evidence),
        other => panic!("not a verdict: {other:?}"),
    }
}
