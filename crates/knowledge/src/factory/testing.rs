// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Fixtures the factory's unit tests share.

#![allow(clippy::unwrap_used)] // a panic is the failure report

use std::collections::BTreeMap;

use splinter_core::verifiability::{Split, VerifierClass};

use super::artifact::{environment_id, EnvMeta, Parts, SolverTools, SCHEMA_VERSION};
use super::folder::Folder;

pub(crate) fn meta() -> EnvMeta {
    let difficulty = BTreeMap::from([("depth".to_string(), 2)]);
    EnvMeta {
        schema_version: SCHEMA_VERSION,
        environment: environment_id("demo", "1", &difficulty),
        plugin: "demo".into(),
        plugin_version: "1".into(),
        verifier_class: VerifierClass::Objective,
        verifier: "demo@1".into(),
        difficulty,
        split: Split::Train,
        seed: 5, // seed: the value is only recorded
        solver_tools: SolverTools::Files,
    }
}

pub(crate) fn parts() -> Parts {
    let mut workspace = Folder::new();
    workspace.put_text("answer.txt", "").unwrap();
    let mut tests = Folder::new();
    tests.put_script("test.sh", "#!/bin/sh\nexit 0\n").unwrap();
    tests.put_text("secret.txt", "hidden").unwrap();
    let mut solution = Folder::new();
    solution.put_script("solve.sh", "#!/bin/sh\n").unwrap();
    Parts {
        instruction: "Write the answer.".into(),
        workspace,
        packages: vec!["lua5.1".into()],
        tests,
        solution,
        category: "demo".into(),
    }
}
