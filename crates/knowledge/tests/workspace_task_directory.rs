// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the conversion of agent-written task
// directories into verified training tasks, for its clients. If your team
// needs expertise in turning generated tasks into executable, reproducible
// benchmarks, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: a task directory (`instruction.md`, `environment/`,
//! `solution/solve.sh`, `tests/test.sh`, optionally the task's own
//! `harness.yaml`) is a workspace task.
//!
//! * the solver's tree is `environment/` plus the instruction as `task.md`;
//!   `tests/test.sh` never reaches it and is the verdict, run in the finished
//!   tree with `TESTS_DIR` set to the tests;
//! * the reference solution is run on a copy of `environment/` and its change
//!   is the task's reference: the verifier passes it and fails the untouched
//!   environment;
//! * the task's own harness travels with the task, hidden from the solver;
//! * a directory that is not laid out as the contract says is refused with
//!   the missing piece named, never dropped.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use common::{write, Repo};
use splinter_core::annotation::Outcome;
use splinter_core::clock::FixedClock;
use splinter_core::experience::{Experience, PrivilegedKind, Provenance};
use splinter_core::kinds::SOLVE_PROCESS;
use splinter_eval::verifiers::workspace::WorkspaceVerifier;
use splinter_eval::verifiers::Verifier;
use splinter_knowledge::tasks::workspace::{
    Held, TaskDirectory, WorkspaceGenerator, WorkspaceTask,
};
use splinter_sandbox::{Limits, ProcessSandbox};

const AT: &str = "2026-10-26T09:00:00.000Z";

const HARNESS: &str = "api_version: \"2\"\nname: calc-solve\n";

/// A task directory as the invent harness leaves one: calc.sh must print the
/// sum of one and two, and the expected value lives with the tests.
fn task_directory(root: &Path) -> PathBuf {
    let dir = root.join("calc-sum");
    write(
        &dir.join("instruction.md"),
        "Make calc.sh print the sum of one and two.\n",
    );
    write(&dir.join("environment/calc.sh"), "echo 1\n");
    write(&dir.join("tests/want"), "3\n");
    write(
        &dir.join("tests/test.sh"),
        "#!/bin/sh\n[ \"$(sh calc.sh)\" = \"$(cat \"$TESTS_DIR/want\")\" ]\n",
    );
    write(
        &dir.join("solution/solve.sh"),
        "#!/bin/sh\necho 'echo 3' > calc.sh\n",
    );
    write(&dir.join("harness.yaml"), HARNESS);
    dir
}

struct Rig {
    repo: Repo,
    generator: WorkspaceGenerator,
}

fn rig(name: &str) -> Rig {
    let repo = Repo::of_files(name, &[("README.md", "x\n")]);
    let calls = repo.scratch.0.join("calls");
    fs::create_dir_all(&calls).unwrap();
    let sandbox = ProcessSandbox::new(&calls, BTreeMap::new(), &[], Limits::default());
    let generator = WorkspaceGenerator::new(
        Arc::new(sandbox),
        repo.host.clone(),
        repo.store.clone(),
        repo.scratch.0.join("work"),
    );
    Rig { repo, generator }
}

fn admit(rig: &Rig, dir: &Path) -> WorkspaceTask {
    let task = TaskDirectory::read(dir).unwrap();
    let mut generation = rig
        .generator
        .import_task_directory(&task, &Held::none(), &FixedClock::new(AT))
        .unwrap();
    assert_eq!(generation.admitted.len(), 1, "{:?}", generation.refused);
    generation.admitted.remove(0)
}

fn verdict(rig: &Rig, admitted: &WorkspaceTask, patch: &str) -> Outcome {
    let verifier = WorkspaceVerifier::new(
        vec![rig.generator.environment(&admitted.snapshot)],
        Arc::new(rig.repo.store.clone()),
        rig.repo.scratch.0.join("verify"),
    );
    let provenance = Provenance::new("test/candidate", &FixedClock::new(AT));
    let experience =
        Experience::answered_without_a_run(admitted.task.clone(), patch, provenance).unwrap();
    verifier
        .verify(&admitted.task, &experience)
        .unwrap()
        .outcome
}

#[test]
fn the_reference_solution_passes_the_tests_and_the_untouched_environment_fails_them() {
    let rig = rig("task-directory");
    let admitted = admit(&rig, &task_directory(&rig.repo.scratch.0));

    let reference = admitted
        .task
        .privileged
        .iter()
        .find(|p| p.kind == PrivilegedKind::Reference)
        .expect("the reference solution's change is the task's reference");
    assert_eq!(verdict(&rig, &admitted, &reference.content), Outcome::Pass);
    assert_eq!(verdict(&rig, &admitted, ""), Outcome::Fail);
    assert_eq!(admitted.family, "calc-sum");
}

#[test]
fn the_tasks_own_harness_travels_with_it_and_the_instruction_is_the_solvers_task_md() {
    let rig = rig("task-directory-process");
    let admitted = admit(&rig, &task_directory(&rig.repo.scratch.0));

    let process = admitted
        .task
        .privileged
        .iter()
        .find(|p| p.kind == PrivilegedKind::Other(SOLVE_PROCESS.into()))
        .expect("the task's harness is kept");
    assert_eq!(process.content, HARNESS);
    let archive = rig.generator.archive_of(&admitted.snapshot).unwrap();
    let listing = std::process::Command::new("tar")
        .arg("-tf")
        .arg(&archive)
        .output()
        .unwrap();
    let listing = String::from_utf8_lossy(&listing.stdout);
    assert!(listing.contains("task.md"), "{listing}");
    assert!(listing.contains("calc.sh"), "{listing}");
    assert!(
        !listing.contains("test.sh"),
        "the tests stay hidden: {listing}"
    );
}

#[test]
fn a_directory_without_tests_is_refused_naming_the_missing_file() {
    let rig = rig("task-directory-malformed");
    let dir = task_directory(&rig.repo.scratch.0);
    fs::remove_file(dir.join("tests/test.sh")).unwrap();

    let error = TaskDirectory::read(&dir).unwrap_err().to_string();
    assert!(error.contains("tests/test.sh"), "{error}");
}
