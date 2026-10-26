// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the conversion of agent-written task
// directories into verified training tasks, for its clients. If your team
// needs expertise in turning generated tasks into executable, reproducible
// benchmarks, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Task directories as workspace tasks.
//!
//! A task directory is what an agent that invents tasks leaves:
//!
//! ```text
//! instruction.md      what the solver is asked
//! environment/        the files the solver starts with
//! solution/solve.sh   a reference solution, run in a copy of environment/
//! tests/test.sh       the verdict, run in the finished workspace; exit 0 = solved
//! harness.yaml        the process that solves this task (optional)
//! ```
//!
//! The solver's tree is `environment/` with the instruction as `task.md`.
//! `tests/` is the hidden overlay: laid over the finished tree under
//! `.tests/`, where `tests/test.sh` runs with `TESTS_DIR` set to it. The
//! directory is held to the bar of every workspace task (the tests fail on
//! the tree the solver is given, on every run, the request does not quote
//! the tests) and its reference solution is run on a copy of `environment/`:
//! the change it makes is the task's reference, and the workspace verifier
//! must pass it. A task's own `harness.yaml` travels with it as the
//! privileged [`SOLVE_PROCESS`] item.
//!
//! A directory that does not hold what the contract names is refused with
//! the missing piece, never skipped.

use std::path::{Path, PathBuf};

use splinter_core::clock::Clock;
use splinter_sandbox::CommandCall;

use super::files::{copy_dir, files_under, io, read_files, refuse, skip_nothing};
use super::held::Held;
use super::{Fixture, WorkspaceError, WorkspaceGeneration, WorkspaceGenerator};

/// The tree-relative directory the tests are laid over the finished tree
/// at: withheld from the solver and protected from its change.
const TESTS_AT: &str = ".tests";

/// The file in the solver's tree that holds the request.
const REQUEST_FILE: &str = "task.md";

/// Runs the tests in the finished tree the way the task's author ran them.
const RUN_TESTS: &str = "TESTS_DIR=\"$PWD/.tests\" exec bash .tests/test.sh";

/// Runs the reference solution in a copy of the environment.
const RUN_SOLUTION: &str =
    "SOLUTION_DIR=\"$PWD/.oracle/solution\" exec bash .oracle/solution/solve.sh";

/// A task directory, read and found laid out as the contract says.
#[derive(Clone, Debug)]
pub struct TaskDirectory {
    /// The task's directory.
    pub dir: PathBuf,
    /// Its name: the directory's, which is also its family.
    pub name: String,
    /// The request.
    pub instruction: String,
    /// The task's own solving process (a sven harness document), when it
    /// has one.
    pub process: Option<String>,
}

impl TaskDirectory {
    /// The task in `dir`, when it holds what the contract names.
    ///
    /// # Errors
    ///
    /// [`WorkspaceError::Fixture`] naming the missing or unusable piece.
    pub fn read(dir: &Path) -> Result<Self, WorkspaceError> {
        files_under(dir)?;
        let text = |name: &str| -> Result<String, WorkspaceError> {
            let path = dir.join(name);
            if !path.is_file() {
                return refuse(dir, format!("{name} is missing"));
            }
            std::fs::read_to_string(&path).map_err(io(&path))
        };
        let instruction = text("instruction.md")?.trim().to_string();
        if instruction.is_empty() {
            return refuse(dir, "instruction.md is empty");
        }
        if !dir.join("environment").is_dir() {
            return refuse(
                dir,
                "environment/ is missing (it may be empty, but it must exist)",
            );
        }
        for reserved in [REQUEST_FILE, TESTS_AT] {
            if dir.join("environment").join(reserved).exists() {
                return refuse(
                    dir,
                    format!("environment/ holds {reserved}, which the host lays out"),
                );
            }
        }
        if text("solution/solve.sh")?.trim().is_empty() {
            return refuse(dir, "solution/solve.sh is empty");
        }
        if text("tests/test.sh")?.trim().is_empty() {
            return refuse(dir, "tests/test.sh is empty");
        }
        let process = match dir.join("harness.yaml").is_file() {
            true => Some(text("harness.yaml")?).filter(|p| !p.trim().is_empty()),
            false => None,
        };
        let name = dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if name.is_empty() {
            return refuse(dir, "it has no name");
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            name,
            instruction,
            process,
        })
    }
}

fn sh(script: &str) -> CommandCall {
    CommandCall {
        argv: vec!["sh".into(), "-c".into(), script.into()],
        stdin: None,
    }
}

impl Fixture {
    /// `task` as an authored fixture, its solver's tree laid out under
    /// `work`; the reference change is not yet known.
    ///
    /// # Errors
    ///
    /// [`WorkspaceError`] when the directory cannot be copied.
    pub fn task_directory(task: &TaskDirectory, work: &Path) -> Result<Self, WorkspaceError> {
        let tree = work.join("tree");
        copy_dir(&task.dir.join("environment"), &tree, skip_nothing)?;
        let request = tree.join(REQUEST_FILE);
        std::fs::write(&request, &task.instruction).map_err(io(&request))?;
        let hidden = read_files(&task.dir.join("tests"))?
            .into_iter()
            .map(|(file, bytes)| (format!("{TESTS_AT}/{file}"), bytes))
            .collect();
        Ok(Self {
            family: task.name.clone(),
            instruction: task.instruction.clone(),
            tree,
            hidden,
            fail_to_pass: vec![sh(RUN_TESTS)],
            pass_to_pass: Vec::new(),
            protected: vec![TESTS_AT.to_string()],
            reference: None,
            expectations: Vec::new(),
            pinned: Vec::new(),
            hidden_strength: None,
            process: task.process.clone(),
        })
    }
}

impl WorkspaceGenerator {
    /// `task` as a workspace task: its reference solution run on a copy of
    /// its environment, its tests held to the bar of every workspace task,
    /// and the verifier shown to pass the reference change.
    ///
    /// # Errors
    ///
    /// [`WorkspaceError`] when the directory cannot be laid out or its
    /// reference solution does not run on the host. A task that does not
    /// meet the bar is counted under its reason in the result, not an error.
    pub fn import_task_directory(
        &self,
        task: &TaskDirectory,
        held: &Held,
        clock: &dyn Clock,
    ) -> Result<WorkspaceGeneration, WorkspaceError> {
        let work = self.temp_dir("splinter-task-directory-")?;
        let mut fixture = Fixture::task_directory(task, work.path())?;
        let change = self
            .solution_change(
                &fixture.tree,
                &task.dir.join("solution"),
                &[],
                &sh(RUN_SOLUTION),
            )
            .or_else(|why| refuse(&task.dir, why))?;
        fixture.reference = Some(change);
        self.import_fixture(&fixture, held, clock)
    }
}
