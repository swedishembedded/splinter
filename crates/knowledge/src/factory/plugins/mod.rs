// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements environment generators with trustworthy
// verifiers for reinforcement learning, for its clients. If your team needs
// expertise in procedural environments for agent training, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The environment plugins, in the order the factory builds them.
//!
//! | Plugin | Skill | Oracle |
//! |---|---|---|
//! | [`PROGRAM_OUTPUT_LUA`], [`PROGRAM_OUTPUT_PYTHON`] | predict what a program prints | the interpreter |
//! | [`DB_WORKFLOW`] | complete a workflow through tools over a database | the final database state, diffed against the expected state |
//! | [`REPOSITORY_BUGS`] | fix injected faults in a small embedded library | a hidden test suite, with a canary |
//! | [`REPOSITORY_MUTANTS`] | fix faults mutated into real projects with their own suites | the project's whole suite with hidden tests, with a canary |
//! | [`FIRMWARE_EMULATOR`] | fix faults in a Cortex-M33 firmware library | the emulator's console output against a reference model, on hidden inputs |
//! | [`DOCUMENT_QUESTIONS`] | answer questions a document grounds without it | a number computed from, or a value stated by, the document |

mod db_tools;
pub mod db_workflow;
pub mod document_questions;
pub mod firmware_emulator;
pub mod program_output;
pub mod repository_bugs;
pub mod repository_mutants;

use std::path::PathBuf;

use super::host::Host;
use super::plugin::Plugin;

/// Predict the output of Lua 5.1 programs.
pub const PROGRAM_OUTPUT_LUA: &str = "program-output-lua";
/// Predict the output of Python 3 programs.
pub const PROGRAM_OUTPUT_PYTHON: &str = "program-output-python";
/// Tool-use workflows over a database, graded by final-state diff.
pub const DB_WORKFLOW: &str = "db-workflow";
/// Repository bug-fixing, graded by the project's tests.
pub const REPOSITORY_BUGS: &str = "repository-bugs";
/// Repository bug-fixing on real projects with their own suites.
pub const REPOSITORY_MUTANTS: &str = "repository-mutants";
/// Firmware tasks graded by the emulator.
pub const FIRMWARE_EMULATOR: &str = "firmware-emulator";
/// Questions about a document, graded against its text.
pub const DOCUMENT_QUESTIONS: &str = "document-questions";

/// Where the plugins that read material from disk find it.
#[derive(Clone, Debug, Default)]
pub struct Locations {
    /// The directory of repositories with their suites (`tasks/`): a
    /// `catalog.toml` and one directory per repository.
    pub repositories: Option<PathBuf>,
    /// The emulator's binary (`vemu`), run as a command and never modified.
    pub emulator: Option<PathBuf>,
    /// A directory of documents (Markdown or text) to compile questions from.
    pub documents: Option<PathBuf>,
}

/// The plugins this build can generate with, without any that need material
/// from disk.
#[must_use]
pub fn builtin() -> Vec<Box<dyn Plugin>> {
    builtin_with(&Locations::default(), &Host::default())
}

/// The plugins this build can generate with, including those whose material
/// `locations` names.
#[must_use]
pub fn builtin_with(locations: &Locations, host: &Host) -> Vec<Box<dyn Plugin>> {
    let mut plugins: Vec<Box<dyn Plugin>> = vec![
        Box::new(program_output::ProgramOutput::lua()),
        Box::new(program_output::ProgramOutput::python()),
        Box::new(db_workflow::DbWorkflow::new()),
        Box::new(repository_bugs::RepositoryBugs::new()),
    ];
    if let Some(emulator) = &locations.emulator {
        let host = host
            .clone()
            .with_variable("VEMU", &emulator.display().to_string());
        plugins.push(Box::new(firmware_emulator::FirmwareEmulator::new(host)));
    }
    if let Some(documents) = &locations.documents {
        plugins.push(Box::new(document_questions::DocumentQuestions::new(
            documents,
        )));
    }
    if let Some(root) = &locations.repositories {
        plugins.push(Box::new(repository_mutants::RepositoryMutants::new(
            root,
            host.clone(),
        )));
    }
    plugins
}
