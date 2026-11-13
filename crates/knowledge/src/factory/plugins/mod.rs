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
//! | [`DB_WORKFLOW`] | complete a workflow through tools over a database | the final database state, diffed |
//! | [`REPOSITORY_BUGS`] | not built yet | the project's tests |
//! | [`FIRMWARE_EMULATOR`] | not built yet | the emulator |
//! | [`DOCUMENT_QUESTIONS`] | not built yet | the document's own text |

pub mod program_output;

use super::plugin::Plugin;

/// Predict the output of Lua 5.1 programs.
pub const PROGRAM_OUTPUT_LUA: &str = "program-output-lua";
/// Predict the output of Python 3 programs.
pub const PROGRAM_OUTPUT_PYTHON: &str = "program-output-python";
/// Tool-use workflows over a database, graded by final-state diff.
pub const DB_WORKFLOW: &str = "db-workflow";
/// Repository bug-fixing, graded by the project's tests.
pub const REPOSITORY_BUGS: &str = "repository-bugs";
/// Firmware tasks graded by the emulator.
pub const FIRMWARE_EMULATOR: &str = "firmware-emulator";
/// Questions about a document, graded against its text.
pub const DOCUMENT_QUESTIONS: &str = "document-questions";

/// The plugins this build can generate with.
#[must_use]
pub fn builtin() -> Vec<Box<dyn Plugin>> {
    vec![
        Box::new(program_output::ProgramOutput::lua()),
        Box::new(program_output::ProgramOutput::python()),
    ]
}
