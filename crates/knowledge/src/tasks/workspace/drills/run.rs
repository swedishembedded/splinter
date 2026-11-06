// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements comprehension drills made by code from a
// codebase's own functions and tests, for its clients. If your team needs
// expertise in turning a repository into verified training tasks you can
// procure our services by sending an email to info@swedishembedded.com.

//! Running a drill's program against the clean snapshot, and the admission
//! that rests on what the run shows: it completes, prints something bounded
//! and informative, and prints the same twice.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use splinter_sandbox::{CommandCall, WorkspaceEnvironment};

use super::{lua, python, Call, DrillReason, FunctionSource, Language, RUN_LIMIT_SECS};
use crate::tasks::shown::{undefined_by_the_language, MAX_COMPUTED_OUTPUT};
use crate::tasks::workspace::{run_quiet, FreshTree, WorkspaceError};

/// A program that was run and is admitted.
pub(super) struct Admitted {
    /// The program text, as it is shown and was run.
    pub program: String,
    /// What it prints, without trailing whitespace.
    pub reference: String,
    /// What the interpreter says it is.
    pub interpreter: String,
}

/// What a printed address in a result looks like: a value that is not
/// defined by the program, since the next run puts it elsewhere.
const ADDRESS_MARKS: [&str; 2] = [" at 0x", "table: 0x"];

/// Runs candidate programs, each in a tree of its own laid out afresh from
/// the snapshot: a function that writes files leaves nothing the next
/// program could read.
pub(super) struct Runner<'a> {
    env: &'a WorkspaceEnvironment,
    archive: PathBuf,
    scratch: PathBuf,
    interpreters: BTreeMap<Language, String>,
}

impl<'a> Runner<'a> {
    pub(super) fn new(env: &'a WorkspaceEnvironment, archive: PathBuf, scratch: PathBuf) -> Self {
        Self {
            env,
            archive,
            scratch,
            interpreters: BTreeMap::new(),
        }
    }

    /// Whether the call of `function` is admitted as a drill, and the
    /// reference it has; the reason it is not, otherwise. For Lua a result
    /// that is a table is printed by a small serialiser the program carries,
    /// since a table's own text is an address.
    pub(super) fn admit(
        &mut self,
        function: &FunctionSource,
        call: &Call,
    ) -> Result<Result<Admitted, DrillReason>, WorkspaceError> {
        let mut program = match function.language {
            Language::Python => python::program(function, call),
            Language::Lua => lua::program(function, call, false),
        };
        let tree = FreshTree::lay(self.env, &self.archive, &self.scratch)?;
        let run = |program: &str, hash_seed: u32| -> Result<_, WorkspaceError> {
            let (result, _) = run_quiet(
                self.env,
                tree.path(),
                &command(function.language, program, hash_seed),
                Some(Duration::from_secs(RUN_LIMIT_SECS)),
            )?;
            Ok(result)
        };
        if undefined_by_the_language(runtime_of(function.language))
            .iter()
            .any(|name| program.contains(*name))
        {
            return Ok(Err(DrillReason::UndefinedByTheLanguage));
        }
        let mut first = run(&program, 1)?;
        if function.language == Language::Lua
            && first.exit_code == Some(0)
            && first.stdout.contains("table: 0x")
        {
            program = lua::program(function, call, true);
            first = run(&program, 1)?;
        }
        if first.timed_out || first.exit_code != Some(0) {
            return Ok(Err(DrillReason::DoesNotRun));
        }
        if first.stdout_truncated {
            return Ok(Err(DrillReason::OutputTooLong));
        }
        let printed = first.stdout.trim_end().to_string();
        if printed.trim().is_empty() {
            return Ok(Err(DrillReason::EmptyOutput));
        }
        if printed.chars().count() > MAX_COMPUTED_OUTPUT {
            return Ok(Err(DrillReason::OutputTooLong));
        }
        if printed == "None" {
            return Ok(Err(DrillReason::Uninformative));
        }
        if ADDRESS_MARKS.iter().any(|mark| printed.contains(mark)) {
            return Ok(Err(DrillReason::Nondeterministic));
        }
        let second = run(&program, 2)?;
        if second.timed_out || second.exit_code != Some(0) || second.stdout.trim_end() != printed {
            return Ok(Err(DrillReason::Nondeterministic));
        }
        let interpreter = self.interpreter(function.language, &tree)?;
        Ok(Ok(Admitted {
            program,
            reference: printed,
            interpreter,
        }))
    }

    /// The first line the interpreter prints for its version, asked once per
    /// language, in the tree the first program of it ran in.
    fn interpreter(
        &mut self,
        language: Language,
        tree: &FreshTree,
    ) -> Result<String, WorkspaceError> {
        if let Some(known) = self.interpreters.get(&language) {
            return Ok(known.clone());
        }
        let argv: &[&str] = match language {
            Language::Python => &["python3", "--version"],
            Language::Lua => &["lua", "-v"],
        };
        let (result, _) = run_quiet(
            self.env,
            tree.path(),
            &CommandCall {
                argv: argv.iter().map(|a| (*a).to_string()).collect(),
                stdin: None,
            },
            Some(Duration::from_secs(RUN_LIMIT_SECS)),
        )?;
        let said = format!("{}{}", result.stdout, result.stderr);
        let line = said.lines().next().unwrap_or("").trim().to_string();
        let line = if line.is_empty() {
            "unknown".to_string()
        } else {
            line
        };
        self.interpreters.insert(language, line.clone());
        Ok(line)
    }
}

/// The runtime name the language's undefined behaviours are listed under.
fn runtime_of(language: Language) -> &'static str {
    match language {
        Language::Python => "python3",
        Language::Lua => "lua",
    }
}

/// The command that runs `program` read from standard input. Python's string
/// hash is seeded per process, so two runs under different seeds print the
/// same only when nothing depends on it.
fn command(language: Language, program: &str, hash_seed: u32) -> CommandCall {
    let argv: Vec<String> = match language {
        Language::Python => vec![
            "env".to_string(),
            format!("PYTHONHASHSEED={hash_seed}"),
            "python3".to_string(),
            "-".to_string(),
        ],
        Language::Lua => vec!["lua".to_string(), "-".to_string()],
    };
    CommandCall {
        argv,
        stdin: Some(program.to_string()),
    }
}
