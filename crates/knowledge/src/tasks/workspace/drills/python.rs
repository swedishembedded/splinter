// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements comprehension drills made by code from a
// codebase's own functions and tests, for its clients. If your team needs
// expertise in turning a repository into verified training tasks you can
// procure our services by sending an email to info@swedishembedded.com.

//! Python drills: the functions and the test calls read by the parser, and
//! the program that calls one.

use std::path::Path;
use std::time::Duration;

use serde::Deserialize;
use splinter_sandbox::{CommandCall, WorkspaceEnvironment};

use super::{Call, FunctionSource, Import, Language, Param, Read, RUN_LIMIT_SECS};
use crate::tasks::workspace::{run_quiet, WorkspaceError};

/// The program that reads functions and test calls by the parser.
const EXTRACT: &str = include_str!("extract.py");

#[derive(Deserialize)]
struct Document {
    functions: Vec<Function>,
    unreadable: Vec<String>,
}

#[derive(Deserialize)]
struct Function {
    file: String,
    name: String,
    line_start: usize,
    line_end: usize,
    params: Vec<Parameter>,
    modules: Vec<Module>,
    calls: Vec<CallText>,
}

#[derive(Deserialize)]
struct Parameter {
    name: String,
    kind: String,
    has_default: bool,
}

#[derive(Deserialize)]
struct Module {
    root: String,
    module: String,
}

#[derive(Deserialize)]
struct CallText {
    args: Vec<String>,
    kwargs: Vec<(String, String)>,
}

#[derive(serde::Serialize)]
struct Request<'a> {
    sources: &'a [String],
    tests: &'a [String],
}

/// Reads `sources` and the calls `tests` make of their functions, in `tree`.
pub(super) fn read(
    env: &WorkspaceEnvironment,
    tree: &Path,
    sources: &[String],
    tests: &[String],
) -> Result<Read, WorkspaceError> {
    let stdin = serde_json::to_string(&Request { sources, tests })?;
    let (result, _) = run_quiet(
        env,
        tree,
        &CommandCall {
            argv: ["python3", "-I", "-c", EXTRACT].map(String::from).into(),
            stdin: Some(stdin),
        },
        Some(Duration::from_secs(RUN_LIMIT_SECS * 6)),
    )?;
    if result.timed_out || result.exit_code != Some(0) {
        return Err(WorkspaceError::Unsupported {
            what: format!(
                "reading Python functions and test calls failed ({:?}): {}",
                result.exit_code,
                result.stderr.chars().take(300).collect::<String>()
            ),
        });
    }
    let document: Document = serde_json::from_str(&result.stdout)?;
    let functions = document
        .functions
        .into_iter()
        .map(|f| FunctionSource {
            file: f.file,
            name: f.name,
            language: Language::Python,
            line_start: f.line_start,
            line_end: f.line_end,
            params: f
                .params
                .into_iter()
                .map(|p| Param {
                    variadic: p.kind == "variadic",
                    name: p.name,
                    has_default: p.has_default,
                })
                .collect(),
            // The shortest root the tests import it by: the least path to
            // put before the program's imports.
            import: f
                .modules
                .into_iter()
                .min_by_key(|m| m.root.len())
                .map(|m| Import {
                    root: m.root,
                    module: m.module,
                }),
            calls: f
                .calls
                .into_iter()
                .map(|c| Call {
                    args: c.args,
                    kwargs: c.kwargs,
                })
                .collect(),
        })
        .collect();
    Ok(Read {
        functions,
        unreadable: document.unreadable,
    })
}

/// The program that calls `function` as `call` says and prints the result's
/// `repr`, or the exception it raised.
pub(super) fn program(function: &FunctionSource, call: &Call) -> String {
    let Some(import) = &function.import else {
        // A function with no test call has no import to take; only a call
        // with no arguments reaches here, and it takes its module from the
        // path of its file.
        return program_of(&function.file, "", function, call);
    };
    program_of(&import.module, &import.root, function, call)
}

fn program_of(module: &str, root: &str, function: &FunctionSource, call: &Call) -> String {
    let module = if module.ends_with(".py") {
        module
            .trim_end_matches(".py")
            .trim_end_matches("/__init__")
            .replace('/', ".")
    } else {
        module.to_string()
    };
    let mut text = String::new();
    if !root.is_empty() {
        text.push_str(&format!("import sys\n\nsys.path.insert(0, {root:?})\n"));
    }
    text.push_str(&format!(
        "from {module} import {name}\n\ntry:\n    print(repr({name}({args})))\nexcept Exception as error:\n    print(f\"raised {{type(error).__name__}}: {{error}}\")\n",
        name = function.name,
        args = call.text(),
    ));
    text
}
