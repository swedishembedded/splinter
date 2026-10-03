// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bounded, reproducible capture of tool runs
// as learning evidence, for its clients. If your team needs expertise in
// learning command-line tools or process supervision, you can procure our
// services by sending an email to info@swedishembedded.com.

//! One run of a program captured as a command source: how Splinter will
//! learn a command-line tool, the captured output being the evidence later
//! answers are checked against.
//!
//! The program runs through Splinter's one process runner
//! (`splinter_sandbox::process`): directly (no shell), with exactly the
//! environment it is given, standard input closed, in its own process
//! group. Each output stream is kept up to a cap and read to its end past
//! it, so the program never blocks on a full pipe; a stream cut at the cap
//! is recorded as truncated. A run that outlives its timeout is recorded as
//! timed out, and when the run ends its whole process group is killed.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use splinter_core::clock::Clock;
use splinter_core::source::{CapturedSource, Origin, PartContent};
use splinter_sandbox::process::{environment_from, run, ProcessSpec, Stream};

use super::{io, is_text, utf8, CaptureError};

/// The default time a run may take before it is killed.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// The default cap on each captured output stream, in bytes.
pub const DEFAULT_OUTPUT_CAP: usize = 1024 * 1024;

/// The variables [`default_environment`] passes through from the caller's
/// environment.
pub const DEFAULT_ENV_ALLOWLIST: &[&str] = &["PATH", "HOME", "LANG"];

/// The `TERM` [`default_environment`] sets: no colour, no cursor control,
/// so captured output is plain text.
pub const DEFAULT_TERM: &str = "dumb";

/// One run to capture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommandSpec {
    /// The program and its arguments; the program is looked up on the
    /// `PATH` in [`CommandSpec::env`] when it has no `/`.
    pub argv: Vec<String>,
    /// The working directory.
    pub cwd: PathBuf,
    /// The complete environment of the run; nothing else is inherited.
    pub env: BTreeMap<String, String>,
    /// How long the run may take before its process group is killed.
    pub timeout: Duration,
    /// The most bytes kept of each output stream.
    pub output_cap: usize,
}

impl CommandSpec {
    /// A run of `argv` in `cwd` with environment `env`, under
    /// [`DEFAULT_TIMEOUT`] and [`DEFAULT_OUTPUT_CAP`].
    #[must_use]
    pub fn new(argv: Vec<String>, cwd: impl Into<PathBuf>, env: BTreeMap<String, String>) -> Self {
        Self {
            argv,
            cwd: cwd.into(),
            env,
            timeout: DEFAULT_TIMEOUT,
            output_cap: DEFAULT_OUTPUT_CAP,
        }
    }
}

/// The default environment of a run: the [`DEFAULT_ENV_ALLOWLIST`]
/// variables found in `parent` (the caller's environment, as the caller
/// reads it) and `TERM` set to [`DEFAULT_TERM`]. Everything else in
/// `parent` is dropped.
#[must_use]
pub fn default_environment(
    parent: impl IntoIterator<Item = (String, String)>,
) -> BTreeMap<String, String> {
    let mut env = environment_from(parent, DEFAULT_ENV_ALLOWLIST);
    env.insert("TERM".into(), DEFAULT_TERM.into());
    env
}

/// Captures one run of `spec` as a command source with two parts,
/// `stdout` and `stderr` (`text/plain` when UTF-8 text, else
/// `application/octet-stream`). The origin records the argv, the absolute
/// working directory, the exit code, whether the run timed out, and
/// whether either stream was truncated at the cap. A program that exits
/// on time but leaves a process holding its output open is timed out too.
pub fn capture_command(
    spec: &CommandSpec,
    clock: &dyn Clock,
) -> Result<CapturedSource, CaptureError> {
    let cwd = spec.cwd.canonicalize().map_err(io(&spec.cwd))?;
    let output = run(&ProcessSpec::new(
        spec.argv.clone(),
        &cwd,
        spec.env.clone(),
        spec.timeout,
        spec.output_cap,
    ))?;
    let (stdout, stderr) = (output.stdout, output.stderr);

    let origin = Origin::Command {
        argv: spec.argv.clone(),
        cwd: utf8(&cwd)?.to_string(),
        exit_code: output.exit_code,
        timed_out: output.timed_out,
        stdout_truncated: stdout.truncated,
        stderr_truncated: stderr.truncated,
    };
    let part = |name: &str, stream: Stream| PartContent {
        name: name.to_string(),
        media_type: if is_text(&stream.bytes) {
            "text/plain"
        } else {
            "application/octet-stream"
        }
        .to_string(),
        bytes: stream.bytes,
    };
    Ok(CapturedSource::new(
        origin,
        vec![part("stdout", stdout), part("stderr", stderr)],
        clock,
    )?)
}
