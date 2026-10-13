// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bounded execution of model-written code for
// its clients. If your team needs expertise in agent sandboxing or process
// supervision, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The process sandbox: the runtime runs directly on the host, bounded but
//! **not isolated**.
//!
//! Each call gets a fresh directory under the scratch root (its working
//! directory, `HOME` and `TMPDIR`), removed when the call ends; a command
//! in a directory tree runs there instead, with only `HOME` and `TMPDIR`
//! in such a fresh directory; exactly
//! the environment the sandbox was built with; its standard input from the
//! call; a wall-clock timeout that kills its whole process group; output
//! caps; and kernel resource limits (CPU time, address space, file size)
//! set in the child before the runtime starts. The number of processes a
//! call starts is not bounded: the kernel's per-user limit cannot bound
//! one call, so the sandbox records `max_processes: None`.
//!
//! None of that confines the code. It runs as the calling user and can
//! read and write whatever that user can, reach the network, and leave its
//! process group. Use it where the code is trusted as much as the user
//! running Splinter; the environment it produces records
//! `isolation: "none"`, so an experience produced in it is never mistaken
//! for a sandboxed one.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::backend::{
    io, Backend, CallDir, CodeCall, CodeResult, CommandCall, Isolation, Sandbox, SandboxError,
    SandboxSpec,
};
use crate::limits::Limits;
use crate::process::{environment_from, run, ProcessSpec};
use crate::runtime::{resolve_on_host, ResolvedRuntime, RuntimeSpec};

/// The `PATH` of a call unless the caller passes its own through the
/// allowlist; also where runtimes are looked up.
pub const DEFAULT_SEARCH_PATH: &str = "/usr/local/bin:/usr/bin:/bin";

/// The `LANG` of a call unless the caller passes its own.
pub const DEFAULT_LANG: &str = "C.UTF-8";

/// The `TERM` of a call unless the caller passes its own: no colour, no
/// cursor control, so output is plain text.
pub const DEFAULT_TERM: &str = "dumb";

/// Variables a call always gets set to its own directory.
const PER_CALL_VARIABLES: [&str; 2] = ["HOME", "TMPDIR"];

/// Runs a runtime directly on the host, in a fresh directory per call,
/// under [`Limits`]. **Not isolation**: see the module documentation.
#[derive(Clone, Debug)]
pub struct ProcessSandbox {
    scratch_root: PathBuf,
    env: BTreeMap<String, String>,
    limits: Limits,
}

impl ProcessSandbox {
    /// A sandbox whose calls get their directories under `scratch_root`
    /// and run under `limits`, with an environment of `PATH`, `LANG` and
    /// `TERM` ([`DEFAULT_SEARCH_PATH`], [`DEFAULT_LANG`], [`DEFAULT_TERM`])
    /// plus the variables of `caller_env` (the caller's environment, as the
    /// caller reads it) named in `allowlist`, which override those
    /// defaults. `HOME` and `TMPDIR` are always the call's own directory.
    #[must_use]
    pub fn new(
        scratch_root: impl Into<PathBuf>,
        caller_env: impl IntoIterator<Item = (String, String)>,
        allowlist: &[&str],
        limits: Limits,
    ) -> Self {
        let mut env: BTreeMap<String, String> = [
            ("PATH", DEFAULT_SEARCH_PATH),
            ("LANG", DEFAULT_LANG),
            ("TERM", DEFAULT_TERM),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        env.extend(environment_from(caller_env, allowlist));
        for name in PER_CALL_VARIABLES {
            env.remove(name);
        }
        Self {
            scratch_root: scratch_root.into(),
            env,
            limits: Limits {
                max_processes: None,
                ..limits
            },
        }
    }

    /// The limits every call runs under.
    #[must_use]
    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    fn search_path(&self) -> &str {
        self.env
            .get("PATH")
            .map_or(DEFAULT_SEARCH_PATH, String::as_str)
    }

    /// The bounded run of `argv` in `cwd`, with `HOME` and `TMPDIR` set to
    /// `home`.
    fn process_spec(
        &self,
        argv: Vec<String>,
        cwd: &Path,
        home: &str,
        stdin: Option<&String>,
    ) -> ProcessSpec {
        let mut env = self.env.clone();
        for name in PER_CALL_VARIABLES {
            env.insert(name.to_string(), home.to_string());
        }
        ProcessSpec {
            argv,
            cwd: cwd.to_path_buf(),
            env,
            stdin: stdin.map(|s| s.as_bytes().to_vec()),
            timeout: self.limits.wall_time(),
            output_cap: self.limits.output_cap(),
            resources: Some(self.limits.resources()),
        }
    }

    fn execute(&self, spec: &ProcessSpec) -> Result<CodeResult, SandboxError> {
        let output = run(spec)?;
        // The kernel sends SIGXCPU when the CPU-time limit is spent.
        let timed_out = output.timed_out || output.signal == Some(libc::SIGXCPU);
        Ok(CodeResult::from_output(&output, timed_out))
    }

    fn run_in(
        &self,
        dir: &CallDir,
        runtime: &ResolvedRuntime,
        call: &CodeCall,
    ) -> Result<CodeResult, SandboxError> {
        let file = dir.write_code(runtime, &call.code)?;
        let (Some(file_arg), Some(dir_arg)) = (file.to_str(), dir.path().to_str()) else {
            return Err(io(&file)(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "the scratch root is not a UTF-8 path",
            )));
        };
        let argv = runtime.spec.argv_for(&runtime.executable, file_arg);
        self.execute(&self.process_spec(argv, dir.path(), dir_arg, call.stdin.as_ref()))
    }
}

impl Sandbox for ProcessSandbox {
    fn isolation(&self) -> Isolation {
        Isolation::None
    }

    fn spec(&self) -> SandboxSpec {
        SandboxSpec {
            backend: Backend::Process,
            isolation: Isolation::None,
            limits: self.limits,
            env: self.env.clone(),
        }
    }

    fn resolve(&self, runtime: &RuntimeSpec) -> Result<ResolvedRuntime, SandboxError> {
        resolve_on_host(runtime, self.search_path(), &self.env)
    }

    fn run(&self, runtime: &ResolvedRuntime, call: &CodeCall) -> Result<CodeResult, SandboxError> {
        let dir = CallDir::create(&self.scratch_root)?;
        let result = self.run_in(&dir, runtime, call);
        let removed = dir.remove();
        let result = result?;
        removed?;
        Ok(result)
    }

    /// Runs the command with `dir` as its working directory. `HOME` and
    /// `TMPDIR` name a fresh directory of the call under the scratch root,
    /// removed afterwards, so what the command keeps in `dir` is only what
    /// it means to leave there.
    fn run_command(&self, dir: &Path, call: &CommandCall) -> Result<CodeResult, SandboxError> {
        let home = CallDir::create(&self.scratch_root)?;
        let result = match home.path().to_str() {
            Some(home_arg) => self.execute(&self.process_spec(
                call.argv.clone(),
                dir,
                home_arg,
                call.stdin.as_ref(),
            )),
            None => Err(io(home.path())(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "the scratch root is not a UTF-8 path",
            ))),
        };
        let removed = home.remove();
        let result = result?;
        removed?;
        Ok(result)
    }
}
