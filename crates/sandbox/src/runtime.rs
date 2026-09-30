// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible execution environments for
// agent learning, for its clients. If your team needs expertise in
// replayable agent experiments, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Runtimes: the languages a solver's code can run in, named in a registry
//! and pinned by what they resolve to.
//!
//! A [`RuntimeSpec`] says how to run a code file (an argv template), what
//! extension the file gets, and how to ask the runtime its version. A
//! sandbox resolves it to a [`ResolvedRuntime`]: the program it will run,
//! the version string that program reports, and - where the sandbox can see
//! the executable - the digest of its content. Those are what an
//! environment's snapshot records, so a runtime that changes under the
//! same name is a different environment.
//!
//! The executable's digest pins that one file, not the libraries or the
//! standard library it loads; a container image digest pins them all.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use splinter_store::digest::Digest;

use crate::backend::SandboxError;
use crate::process::{run, ProcessError, ProcessOutput, ProcessSpec};

/// The placeholder in an argv template for the runtime's executable.
pub const EXE: &str = "{exe}";

/// The placeholder in an argv template for the code file.
pub const FILE: &str = "{file}";

/// How long a version probe may take.
pub const PROBE_TIMEOUT: Duration = Duration::from_secs(10);

/// The most bytes of a version probe's output kept.
const PROBE_OUTPUT_CAP: usize = 4096;

/// How to run code in one language.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeSpec {
    /// The runtime's name in the registry: `python3`, `lua`, ...
    pub name: String,
    /// The program, looked up on the sandbox's search path when it has no
    /// `/`.
    pub program: String,
    /// The command that runs a code file, with [`EXE`] and [`FILE`]
    /// placeholders.
    pub argv: Vec<String>,
    /// The code file's extension, without the dot.
    pub extension: String,
    /// The command that prints the runtime's version, with an [`EXE`]
    /// placeholder; `None` for a runtime with no way to ask (`sh`), whose
    /// identity is then its executable's digest alone.
    pub version_argv: Option<Vec<String>>,
}

impl RuntimeSpec {
    fn new(name: &str, argv: &[&str], extension: &str, version_argv: Option<&[&str]>) -> Self {
        let owned = |args: &[&str]| args.iter().map(|a| a.to_string()).collect();
        Self {
            name: name.to_string(),
            program: name.to_string(),
            argv: owned(argv),
            extension: extension.to_string(),
            version_argv: version_argv.map(owned),
        }
    }

    /// The command that runs `file` with the executable `exe`.
    #[must_use]
    pub fn argv_for(&self, exe: &str, file: &str) -> Vec<String> {
        self.argv
            .iter()
            .map(|arg| arg.replace(EXE, exe).replace(FILE, file))
            .collect()
    }

    /// The command that prints the version of the executable `exe`.
    #[must_use]
    pub fn version_argv_for(&self, exe: &str) -> Option<Vec<String>> {
        self.version_argv
            .as_ref()
            .map(|argv| argv.iter().map(|arg| arg.replace(EXE, exe)).collect())
    }
}

/// The runtimes a sandbox can be asked for, by name.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RuntimeRegistry {
    runtimes: BTreeMap<String, RuntimeSpec>,
}

impl RuntimeRegistry {
    /// `python3` (in isolated mode, `-I`: no user site directory, no
    /// `PYTHON*` variables), `lua`, `node` and `sh`.
    #[must_use]
    pub fn builtin() -> Self {
        let mut registry = Self::default();
        for spec in [
            RuntimeSpec::new(
                "python3",
                &[EXE, "-I", FILE],
                "py",
                Some(&[EXE, "--version"]),
            ),
            RuntimeSpec::new("lua", &[EXE, FILE], "lua", Some(&[EXE, "-v"])),
            RuntimeSpec::new("node", &[EXE, FILE], "js", Some(&[EXE, "--version"])),
            RuntimeSpec::new("sh", &[EXE, FILE], "sh", None),
        ] {
            registry.register(spec);
        }
        registry
    }

    /// Adds `spec`, replacing a runtime of the same name.
    pub fn register(&mut self, spec: RuntimeSpec) {
        self.runtimes.insert(spec.name.clone(), spec);
    }

    /// The runtime named `name`.
    pub fn get(&self, name: &str) -> Result<&RuntimeSpec, RuntimeError> {
        self.runtimes
            .get(name)
            .ok_or_else(|| RuntimeError::Unknown {
                name: name.to_string(),
                known: self.runtimes.keys().cloned().collect(),
            })
    }

    /// Every registered runtime's name, in order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.runtimes.keys().map(String::as_str)
    }
}

/// A runtime resolved to the exact program a sandbox runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedRuntime {
    /// The registry entry it was resolved from.
    pub spec: RuntimeSpec,
    /// What the argv's [`EXE`] becomes: the executable's absolute path on
    /// the host, or the program's name inside a container image.
    pub executable: String,
    /// What the version probe printed (its first non-empty line), when the
    /// runtime has one.
    pub version: Option<String>,
    /// The SHA-256 of the executable's content, when the sandbox runs it
    /// from the host; a container's image digest pins it instead.
    pub executable_digest: Option<Digest>,
}

/// Why a runtime could not be resolved.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    /// No runtime of that name is registered.
    #[error("no runtime named {name:?} (registered: {known:?})")]
    Unknown {
        /// The name asked for.
        name: String,
        /// The names registered.
        known: Vec<String>,
    },
    /// The runtime's program is not an executable file on the search path.
    #[error("runtime {runtime:?}: no executable {program:?} on the search path {search_path:?}")]
    ExecutableNotFound {
        /// The runtime.
        runtime: String,
        /// The program looked for.
        program: String,
        /// Where it was looked for.
        search_path: String,
    },
    /// The executable could not be read to compute its digest.
    #[error("runtime {runtime:?}: cannot read {path}: {source}")]
    Unreadable {
        /// The runtime.
        runtime: String,
        /// The executable.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
    /// The version probe could not be run.
    #[error("runtime {runtime:?}: the version probe could not run: {source}")]
    ProbeFailed {
        /// The runtime.
        runtime: String,
        /// Why.
        source: ProcessError,
    },
    /// The version probe ran and did not report a version.
    #[error("runtime {runtime:?}: the version probe reported no version: {detail}")]
    NoVersion {
        /// The runtime.
        runtime: String,
        /// How the probe ended.
        detail: String,
    },
}

/// Resolves `spec` on the host: its program found on `search_path`, the
/// executable's canonical path and content digest, and its version as the
/// probe reports it, run with environment `env`.
pub(crate) fn resolve_on_host(
    spec: &RuntimeSpec,
    search_path: &str,
    env: &BTreeMap<String, String>,
) -> Result<ResolvedRuntime, SandboxError> {
    let not_found = || RuntimeError::ExecutableNotFound {
        runtime: spec.name.clone(),
        program: spec.program.clone(),
        search_path: search_path.to_string(),
    };
    let found = find_executable(&spec.program, search_path).ok_or_else(not_found)?;
    let unreadable = |path: &Path| {
        let path = path.to_path_buf();
        |source| RuntimeError::Unreadable {
            runtime: spec.name.clone(),
            path,
            source,
        }
    };
    let executable = found.canonicalize().map_err(unreadable(&found))?;
    let file = std::fs::File::open(&executable).map_err(unreadable(&executable))?;
    let digest = Digest::of_reader(file).map_err(unreadable(&executable))?;
    let Some(executable) = executable.to_str().map(str::to_string) else {
        return Err(not_found().into());
    };
    let version = match spec.version_argv_for(&executable) {
        None => None,
        Some(argv) => {
            let probe = ProcessSpec::new(argv, "/", env.clone(), PROBE_TIMEOUT, PROBE_OUTPUT_CAP);
            let output = run(&probe).map_err(|source| RuntimeError::ProbeFailed {
                runtime: spec.name.clone(),
                source,
            })?;
            Some(version_from(&spec.name, &output)?)
        }
    };
    Ok(ResolvedRuntime {
        spec: spec.clone(),
        executable,
        version,
        executable_digest: Some(digest),
    })
}

/// The version a probe that ended as `output` reports: the first non-empty
/// line of its standard output, or of its standard error when standard
/// output is empty (older interpreters print their version there).
pub(crate) fn version_from(runtime: &str, output: &ProcessOutput) -> Result<String, RuntimeError> {
    let first_line = |bytes: &[u8]| {
        String::from_utf8_lossy(bytes)
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty())
            .map(str::to_string)
    };
    let version = first_line(&output.stdout.bytes).or_else(|| first_line(&output.stderr.bytes));
    match version {
        Some(version) if output.exit_code == Some(0) && !output.timed_out => Ok(version),
        _ => Err(RuntimeError::NoVersion {
            runtime: runtime.to_string(),
            detail: format!(
                "exit code {:?}, timed out {}, output {:?}",
                output.exit_code,
                output.timed_out,
                String::from_utf8_lossy(&output.stderr.bytes)
            ),
        }),
    }
}

/// `program` itself when it names a path, else the first executable file
/// of that name on `search_path`.
fn find_executable(program: &str, search_path: &str) -> Option<PathBuf> {
    if program.contains('/') {
        let path = PathBuf::from(program);
        return is_executable_file(&path).then_some(path);
    }
    search_path
        .split(':')
        .filter(|dir| !dir.is_empty())
        .map(|dir| Path::new(dir).join(program))
        .find(|candidate| is_executable_file(candidate))
}

fn is_executable_file(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
}
