// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bounded execution of model-written code for
// its clients. If your team needs expertise in agent sandboxing, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The sandbox seam: where one code call runs, and what it reports.
//!
//! A [`Sandbox`] resolves a runtime to the exact program it will run and
//! runs code calls with it. [`crate::ProcessSandbox`] runs the runtime
//! directly on the host and is not isolation; [`crate::ContainerSandbox`]
//! runs it in a container with no network and a read-only root. What a
//! backend is - its isolation, limits and environment - is its
//! [`SandboxSpec`], part of every environment snapshot it produces.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::container::ImageRef;
use crate::limits::Limits;
use crate::process::{ProcessError, ProcessOutput};
use crate::runtime::{ResolvedRuntime, RuntimeError, RuntimeSpec};

/// The name of the file a call's code is written to, before the runtime's
/// extension.
pub const CODE_FILE_STEM: &str = "main";

/// How far a sandbox confines what runs in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Isolation {
    /// None: the code runs as the calling user, with its filesystem and
    /// network. Bounded by limits, not confined.
    None,
    /// A container: no network, a read-only root, its own process and
    /// memory limits.
    Container,
}

/// One code call, as the solver's tool receives it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeCall {
    /// The program text.
    pub code: String,
    /// What the program reads on standard input; absent closes it.
    pub stdin: Option<String>,
}

/// How a code call ended. The output streams are decoded as UTF-8, an
/// invalid sequence replaced by U+FFFD.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeResult {
    /// Standard output, up to the cap.
    pub stdout: String,
    /// Standard error, up to the cap.
    pub stderr: String,
    /// The exit code, when the program exited rather than being killed.
    pub exit_code: Option<i32>,
    /// The signal that ended the program, when one did.
    pub signal: Option<i32>,
    /// Whether the call was stopped by its wall-clock or CPU-time limit.
    pub timed_out: bool,
    /// Whether standard output was cut at the cap.
    pub stdout_truncated: bool,
    /// Whether standard error was cut at the cap.
    pub stderr_truncated: bool,
}

impl CodeResult {
    /// The result of a run that ended as `output`, where `timed_out` says
    /// whether a limit, not the program, ended it.
    pub(crate) fn from_output(output: &ProcessOutput, timed_out: bool) -> Self {
        Self {
            stdout: String::from_utf8_lossy(&output.stdout.bytes).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr.bytes).into_owned(),
            exit_code: output.exit_code,
            signal: output.signal,
            timed_out,
            stdout_truncated: output.stdout.truncated,
            stderr_truncated: output.stderr.truncated,
        }
    }
}

/// Which backend a sandbox is, with what only that backend has.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "backend", rename_all = "kebab-case")]
pub enum Backend {
    /// The runtime runs directly on the host.
    Process,
    /// The runtime runs in a container of a pinned image.
    Container {
        /// The image, pinned by digest.
        image: ImageRef,
        /// CPU share of the container, in thousandths of a CPU.
        cpu_millis: u32,
    },
}

/// Everything about a sandbox that determines how a call in it behaves.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SandboxSpec {
    /// The backend and its own settings.
    #[serde(flatten)]
    pub backend: Backend,
    /// How far it confines a call.
    pub isolation: Isolation,
    /// The limits every call runs under.
    pub limits: Limits,
    /// The environment variables of every call, except `HOME` and
    /// `TMPDIR`, which name the call's own working directory.
    pub env: BTreeMap<String, String>,
}

/// Where code calls run.
pub trait Sandbox: Send + Sync + std::fmt::Debug {
    /// How far this sandbox confines a call.
    fn isolation(&self) -> Isolation;

    /// Everything about this sandbox that determines how a call behaves.
    fn spec(&self) -> SandboxSpec;

    /// Resolves `runtime` to the exact program this sandbox would run for
    /// it, with its version and, where the sandbox can see it, the digest
    /// of its executable.
    fn resolve(&self, runtime: &RuntimeSpec) -> Result<ResolvedRuntime, SandboxError>;

    /// Runs one call of `runtime`. A program that fails or is stopped by a
    /// limit is a [`CodeResult`]; an error means the call could not be
    /// carried out.
    fn run(&self, runtime: &ResolvedRuntime, call: &CodeCall) -> Result<CodeResult, SandboxError>;
}

/// Why a sandbox could not resolve a runtime or carry out a call.
#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    /// The runtime could not be resolved.
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    /// The program could not be run.
    #[error(transparent)]
    Process(#[from] ProcessError),
    /// A file operation on a call's directory failed.
    #[error("{path}: {source}")]
    Io {
        /// The file or directory.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
    /// The container engine refused or failed the call.
    #[error("the container engine failed ({what}): {detail}")]
    Container {
        /// What was being done.
        what: &'static str,
        /// What the engine reported.
        detail: String,
    },
    /// The environment record cannot be serialized.
    #[error("cannot serialize the environment: {0}")]
    Serialize(#[from] serde_json::Error),
}

/// `|source| SandboxError::Io { path, source }` for `path`.
pub(crate) fn io(path: &Path) -> impl FnOnce(std::io::Error) -> SandboxError + '_ {
    move |source| SandboxError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// A call's own directory under a scratch root, holding its code file;
/// created fresh for the call and removed with [`CallDir::remove`].
pub(crate) struct CallDir {
    path: PathBuf,
}

impl CallDir {
    /// A new, empty directory under `root`, readable only by this user.
    pub(crate) fn create(root: &Path) -> Result<Self, SandboxError> {
        let dir = tempfile::Builder::new()
            .prefix("splinter-call-")
            .tempdir_in(root)
            .map_err(io(root))?;
        Ok(Self { path: dir.keep() })
    }

    /// The directory.
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Writes `code` to the call's code file, named for `runtime`'s
    /// extension, and returns its path.
    pub(crate) fn write_code(
        &self,
        runtime: &ResolvedRuntime,
        code: &str,
    ) -> Result<PathBuf, SandboxError> {
        let file = self
            .path
            .join(format!("{CODE_FILE_STEM}.{}", runtime.spec.extension));
        std::fs::write(&file, code).map_err(io(&file))?;
        Ok(file)
    }

    /// Removes the directory and everything the call left in it. A call
    /// may have taken away its own permissions on what it created, so a
    /// failed removal restores owner access throughout and tries again.
    pub(crate) fn remove(self) -> Result<(), SandboxError> {
        if std::fs::remove_dir_all(&self.path).is_ok() {
            return Ok(());
        }
        restore_access(&self.path);
        std::fs::remove_dir_all(&self.path).map_err(io(&self.path))
    }
}

/// Gives the owner full access to `path` and every directory below it, so
/// their entries can be removed. Best effort: what cannot be changed is
/// reported by the removal that follows.
fn restore_access(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let Ok(meta) = std::fs::symlink_metadata(path) else {
        return;
    };
    if !meta.is_dir() {
        return;
    }
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700));
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            restore_access(&entry.path());
        }
    }
}
