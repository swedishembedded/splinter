// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible, auditable workspaces for
// agents that change code, for its clients. If your team needs expertise in
// sandboxed code-editing agents or reproducible agent runs, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The environments tasks are solved and checked in, resolved from the
//! configuration alone.
//!
//! * A **runtime** (`runtime:<name>`) runs code in the process sandbox under
//!   the state root. Its environment and limits are fixed - no variable of
//!   this process reaches a call - so a runtime resolves to the same
//!   environment record in every command, and a task generated in one
//!   command is solved and checked in exactly its environment in the next.
//! * A **workspace** (`workspace`) is a source tree commands run in. Its
//!   commands run in a container of a pinned image when the configuration
//!   names a docker client and an image
//!   ([`WorkspaceSettings`]); otherwise directly on the host, bounded but not
//!   isolated, and the record says so. The host tools that lay a tree out
//!   and read a change back (`git`, `tar`) are resolved once per runtime.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde::Deserialize;
use splinter_core::digest::Digest;
use splinter_core::experience::Environment;
use splinter_sandbox::host::DEFAULT_SEARCH_PATH;
use splinter_sandbox::{
    ContainerSandbox, ImageRef, Limits, ProcessSandbox, ResolvedEnvironment, RuntimeEnvironment,
    RuntimeRegistry, Sandbox, SandboxError, TreeSnapshot, WorkspaceEnvironment, WorkspaceHost,
};
use splinter_store::StateRoot;

use crate::config::WorkspaceSettings;
use crate::error::OrchestratorError;

/// The longest one command of a workspace may run, in milliseconds: a
/// project's whole test suite, or a build, is one command.
pub const WORKSPACE_WALL_TIME_MS: u64 = 600_000;

/// The CPU time one command of a workspace may use, in seconds.
pub const WORKSPACE_CPU_SECONDS: u64 = 580;

/// The memory one command of a workspace may use, in bytes: address space
/// in the process sandbox, which a compiler maps generously.
pub const WORKSPACE_MEMORY_BYTES: u64 = 32 * 1024 * 1024 * 1024;

/// The most memory the processes of one command of a workspace may hold
/// resident together, in bytes, in the process sandbox. The address space
/// limit binds each process alone; a test command that starts many
/// interpreters (a mutant that leaks in each) is bounded only by this, and a
/// command that passes it is killed and reported as stopped by a limit. It
/// is part of the environment record of the tasks made from now on.
pub const WORKSPACE_TREE_MEMORY_BYTES: u64 = 24 * 1024 * 1024 * 1024;

/// The largest file one command of a workspace may write, in bytes.
pub const WORKSPACE_FILE_SIZE_BYTES: u64 = 1024 * 1024 * 1024;

/// What a workspace's commands run under: [`Limits::default`] raised to the
/// size of a project's own test suite and its report, with each output
/// stream kept up to `output_cap_bytes`.
///
/// `memory_bytes` bounds the address space of each process of a command;
/// `None` keeps [`WORKSPACE_MEMORY_BYTES`]. The bound is part of the
/// environment record, so only a run that names one changes the record of
/// the tasks it makes.
#[must_use]
pub fn workspace_limits(output_cap_bytes: u64, memory_bytes: Option<u64>) -> Limits {
    Limits {
        wall_time_ms: WORKSPACE_WALL_TIME_MS,
        cpu_seconds: WORKSPACE_CPU_SECONDS,
        memory_bytes: memory_bytes.unwrap_or(WORKSPACE_MEMORY_BYTES),
        file_size_bytes: WORKSPACE_FILE_SIZE_BYTES,
        output_cap_bytes,
        ..Limits::default()
    }
}

/// [`workspace_limits`] for the process sandbox, which also bounds the memory
/// of the whole command: by `tree_memory` when one is named (`Some(0)` names
/// none, which is the environment identity of every task made before the
/// bound existed), else by `memory_bytes` when one is named (so one figure
/// bounds each process and their sum), else by [`WORKSPACE_TREE_MEMORY_BYTES`].
#[must_use]
pub fn process_workspace_limits(
    output_cap_bytes: u64,
    memory_bytes: Option<u64>,
    tree_memory: Option<u64>,
) -> Limits {
    let tree_memory_bytes = match tree_memory {
        Some(0) => None,
        Some(bytes) => Some(bytes),
        None => Some(memory_bytes.unwrap_or(WORKSPACE_TREE_MEMORY_BYTES)),
    };
    Limits {
        tree_memory_bytes,
        ..workspace_limits(output_cap_bytes, memory_bytes)
    }
}

/// What a workspace is made of, short of its tree: the sandbox its commands
/// run in, the host tools, and where scratch trees go. Cheap to clone:
/// clones share the sandbox and the tools.
#[derive(Clone, Debug)]
pub struct WorkspaceBackend {
    /// Where commands run.
    pub sandbox: Arc<dyn Sandbox>,
    /// The tools that lay a tree out and read its changes back.
    pub host: Arc<WorkspaceHost>,
    /// The directory scratch trees are made under.
    pub scratch: PathBuf,
}

impl WorkspaceBackend {
    /// The environment of `snapshot` in this backend.
    #[must_use]
    pub fn environment(&self, snapshot: TreeSnapshot) -> WorkspaceEnvironment {
        WorkspaceEnvironment::new(snapshot, self.sandbox.clone(), self.host.clone())
    }
}

/// The search path of the commands a sandbox runs: `dirs` first, then the
/// system's.
fn search_path(dirs: &[std::path::PathBuf]) -> String {
    dirs.iter()
        .map(|dir| dir.display().to_string())
        .chain(std::iter::once(DEFAULT_SEARCH_PATH.to_string()))
        .collect::<Vec<_>>()
        .join(":")
}

/// The runtimes code runs in and the workspaces commands run in, each
/// resolved once.
pub struct Environments {
    sandbox: Arc<dyn Sandbox>,
    registry: RuntimeRegistry,
    resolved: Mutex<BTreeMap<String, RuntimeEnvironment>>,
    workspace_settings: WorkspaceSettings,
    workspace_scratch: PathBuf,
    workspace_calls: PathBuf,
    workspace: Mutex<Option<WorkspaceBackend>>,
}

impl Environments {
    /// The process sandbox under `root`, with the built-in runtimes, and the
    /// workspace backend `workspace` describes (resolved when first used).
    #[must_use]
    pub fn new(root: &StateRoot, workspace: &WorkspaceSettings) -> Self {
        // The interpreters code runs in are looked for in the directories
        // named for them first.
        let sandbox = ProcessSandbox::new(
            root.sandbox(),
            [("PATH".to_string(), search_path(&workspace.runtime_dirs))],
            &["PATH"],
            Limits::default(),
        );
        let scratch = workspace
            .scratch_root
            .clone()
            .unwrap_or_else(|| root.sandbox().join("workspace"));
        Self {
            sandbox: Arc::new(sandbox),
            registry: RuntimeRegistry::builtin(),
            resolved: Mutex::new(BTreeMap::new()),
            workspace_settings: workspace.clone(),
            workspace_calls: scratch.join("calls"),
            workspace_scratch: scratch,
            workspace: Mutex::new(None),
        }
    }

    /// The directories the configuration pins the task runtimes' executables
    /// to (an interpreter a task's tests start), searched first; the
    /// directories a subprocess that runs a task's code needs on its `PATH`.
    #[must_use]
    pub fn runtime_dirs(&self) -> &[PathBuf] {
        &self.workspace_settings.runtime_dirs
    }

    /// The runtime `name`, resolved in the sandbox.
    pub fn runtime(&self, name: &str) -> Result<RuntimeEnvironment, SandboxError> {
        let mut resolved = self
            .resolved
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(env) = resolved.get(name) {
            return Ok(env.clone());
        }
        let env = RuntimeEnvironment::new(&self.registry, name, Arc::clone(&self.sandbox))?;
        resolved.insert(name.to_string(), env.clone());
        Ok(env)
    }

    /// Those of the runtimes `names` that resolve here.
    pub fn available<'a>(
        &self,
        names: impl IntoIterator<Item = &'a str>,
    ) -> Vec<RuntimeEnvironment> {
        names
            .into_iter()
            .filter_map(|name| self.runtime(name).ok())
            .collect()
    }

    /// The workspace backend: a container sandbox when the configuration
    /// names a docker client and an image, else the process sandbox, and the
    /// host tools. Resolved once per runtime; a configuration that cannot be
    /// resolved (an image not pinned by digest, no `git` or `tar` to be
    /// found) is refused by name each time it is asked for.
    pub fn workspace(&self) -> Result<WorkspaceBackend, OrchestratorError> {
        let mut held = self
            .workspace
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(backend) = held.as_ref() {
            return Ok(backend.clone());
        }
        std::fs::create_dir_all(&self.workspace_scratch)
            .map_err(crate::error::io(&self.workspace_scratch))?;
        let host = WorkspaceHost::resolve(DEFAULT_SEARCH_PATH, &self.workspace_scratch)?;
        let backend = WorkspaceBackend {
            sandbox: self.workspace_sandbox()?,
            host: Arc::new(host),
            scratch: self.workspace_scratch.clone(),
        };
        *held = Some(backend.clone());
        Ok(backend)
    }

    fn workspace_sandbox(&self) -> Result<Arc<dyn Sandbox>, OrchestratorError> {
        let settings = &self.workspace_settings;
        if let (Some(docker), Some(image)) = (&settings.docker, &settings.image) {
            let image = ImageRef::parse(image).map_err(|e| {
                OrchestratorError::Refused(format!(
                    "the workspace image {image:?} is not usable: {e}"
                ))
            })?;
            return Ok(Arc::new(ContainerSandbox::new(
                docker.clone(),
                settings.docker_env.clone(),
                image,
                &self.workspace_calls,
                workspace_limits(settings.output_cap_bytes, settings.memory_bytes),
            )));
        }
        // Cargo builds offline from the cargo home it is given and leaves no
        // incremental cache in a tree that is thrown away.
        let mut cargo: Vec<(String, String)> = settings
            .cargo_home
            .iter()
            .flat_map(|home| {
                [
                    ("CARGO_HOME".to_string(), home.display().to_string()),
                    ("CARGO_INCREMENTAL".to_string(), "0".to_string()),
                ]
            })
            .collect();
        // A pinned interpreter is found first by the project's commands too:
        // a Lua family's tests start `lua`.
        if !settings.runtime_dirs.is_empty() {
            cargo.push(("PATH".to_string(), search_path(&settings.runtime_dirs)));
        }
        // What the configuration sets for every project command (it is part of
        // the environment's record).
        cargo.extend(settings.env.iter().map(|(k, v)| (k.clone(), v.clone())));
        let allowed: Vec<&str> = ["CARGO_HOME", "CARGO_INCREMENTAL", "PATH"]
            .into_iter()
            .chain(settings.env.keys().map(String::as_str))
            .collect();
        Ok(Arc::new(
            ProcessSandbox::new(
                &self.workspace_calls,
                cargo,
                &allowed,
                process_workspace_limits(
                    settings.output_cap_bytes,
                    settings.memory_bytes,
                    settings.tree_memory_bytes,
                ),
            )
            .allow_unconfined(settings.allow_unconfined)
            .allow_network(settings.allow_network)
            .max_processes(settings.max_processes),
        ))
    }

    /// The environment `recorded` names, resolved here. Whether it is
    /// still the same environment is the solve's check, against the
    /// snapshot.
    pub fn for_record(
        &self,
        recorded: &Environment,
    ) -> Result<ResolvedEnvironment, OrchestratorError> {
        if recorded.kind == Environment::CLOSED_BOOK {
            return Ok(ResolvedEnvironment::ClosedBook);
        }
        if recorded.kind == Environment::WORKSPACE {
            return self.workspace_of(recorded);
        }
        let name = recorded
            .kind
            .strip_prefix(splinter_sandbox::environment::RUNTIME_KIND_PREFIX)
            .ok_or_else(|| {
                OrchestratorError::Refused(format!(
                    "environment {:?} is neither closed-book, a runtime nor a workspace",
                    recorded.kind
                ))
            })?;
        Ok(ResolvedEnvironment::Runtime(self.runtime(name)?))
    }

    /// The workspace of the tree a record names, in this configuration's
    /// backend.
    fn workspace_of(
        &self,
        recorded: &Environment,
    ) -> Result<ResolvedEnvironment, OrchestratorError> {
        /// The part of a workspace record that names the tree.
        #[derive(Deserialize)]
        struct Tree {
            snapshot: Digest,
            tree_id: String,
        }
        let tree: Tree = serde_json::from_value(recorded.spec.clone()).map_err(|e| {
            OrchestratorError::Refused(format!("a workspace record does not name its tree: {e}"))
        })?;
        let snapshot = TreeSnapshot {
            digest: tree.snapshot,
            tree_id: tree.tree_id,
        };
        Ok(ResolvedEnvironment::Workspace(
            self.workspace()?.environment(snapshot),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_named_memory_bound_changes_the_limits_and_none_keeps_the_recorded_default() {
        let default = workspace_limits(1 << 20, None);
        assert_eq!(default.memory_bytes, WORKSPACE_MEMORY_BYTES);
        let bounded = workspace_limits(1 << 20, Some(8 << 30));
        assert_eq!(bounded.memory_bytes, 8 << 30);
        assert_eq!(
            Limits {
                memory_bytes: WORKSPACE_MEMORY_BYTES,
                ..bounded
            },
            default
        );
    }

    #[test]
    fn the_process_sandbox_also_bounds_the_whole_command() {
        let default = process_workspace_limits(1 << 20, None, None);
        assert_eq!(default.tree_memory_bytes, Some(WORKSPACE_TREE_MEMORY_BYTES));
        assert_eq!(default.memory_bytes, WORKSPACE_MEMORY_BYTES);
        let named = process_workspace_limits(1 << 20, Some(8 << 30), None);
        assert_eq!(named.tree_memory_bytes, Some(8 << 30));
        assert_eq!(named.memory_bytes, 8 << 30);
        // A container has its own memory limit and records no such bound.
        assert_eq!(workspace_limits(1 << 20, None).tree_memory_bytes, None);
    }

    /// Tasks stored before the whole-command bound name an environment without
    /// it; a run that names no bound (`Some(0)`) offers exactly that one, so
    /// those tasks can still be solved, and a figure names its own bound.
    #[test]
    fn no_whole_command_bound_offers_the_environment_of_tasks_made_before_it() {
        let before = process_workspace_limits(1 << 20, None, Some(0));
        assert_eq!(before, workspace_limits(1 << 20, None));
        let named = process_workspace_limits(1 << 20, None, Some(4 << 30));
        assert_eq!(named.tree_memory_bytes, Some(4 << 30));
        assert_eq!(named.memory_bytes, WORKSPACE_MEMORY_BYTES);
    }
}
