// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bounded, replayable execution environments
// for agent learning, for its clients. If your team needs expertise in agent
// sandboxing or reproducible agent runs, you can procure our services
// by sending an email to info@swedishembedded.com.

//! Where a solver works: bounded process runs, runtimes, sandboxes, and
//! environments pinned by a snapshot digest.
//!
//! * [`process`] - one bounded run of a program: explicit environment,
//!   process-group kill at a timeout, output caps, optional kernel resource
//!   limits. Every process Splinter runs goes through it.
//! * [`landlock`] - filesystem confinement of a child process by the kernel.
//! * [`limits`] - what one code call may use, with its defaults.
//! * [`runtime`] - the registry of runtimes (`python3`, `lua`, `node`,
//!   `sh`) and what a runtime resolves to: version and executable digest.
//! * [`backend`] - the [`Sandbox`] seam, a code call and its result.
//! * [`host`] - [`ProcessSandbox`]: the runtime run directly, bounded but
//!   not isolated (`isolation: "none"`).
//! * [`container`] - [`ContainerSandbox`]: a fresh container of an image
//!   pinned by digest, no network, read-only root
//!   (`isolation: "container"`).
//! * [`workspace`] and [`workspace_host`] - [`WorkspaceEnvironment`]: a
//!   source tree pinned by the digest of its archive, laid out byte for
//!   byte, commands run in it, and what a solver changed read back from git
//!   as a patch.
//! * [`environment`] - [`ResolvedEnvironment`] (closed-book, a runtime in a
//!   sandbox, or a workspace) and the record whose snapshot pins it.
//!
//! Nothing here reads the process environment: what a child sees, and
//! where runtimes are looked up, are values the caller passes.

#![warn(missing_docs)]

#[cfg(not(unix))]
compile_error!("splinter-sandbox needs unix process groups and resource limits");

pub mod backend;
pub mod container;
pub mod environment;
pub mod host;
pub mod landlock;
pub mod limits;
pub mod process;
pub mod runtime;
pub mod seccomp;
pub mod workspace;
pub mod workspace_host;

pub use backend::{
    Backend, CodeCall, CodeResult, CommandCall, Isolation, Sandbox, SandboxError, SandboxSpec,
};
pub use container::{ContainerSandbox, ImageRef, ImageRefError};
pub use environment::{ResolvedEnvironment, RuntimeEnvironment};
pub use host::ProcessSandbox;
pub use limits::Limits;
pub use runtime::{ResolvedRuntime, RuntimeError, RuntimeRegistry, RuntimeSpec};
pub use workspace::{Baseline, Patch, TreeSnapshot, WorkspaceEnvironment};
pub use workspace_host::WorkspaceHost;
