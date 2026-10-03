// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible execution environments for
// agent learning, for its clients. If your team needs expertise in
// replayable agent runs, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Environments resolved to exactly what a solver works in, and the record
//! an experience keeps of them.
//!
//! [`ResolvedEnvironment::record`] is the store's
//! [`Environment`](splinter_record::experience::Environment): its kind, a
//! spec holding everything that determines its behaviour, and the snapshot
//! digest of the two. For a runtime environment the spec is the runtime
//! (its registry entry, probed version and executable digest) and the
//! sandbox (backend, isolation, image digest, limits, environment
//! variables); for closed-book it is empty. Resolving the same runtime in
//! the same sandbox again yields the same record, which is how a task names
//! the environment it is to be solved in.

use std::sync::Arc;

use serde::Serialize;
use splinter_record::digest::Digest;
use splinter_record::experience::Environment;

use crate::backend::{CodeCall, CodeResult, Sandbox, SandboxError, SandboxSpec};
use crate::runtime::{ResolvedRuntime, RuntimeRegistry, RuntimeSpec};

/// The prefix of a runtime environment's kind: `runtime:<name>`.
pub const RUNTIME_KIND_PREFIX: &str = "runtime:";

/// A runtime resolved in a sandbox: where a solver's code calls run. Cheap
/// to clone: clones share the resolution and the sandbox.
#[derive(Clone, Debug)]
pub struct RuntimeEnvironment {
    runtime: Arc<ResolvedRuntime>,
    sandbox: Arc<dyn Sandbox>,
}

impl RuntimeEnvironment {
    /// The runtime `name` of `registry`, resolved in `sandbox`.
    pub fn new(
        registry: &RuntimeRegistry,
        name: &str,
        sandbox: Arc<dyn Sandbox>,
    ) -> Result<Self, SandboxError> {
        let runtime = Arc::new(sandbox.resolve(registry.get(name)?)?);
        Ok(Self { runtime, sandbox })
    }

    /// The runtime, as resolved.
    #[must_use]
    pub fn runtime(&self) -> &ResolvedRuntime {
        &self.runtime
    }

    /// The sandbox calls run in.
    #[must_use]
    pub fn sandbox(&self) -> &dyn Sandbox {
        self.sandbox.as_ref()
    }

    /// Runs one code call. Blocks until it ends or a limit stops it.
    pub fn run(&self, call: &CodeCall) -> Result<CodeResult, SandboxError> {
        self.sandbox.run(&self.runtime, call)
    }
}

/// Where a solver works, resolved to exactly what it will run.
#[derive(Clone, Debug)]
pub enum ResolvedEnvironment {
    /// No tools at all.
    ClosedBook,
    /// One tool that runs code in a runtime.
    Runtime(RuntimeEnvironment),
}

/// A runtime as its snapshot records it: the registry entry, and what it
/// resolved to that determines behaviour. Where the executable was found
/// is left out; it is identified by its content.
#[derive(Serialize)]
struct PinnedRuntime<'a> {
    #[serde(flatten)]
    spec: &'a RuntimeSpec,
    version: &'a Option<String>,
    executable_digest: &'a Option<Digest>,
}

/// The spec of a runtime environment.
#[derive(Serialize)]
struct RuntimeEnvironmentSpec<'a> {
    runtime: PinnedRuntime<'a>,
    sandbox: SandboxSpec,
}

impl ResolvedEnvironment {
    /// The environment's kind: `closed-book` or `runtime:<name>`.
    #[must_use]
    pub fn kind(&self) -> String {
        match self {
            Self::ClosedBook => Environment::CLOSED_BOOK.to_string(),
            Self::Runtime(env) => format!("{RUNTIME_KIND_PREFIX}{}", env.runtime.spec.name),
        }
    }

    /// The record an experience keeps of this environment, its snapshot
    /// computed.
    pub fn record(&self) -> Result<Environment, SandboxError> {
        match self {
            Self::ClosedBook => Ok(Environment::closed_book()),
            Self::Runtime(env) => {
                let spec = serde_json::to_value(RuntimeEnvironmentSpec {
                    runtime: PinnedRuntime {
                        spec: &env.runtime.spec,
                        version: &env.runtime.version,
                        executable_digest: &env.runtime.executable_digest,
                    },
                    sandbox: env.sandbox.spec(),
                })?;
                Ok(Environment::new(self.kind(), spec))
            }
        }
    }
}
