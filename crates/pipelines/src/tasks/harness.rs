// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent-driven training-task generation that
// is judged by what the tasks do when run, for its clients. If your team
// needs expertise in synthetic task generation then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Which proposer writes the model kinds' tasks: one typed call to the
//! generator model, or a sven harness run as a subprocess.

use std::path::PathBuf;
use std::sync::Arc;

use serde::Serialize;
use splinter_agent::harness_proposer::{HarnessLaunch, HarnessProposer};
use splinter_agent::proposer::SvenProposer;
use splinter_agent::CancelToken;
use splinter_knowledge::tasks::propose::TaskProposer;
use splinter_orchestrator::context::Context;
use splinter_sandbox::RuntimeEnvironment;

/// The directory under the state root that keeps the trajectories of every
/// harness run.
const HARNESS_RUNS: &str = "task-harness";

/// A sven harness that writes the model kinds' tasks in place of a single
/// typed call, run as `sven harness run` against a model server that is
/// already running.
#[derive(Clone, Debug, Serialize)]
pub struct TaskHarness {
    /// The harness document or directory.
    pub harness: PathBuf,
    /// The `sven` executable that runs it.
    pub sven: PathBuf,
    /// The OpenAI-compatible server the harness's agents talk to.
    pub endpoint: String,
    /// The model they ask for there, as `provider/name`.
    pub model: String,
}

/// The proposer for a generation: `harness` when it names one, else `typed`.
/// The harness proposer hands every request that is not for a kind's tasks
/// (variants) to `typed`.
pub(super) fn proposer(
    ctx: &Context,
    harness: Option<&TaskHarness>,
    typed: SvenProposer,
    runtimes: &[RuntimeEnvironment],
    cancel: &CancelToken,
) -> Arc<dyn TaskProposer> {
    let Some(harness) = harness else {
        return Arc::new(typed);
    };
    let launch = HarnessLaunch {
        sven: harness.sven.clone(),
        harness: harness.harness.clone(),
        model: harness.model.clone(),
        endpoint: harness.endpoint.clone(),
        runs: ctx.root().path().join(HARNESS_RUNS),
        first_byte_timeout: None,
    };
    Arc::new(
        HarnessProposer::new(launch, runtimes.to_vec(), Arc::new(typed))
            .with_cancel(cancel.clone()),
    )
}
