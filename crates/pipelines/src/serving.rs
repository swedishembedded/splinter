// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that hand their model
// servers to other tools while they run, on one machine or across a cluster
// allocation, for its clients. If your team needs expertise in sharing model
// serving between training campaigns and evaluation harnesses, you can
// procure our services by sending an email to info@swedishembedded.com.

//! `serve`: the brain servers the configuration names, started and kept
//! for other programs (a benchmark, a campaign loop) until interrupted, and
//! where those programs find them.

use std::path::PathBuf;
use std::time::Duration;

use serde::Serialize;
use splinter_model::endpoint::TrainSpan;
use splinter_model::service::{
    ADAPTERS_DIR, ARTIFACTS_DIR, AUTHORITY_FILE, ENDPOINTS_FILE, KEY_FILE, TRAIN_JOB_SECRET_FILE,
};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::OrchestratorError;

/// Where the servers are, for a client outside this process.
#[derive(Clone, Debug, Serialize)]
pub struct Serving {
    /// Their OpenAI base URLs, one per server.
    pub endpoints: Vec<String>,
    /// The file listing them, one URL per line.
    pub endpoints_file: PathBuf,
    /// The file holding the key they answer to (owner-only).
    pub key_file: PathBuf,
    /// The certificate authority their TLS certificates chain to, for a
    /// fleet serving TLS; `None` for plaintext servers.
    pub authority_file: Option<PathBuf>,
    /// The directory they serve adapters from: an adapter written there is
    /// a model of its own on every server (`--brain-adapter-dir`).
    pub adapter_dir: PathBuf,
    /// The artifact store training on them moves data through
    /// (`--brain-artifact-store`).
    pub artifact_store: PathBuf,
    /// The cards a data-parallel training run takes on each server, in the
    /// order of `endpoints` (`--brain-train-cards`); empty for one card.
    pub train_cards: Vec<u32>,
    /// The job secret the servers were started with, for one training job
    /// over several (`--brain-train-job-secret`).
    pub train_job_secret: Option<PathBuf>,
    /// The address the first server binds as such a job's coordinator
    /// (`--brain-train-coordinator`).
    pub train_coordinator: Option<String>,
}

/// Starts the servers `ctx`'s configuration names (one per usable node of an
/// allocation that spans nodes, else one on this machine) and says where
/// they are.
///
/// # Errors
/// They did not start, or were given (`--brain-endpoint`): this process
/// serves nothing of its own then.
pub fn start(ctx: &Context) -> Result<Serving, OrchestratorError> {
    let endpoints = ctx.brain()?;
    let directory = ctx.runtime().brain_directory().ok_or_else(|| {
        OrchestratorError::Refused(
            "serve starts brain servers of its own: --brain-endpoint names servers that are \
             already running"
                .into(),
        )
    })?;
    let authority = directory.join(AUTHORITY_FILE);
    let (train_cards, train_job_secret, train_coordinator) = match endpoints.train_span() {
        TrainSpan::OneCard => (Vec::new(), None, None),
        TrainSpan::Cards(cards) => (vec![*cards], None, None),
        TrainSpan::Servers(job) => (
            job.cards.clone(),
            Some(directory.join(TRAIN_JOB_SECRET_FILE)),
            Some(job.coordinator.clone()),
        ),
    };
    Ok(Serving {
        train_cards,
        train_job_secret,
        train_coordinator,
        endpoints: endpoints.urls().to_vec(),
        endpoints_file: directory.join(ENDPOINTS_FILE),
        key_file: directory.join(KEY_FILE),
        authority_file: authority.exists().then_some(authority),
        adapter_dir: ctx.root().brain_service().join(ADAPTERS_DIR),
        artifact_store: ctx.root().brain_service().join(ARTIFACTS_DIR),
    })
}

/// Blocks until an interrupt asks the process to stop; the servers stop
/// when the context that started them is dropped.
pub fn hold_until_interrupted() {
    while !splinter_orchestrator::interrupt::interrupted() {
        std::thread::sleep(Duration::from_millis(250));
    }
}
