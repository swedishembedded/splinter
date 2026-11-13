// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that run on a model
// service they own the lifetime of, so that inference, training and
// evaluation share one set of cards, for its clients. If your team needs
// expertise in operating a model server as part of a training pipeline, you
// can procure our services by sending an email to info@swedishembedded.com.

//! The brain server a run is requested from.
//!
//! Splinter reaches brain only through its API. A run is either given a
//! server (`--brain-endpoint`) or starts one it owns for the length of the
//! run: a `brain serve` over the model store, with adapters served from a
//! directory and the training workload on, so inference, evaluation and
//! training jobs share its cards. [`BrainConnection`] is that choice, and
//! [`BrainConnection::endpoints`] is what every request is made against.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use sven_sdk::CancelToken;

use crate::endpoint::BrainEndpoints;
use crate::error::PolicyError;
use crate::served_train::ArtifactStore;
use crate::server::{remember, start_service, Failure, Running, ServiceLaunch};

/// How long a server Splinter starts may take to bind and report ready: it
/// scans the model store before it does.
const STARTUP: Duration = Duration::from_secs(600);

/// What it takes to start the server a run owns.
#[derive(Clone, Debug)]
pub struct ServiceSettings {
    /// The `brain` binary.
    pub binary: PathBuf,
    /// Brain's model store.
    pub models: PathBuf,
    /// Where the service keeps its state for the run: its scratch directory,
    /// the adapters it serves and the artifact store training moves data
    /// through. Created if absent.
    pub state: PathBuf,
    /// The compute it may schedule, in brain's `--device` grammar.
    pub device: Option<String>,
    /// How the hardware is driven (`--backend`).
    pub backend: Option<String>,
    /// Sequences it decodes together, which is also how many requests are
    /// asked of it at once.
    pub max_batch: Option<u32>,
}

/// A brain server this process started: stopped, and its scratch directory
/// removed, when dropped.
pub struct ManagedBrain {
    running: Running,
    endpoints: BrainEndpoints,
    work: PathBuf,
    /// The last lines the server wrote, kept for an operator to read when a
    /// request to it fails.
    output: Arc<Mutex<Vec<String>>>,
}

impl ManagedBrain {
    /// Starts a server as `settings` say and waits until it is ready.
    pub fn start(settings: &ServiceSettings, cancel: &CancelToken) -> Result<Self, PolicyError> {
        let io = |path: &Path| {
            let path = path.to_path_buf();
            move |source| PolicyError::Io { path, source }
        };
        let adapters = settings.state.join("adapters");
        let artifacts = settings.state.join("artifacts");
        let work = settings.state.join(format!("run-{}", std::process::id()));
        for dir in [&adapters, &artifacts, &work] {
            std::fs::create_dir_all(dir).map_err(io(dir))?;
        }
        let launch = ServiceLaunch {
            binary: &settings.binary,
            work: &work,
            models: &settings.models,
            adapters: &adapters,
            artifacts: &artifacts,
            device: settings.device.as_deref(),
            backend: settings.backend.as_deref(),
            max_batch: settings.max_batch,
            startup: STARTUP,
            log: None,
        };
        let started = start_service(&launch, cancel);
        let mut service = match started {
            Ok(service) => service,
            Err(failure) => {
                let _ = std::fs::remove_dir_all(&work);
                return Err(PolicyError::Endpoint {
                    url: settings.binary.display().to_string(),
                    reason: match failure {
                        Failure::Cancelled => "starting the brain server was cancelled".into(),
                        Failure::Unmeasured(why) => why,
                    },
                });
            }
        };
        let endpoints = BrainEndpoints::new(vec![service.url()], service.key.clone())?
            .with_attempt_concurrency(service.slots as usize)
            .with_adapter_directory(adapters)
            .with_artifact_store(ArtifactStore::open(&artifacts)?);
        // Everything the server writes is read as it comes, so a full pipe
        // never stops it; only the last lines are kept.
        let output = Arc::new(Mutex::new(std::mem::take(&mut service.kept)));
        let drained = Arc::clone(&output);
        let lines = service.lines;
        std::thread::spawn(move || {
            for line in lines {
                remember(
                    &mut drained
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner),
                    line,
                );
            }
        });
        Ok(Self {
            running: service.running,
            endpoints,
            work,
            output,
        })
    }

    /// The last lines the server wrote, as one line of text.
    #[must_use]
    pub fn recent_output(&self) -> String {
        self.output
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .join(" | ")
    }

    /// The endpoints that reach this server.
    #[must_use]
    pub fn endpoints(&self) -> &BrainEndpoints {
        &self.endpoints
    }
}

impl Drop for ManagedBrain {
    fn drop(&mut self) {
        // Stopped before its scratch directory goes: the keys file lives there.
        let _ = self.running.0.kill();
        let _ = self.running.0.wait();
        let _ = std::fs::remove_dir_all(&self.work);
    }
}

/// How a run reaches brain: a server it was given, or one it owns.
pub enum BrainConnection {
    /// A server somebody else runs.
    Given(BrainEndpoints),
    /// A server this process started.
    Owned(ManagedBrain),
}

impl BrainConnection {
    /// The endpoints requests are made against.
    #[must_use]
    pub fn endpoints(&self) -> &BrainEndpoints {
        match self {
            Self::Given(endpoints) => endpoints,
            Self::Owned(owned) => owned.endpoints(),
        }
    }
}
