// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fast, reproducible evaluation of fine-tuned
// language models for its clients. If your team needs expertise in putting a
// large examination to many adapters in batches, you can procure our services
// by sending an email to info@swedishembedded.com.

//! The examination put to models a `brain serve` holds: one server per model
//! the stage loads (the base, or the base with one adapter), its decode slots
//! kept full by asking that many items at once. The answers are the server's,
//! decoded in batches; every arm of a study is asked the same way.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use splinter_sdk::agent::sven::model::ModelProvider;
use splinter_sdk::agent::CancelToken;
use splinter_sdk::brain_server::{self, Failure, Launch, Offer, Started};
use splinter_sdk::model::selection::served_model;

use super::examine::{Examinee, Loader};

/// How long a server may take to load the base and bind.
const STARTUP: Duration = Duration::from_secs(1800);

/// Starts a `brain serve` for every model the examination loads.
pub struct ServedLoader {
    /// The `brain` binary.
    pub brain: PathBuf,
    /// The base checkpoint.
    pub base: PathBuf,
    /// Where each server gets a fresh directory for its keys and ready
    /// marker, so examinations running side by side never share one.
    pub work: PathBuf,
    /// The compute the servers may schedule, in brain's `--device` grammar.
    pub device: Option<String>,
    /// Sequences a server decodes together, and so items asked at once.
    pub slots: u32,
}

/// A model one running server answers for.
struct Served {
    /// Kept for the model's life; the server stops when it is dropped.
    server: Started,
    /// The server's directory, removed after the server has stopped (fields
    /// drop in declaration order).
    _work: tempfile::TempDir,
    url: String,
    greedy: Arc<dyn ModelProvider>,
    slots: usize,
}

impl Served {
    fn provider(
        &self,
        temperature: f32,
        seed: Option<u64>,
    ) -> anyhow::Result<Arc<dyn ModelProvider>> {
        let loaded = served_model(
            &self.url,
            &self.server.key,
            &self.server.model,
            Some(temperature),
            seed,
        )?;
        Ok(loaded.provider())
    }
}

impl Examinee for Served {
    fn greedy(&self) -> Arc<dyn ModelProvider> {
        Arc::clone(&self.greedy)
    }

    fn sampled(&self, temperature: f32, seed: u64) -> anyhow::Result<Arc<dyn ModelProvider>> {
        self.provider(temperature, Some(seed))
    }

    fn slots(&self) -> usize {
        self.slots
    }
}

impl Loader for ServedLoader {
    fn load(&self, adapter: Option<&Path>) -> anyhow::Result<Box<dyn Examinee>> {
        std::fs::create_dir_all(&self.work)?;
        let work = tempfile::Builder::new()
            .prefix(if adapter.is_some() {
                "adapter-"
            } else {
                "base-"
            })
            .tempdir_in(&self.work)?;
        let digest = match adapter {
            Some(file) => crate::binding::file_digest(file)?,
            None => String::new(),
        };
        // The server serves the base it is given and the adapter offered,
        // nothing else: brain scans the store it is given at startup, so it
        // gets an empty one (a store that does not exist would leave brain
        // its default, every model pulled on the machine).
        let no_store = work.path().join("no-store");
        std::fs::create_dir(&no_store)?;
        // The server's own output, kept beside its directory once it is gone.
        let log = work.path().with_extension("log");
        eprintln!("brain serve for this model logs to {}", log.display());
        let launch = Launch {
            binary: &self.brain,
            work: work.path(),
            models: &no_store,
            base: &self.base,
            offer: adapter.map_or(Offer::Nothing, Offer::Adapter),
            expected_digest: &digest,
            startup: STARTUP,
            device: self.device.as_deref(),
            max_batch: Some(self.slots),
            log: Some(&log),
        };
        let server =
            brain_server::start(&launch, &CancelToken::default()).map_err(|e| match e {
                Failure::Cancelled => anyhow::anyhow!("starting brain serve was cancelled"),
                Failure::Unmeasured(why) => anyhow::anyhow!("brain serve: {why}"),
            })?;
        anyhow::ensure!(
            adapter.is_none() || server.served_digest == digest,
            "brain serve offers {} but the adapter is {digest}",
            server.served_digest
        );
        let url = format!("http://{}/v1", server.address);
        let greedy = served_model(&url, &server.key, &server.model, Some(0.0), None)?.provider();
        Ok(Box::new(Served {
            server,
            _work: work,
            url,
            greedy,
            slots: self.slots as usize,
        }))
    }
}
