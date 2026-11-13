// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that put an informative on-policy
// reinforcement-learning signal through a language model against verifiers it
// cannot influence, for its clients. If your team needs expertise in running
// a lineage of policies behind one continuously batching server you can
// procure our services by sending an email to info@swedishembedded.com.

//! One `brain serve` for the whole lineage: the base, every adapter of the
//! lineage offered from a directory it follows, and training jobs on the same
//! cards. Rollouts, the dev exam, the policy's proposals and the training of
//! each arm all go through its API, many requests at once. A new adapter is
//! hot-loaded by placing it in the directory and waiting until the server
//! lists it under its digest.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use splinter_sdk::agent::solve::Model;
use splinter_sdk::agent::CancelToken;
use splinter_sdk::brain_server::{self, Failure, Launch, Offer, Started};
use splinter_sdk::campaign::{CampaignError, PolicyRef};
use splinter_sdk::model::served_train::ArtifactStore;
use splinter_sdk::model::BrainEndpoints;
use splinter_sdk::store::StateRoot;
use splinter_sdk::vocabulary::digest::Digest;
use splinter_sdk::{Config, Context, Splinter};

/// How the server is started and sampled.
#[derive(Clone, Debug)]
pub struct HostSettings {
    /// The `brain` binary.
    pub brain: PathBuf,
    /// Brain's model store.
    pub models: PathBuf,
    /// The base checkpoint.
    pub base: PathBuf,
    /// A scratch directory for the server's keys and ready marker.
    pub work: PathBuf,
    /// Splinter's state for the handle the models are asked through.
    pub state: PathBuf,
    /// The artifact store training jobs exchange datasets and adapters
    /// through.
    pub store: PathBuf,
    /// The directory of adapters the server follows.
    pub adapters: PathBuf,
    /// Where the server's output is appended.
    pub log: PathBuf,
    /// The card, in brain's `--device` grammar.
    pub device: String,
    /// The KV-cache tier (`f32`, `bf16`, `int8`, `fp8`): sampling and training
    /// agree best with `f32`.
    pub kv_cache: String,
    /// Sequences the server decodes together, and requests kept in flight.
    pub slots: u32,
    /// The directory of the interpreter the task kinds run code in.
    pub lua_dir: PathBuf,
    /// The sampling temperature of [`ServerHost::predicting`].
    pub predict_temperature: f32,
}

/// The running server and the handles that ask it; the server is stopped
/// when this is dropped.
struct Running {
    sampled: Splinter,
    greedy: Splinter,
    predicting: Splinter,
    endpoints: BrainEndpoints,
    _server: Mutex<Started>,
}

/// How long a placed adapter is waited for before the server is declared not
/// to follow its directory.
const ADOPT_WAIT: Duration = Duration::from_secs(180);

/// The server of the lineage; see the module documentation.
pub struct ServerHost {
    settings: HostSettings,
    running: Mutex<Option<Arc<Running>>>,
}

impl ServerHost {
    /// A host that has started nothing yet.
    #[must_use]
    pub fn new(settings: HostSettings) -> Self {
        Self {
            settings,
            running: Mutex::new(None),
        }
    }

    fn handle(
        &self,
        server: &Started,
        endpoints: &BrainEndpoints,
        temperature: f32,
    ) -> Result<Splinter> {
        let s = &self.settings;
        let mut config = Config::from_env();
        config.state_root = StateRoot::new(&s.state);
        config.policy_base = std::path::absolute(&s.base)?;
        config.model_store = s.models.clone();
        config.workspace.runtime_dirs = vec![std::path::absolute(&s.lua_dir)?];
        config.thinking = false;
        config.allow_remote = false;
        config.brain = Some(endpoints.clone());
        // The training distribution itself: temperature as asked, no nucleus
        // or top-k cut.
        config.served_temperature = Some(temperature);
        config.served_top_p = Some(1.0);
        config.served_top_k = Some(0);
        let _ = server;
        Splinter::builder(config)
            .build()
            .context("starting a Splinter handle on the server")
    }

    fn launch(&self) -> Result<Running> {
        let s = &self.settings;
        for dir in [&s.work, &s.store, &s.adapters] {
            std::fs::create_dir_all(dir)?;
        }
        let launch = Launch {
            binary: &s.brain,
            work: &s.work,
            models: &s.models,
            base: &s.base,
            offer: Offer::Directory(&s.adapters),
            expected_digest: "",
            startup: Duration::from_secs(900),
            device: Some(&s.device),
            max_batch: Some(s.slots),
            kv_offload_gb: None,
            kv_cache: Some(&s.kv_cache),
            artifact_store: Some(&s.store),
            log: Some(&s.log),
        };
        let server = brain_server::start(&launch, &CancelToken::new()).map_err(|f| match f {
            Failure::Cancelled => anyhow::anyhow!("starting brain serve was cancelled"),
            Failure::Unmeasured(why) => anyhow::anyhow!("brain serve did not start: {why}"),
        })?;
        let endpoints = BrainEndpoints::new(
            vec![format!("http://{}/v1", server.address)],
            server.key.clone(),
        )?
        .with_attempt_concurrency(s.slots as usize)
        .with_artifact_store(ArtifactStore::open(&s.store)?);
        Ok(Running {
            sampled: self.handle(&server, &endpoints, 1.0)?,
            greedy: self.handle(&server, &endpoints, 0.0)?,
            predicting: self.handle(&server, &endpoints, s.predict_temperature)?,
            endpoints,
            _server: Mutex::new(server),
        })
    }

    fn running(&self) -> Result<Arc<Running>> {
        let mut running = self
            .running
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(r) = running.as_ref() {
            return Ok(Arc::clone(r));
        }
        let started = Arc::new(self.launch()?);
        *running = Some(Arc::clone(&started));
        Ok(started)
    }

    /// The service's endpoints, with its artifact store: what a training job
    /// is sent to. Starts the server when it is not running.
    ///
    /// # Errors
    /// The server does not start.
    pub fn endpoints(&self) -> Result<BrainEndpoints> {
        Ok(self.running()?.endpoints.clone())
    }

    /// Makes `policy` servable: its adapter, when it has one, is offered to
    /// the server's directory and waited for until the server lists it under
    /// the digest of the file.
    fn adopt(&self, running: &Running, policy: &PolicyRef) -> Result<()> {
        let Some(adapter) = &policy.adapter else {
            return Ok(());
        };
        let digest = Digest::sha256_of_reader(std::fs::File::open(adapter)?)?.to_string();
        let target = self.settings.adapters.join(format!(
            "{}.safetensors",
            digest.trim_start_matches("sha256:")
        ));
        if !target.exists() {
            let partial = target.with_extension("partial");
            std::fs::copy(adapter, &partial)
                .with_context(|| format!("offering {}", adapter.display()))?;
            std::fs::rename(&partial, &target)?;
        }
        let deadline = Instant::now() + ADOPT_WAIT;
        loop {
            let listed = running.endpoints.list(0)?;
            if listed
                .iter()
                .any(|card| card.adapter_digest.as_deref() == Some(digest.as_str()))
            {
                return Ok(());
            }
            anyhow::ensure!(
                Instant::now() < deadline,
                "the server did not list the adapter {digest} within {}s",
                ADOPT_WAIT.as_secs()
            );
            std::thread::sleep(Duration::from_millis(250));
        }
    }

    /// The context that samples at temperature one, with `policy` servable.
    ///
    /// # Errors
    /// The server does not start or does not list the policy's adapter.
    pub fn context(&self, policy: &PolicyRef) -> Result<Context> {
        let running = self.running()?;
        self.adopt(&running, policy)?;
        Ok(running.sampled.context())
    }

    /// Stops the server.
    pub fn stop(&self) {
        *self
            .running
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    }
}

impl ServerHost {
    /// The model that answers as `policy` does, decoding through `handle`.
    fn model_on(
        &self,
        policy: &PolicyRef,
        handle: impl FnOnce(&Running) -> &Splinter,
    ) -> Result<Model, CampaignError> {
        let fail = |e: anyhow::Error| CampaignError::Collector(format!("{e:#}"));
        let running = self.running().map_err(fail)?;
        self.adopt(&running, policy).map_err(fail)?;
        let ctx = handle(&running).context();
        let reference = splinter_sdk::release::arm_on(
            ctx.config(),
            &self.settings.base,
            policy.adapter.as_deref(),
        );
        ctx.model(&reference)
            .map_err(|e| CampaignError::Collector(format!("{e}")))
    }

    /// The model that answers as `policy` does, sampling at the host's
    /// prediction temperature (frontier selection).
    ///
    /// # Errors
    /// The policy cannot be served.
    pub fn predicting(&self, policy: &PolicyRef) -> Result<Model, CampaignError> {
        self.model_on(policy, |running| &running.predicting)
    }
}

impl splinter_sdk::campaign::served::PolicyModels for ServerHost {
    fn model(&self, policy: &PolicyRef, greedy: bool) -> Result<Model, CampaignError> {
        self.model_on(policy, |running| {
            if greedy {
                &running.greedy
            } else {
                &running.sampled
            }
        })
    }
}
