// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that put an informative on-policy
// learning signal through a language model against verifiers it cannot
// influence, for its clients. If your team needs expertise in running a
// lineage of policies behind one continuously batching server you can
// procure our services by sending an email to info@swedishembedded.com.

//! One `brain serve` for a whole experiment: the base, every adapter of it
//! offered from a directory the server follows, and training jobs on the same
//! cards. Exams, rollouts and training all go through its API, many requests
//! at once. A new adapter is hot-loaded by placing it in the directory and
//! waiting until the server lists it under its digest.
//!
//! The server is started on first use and stopped when the host is dropped
//! or [`ServedHost::stop`]ped; a host started again is a new server session.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use splinter_agent::solve::Model;
use splinter_agent::CancelToken;
use splinter_core::digest::Digest;
use splinter_model::served_train::ArtifactStore;
use splinter_model::BrainEndpoints;
use splinter_pipelines::brain_server::{self, Failure, Launch, Offer, Started};
use splinter_pipelines::campaign::served::PolicyModels;
use splinter_pipelines::campaign::{CampaignError, PolicyRef};
use splinter_store::StateRoot;

use crate::{Config, Context, Splinter};

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
    /// Splinter's state for the handles the models are asked through.
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
    /// agree best with `f32`; `None` is brain's own default tier.
    pub kv_cache: Option<String>,
    /// Sequences the server decodes together, and requests kept in flight.
    pub slots: u32,
    /// Directories put on the path of the commands the task kinds run (an
    /// interpreter a verifier needs, say); may be empty.
    pub runtime_dirs: Vec<PathBuf>,
    /// The sampling temperature of [`ServedHost::predicting`].
    pub predict_temperature: f32,
    /// Let the predicting and the greedy (exam) models reason first; `None`
    /// asks every model for no reasoning block, as before.
    pub reasoning: Option<ReasoningHost>,
}

/// How the models that answer with reasoning on decode. The proposing model
/// ([`ServedHost::context`]) never reasons.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReasoningHost {
    /// Nucleus sampling of the predicting model (`1.0` cuts nothing).
    pub top_p: f32,
    /// `top_k` of the predicting model (`0` cuts nothing).
    pub top_k: u32,
    /// How the exam model decodes.
    pub exam: ExamDecoding,
}

/// How the exam model ([`PolicyModels::model`] asked greedily) decodes when
/// it reasons.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExamDecoding {
    /// Temperature zero.
    Greedy,
    /// Sampled at the predicting temperature and top-p / top-k, every request
    /// carrying `seed`, so the same weights answer the same question alike.
    Sampled {
        /// The seed of the draws.
        seed: u64,
    },
}

/// What a handle asks of the server on every request.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Asked {
    temperature: f32,
    top_p: f32,
    top_k: u32,
    thinking: bool,
    seed: Option<u64>,
}

impl Asked {
    /// The training distribution itself, no reasoning: the proposing handle.
    const fn sampled(temperature: f32) -> Self {
        Self {
            temperature,
            top_p: 1.0,
            top_k: 0,
            thinking: false,
            seed: None,
        }
    }
}

impl HostSettings {
    /// What the predicting handle asks.
    fn predicting_asked(&self) -> Asked {
        match self.reasoning {
            None => Asked::sampled(self.predict_temperature),
            Some(r) => Asked {
                temperature: self.predict_temperature,
                top_p: r.top_p,
                top_k: r.top_k,
                thinking: true,
                seed: None,
            },
        }
    }

    /// What the exam (greedy) handle asks.
    fn exam_asked(&self) -> Asked {
        match self.reasoning {
            None => Asked::sampled(0.0),
            Some(r) => {
                let (temperature, seed) = match r.exam {
                    ExamDecoding::Greedy => (0.0, None),
                    ExamDecoding::Sampled { seed } => (self.predict_temperature, Some(seed)),
                };
                Asked {
                    temperature,
                    top_p: r.top_p,
                    top_k: r.top_k,
                    thinking: true,
                    seed,
                }
            }
        }
    }
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

fn failed(what: &str) -> impl Fn(String) -> CampaignError + '_ {
    move |why| CampaignError::Config(format!("{what}: {why}"))
}

/// The server of an experiment; see the module documentation.
pub struct ServedHost {
    settings: HostSettings,
    running: Mutex<Option<Arc<Running>>>,
    /// Handles whose runtimes look in other directories than the host's,
    /// by those directories (see [`ServedHost::context_in`]).
    scoped: Mutex<std::collections::BTreeMap<Vec<PathBuf>, Splinter>>,
}

impl ServedHost {
    /// A host that has started nothing yet.
    #[must_use]
    pub fn new(settings: HostSettings) -> Self {
        Self {
            settings,
            running: Mutex::new(None),
            scoped: Mutex::new(std::collections::BTreeMap::new()),
        }
    }

    fn handle(&self, endpoints: &BrainEndpoints, asked: Asked) -> Result<Splinter, CampaignError> {
        self.handle_in(endpoints, asked, &self.settings.runtime_dirs)
    }

    fn handle_in(
        &self,
        endpoints: &BrainEndpoints,
        asked: Asked,
        runtime_dirs: &[PathBuf],
    ) -> Result<Splinter, CampaignError> {
        let s = &self.settings;
        let absolute = |p: &PathBuf| {
            std::path::absolute(p)
                .map_err(|e| failed("resolving a path")(format!("{}: {e}", p.display())))
        };
        let mut config = Config::from_env();
        config.state_root = StateRoot::new(&s.state);
        config.policy_base = absolute(&s.base)?;
        config.model_store = s.models.clone();
        config.workspace.runtime_dirs = runtime_dirs
            .iter()
            .map(absolute)
            .collect::<Result<_, _>>()?;
        config.thinking = asked.thinking;
        config.allow_remote = false;
        config.brain = Some(endpoints.clone());
        // Without reasoning, the training distribution itself: temperature
        // as asked, no nucleus or top-k cut.
        config.served_temperature = Some(asked.temperature);
        config.served_top_p = Some(asked.top_p);
        config.served_top_k = Some(asked.top_k);
        if asked.seed.is_some() {
            config.served_seed = asked.seed;
        }
        Splinter::builder(config)
            .build()
            .map_err(|e| failed("starting a Splinter handle on the server")(e.to_string()))
    }

    fn launch(&self) -> Result<Running, CampaignError> {
        let s = &self.settings;
        for dir in [&s.work, &s.store, &s.adapters] {
            std::fs::create_dir_all(dir).map_err(|e| {
                failed("creating a server directory")(format!("{}: {e}", dir.display()))
            })?;
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
            kv_cache: s.kv_cache.as_deref(),
            artifact_store: Some(&s.store),
            log: Some(&s.log),
        };
        let server = brain_server::start(&launch, &CancelToken::new()).map_err(|f| match f {
            Failure::Cancelled => failed("brain serve")("starting it was cancelled".into()),
            Failure::Unmeasured(why) => failed("brain serve did not start")(why),
        })?;
        let store = ArtifactStore::open(&s.store)
            .map_err(|e| failed("the artifact store")(e.to_string()))?;
        let endpoints = BrainEndpoints::new(
            vec![format!("http://{}/v1", server.address)],
            server.key.clone(),
        )
        .map_err(|e| failed("the server's endpoints")(e.to_string()))?
        .with_attempt_concurrency(s.slots as usize)
        .with_artifact_store(store);
        Ok(Running {
            sampled: self.handle(&endpoints, Asked::sampled(1.0))?,
            greedy: self.handle(&endpoints, s.exam_asked())?,
            predicting: self.handle(&endpoints, s.predicting_asked())?,
            endpoints,
            _server: Mutex::new(server),
        })
    }

    fn running(&self) -> Result<Arc<Running>, CampaignError> {
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
    pub fn endpoints(&self) -> Result<BrainEndpoints, CampaignError> {
        Ok(self.running()?.endpoints.clone())
    }

    /// Makes `policy` servable: its adapter, when it has one, is offered to
    /// the server's directory and waited for until the server lists it under
    /// the digest of the file.
    fn adopt(&self, running: &Running, policy: &PolicyRef) -> Result<(), CampaignError> {
        let Some(adapter) = &policy.adapter else {
            return Ok(());
        };
        let offer = failed("offering an adapter");
        let digest = std::fs::File::open(adapter)
            .and_then(Digest::sha256_of_reader)
            .map_err(|e| offer(format!("{}: {e}", adapter.display())))?
            .to_string();
        let target = self.settings.adapters.join(format!(
            "{}.safetensors",
            digest.trim_start_matches("sha256:")
        ));
        if !target.exists() {
            let partial = target.with_extension("partial");
            std::fs::copy(adapter, &partial)
                .and_then(|_| std::fs::rename(&partial, &target))
                .map_err(|e| offer(format!("{}: {e}", adapter.display())))?;
        }
        let deadline = Instant::now() + ADOPT_WAIT;
        loop {
            let listed = running
                .endpoints
                .list(0)
                .map_err(|e| offer(e.to_string()))?;
            if listed
                .iter()
                .any(|card| card.adapter_digest.as_deref() == Some(digest.as_str()))
            {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(offer(format!(
                    "the server did not list the adapter {digest} within {}s",
                    ADOPT_WAIT.as_secs()
                )));
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }

    /// The context that samples at temperature one, with `policy` servable.
    ///
    /// # Errors
    /// The server does not start or does not list the policy's adapter.
    pub fn context(&self, policy: &PolicyRef) -> Result<Context, CampaignError> {
        let running = self.running()?;
        self.adopt(&running, policy)?;
        Ok(running.sampled.context())
    }

    /// The context that samples at temperature one, with `policy` servable,
    /// whose task runtimes find their interpreters in `runtime_dirs` first
    /// (instead of the host's own directories): the proposer of a task
    /// family whose interpreter is not the host's default.
    ///
    /// # Errors
    /// The server does not start or does not list the policy's adapter.
    pub fn context_in(
        &self,
        policy: &PolicyRef,
        runtime_dirs: &[PathBuf],
    ) -> Result<Context, CampaignError> {
        let running = self.running()?;
        self.adopt(&running, policy)?;
        let mut scoped = self
            .scoped
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !scoped.contains_key(runtime_dirs) {
            let handle = self.handle_in(&running.endpoints, Asked::sampled(1.0), runtime_dirs)?;
            scoped.insert(runtime_dirs.to_vec(), handle);
        }
        Ok(scoped[runtime_dirs].context())
    }

    /// Stops the server; the next use starts a new server session.
    pub fn stop(&self) {
        self.scoped
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
        *self
            .running
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    }

    /// The model that answers as `policy` does, decoding through `handle`.
    fn model_on(
        &self,
        policy: &PolicyRef,
        handle: impl FnOnce(&Running) -> &Splinter,
    ) -> Result<Model, CampaignError> {
        let running = self.running()?;
        self.adopt(&running, policy)?;
        let ctx = handle(&running).context();
        let reference = splinter_pipelines::release::arm_on(
            ctx.config(),
            &self.settings.base,
            policy.adapter.as_deref(),
        );
        ctx.model(&reference)
            .map_err(|e| CampaignError::Collector(format!("{e}")))
    }

    /// The model that answers as `policy` does, sampling at the host's
    /// prediction temperature.
    ///
    /// # Errors
    /// The policy cannot be served.
    pub fn predicting(&self, policy: &PolicyRef) -> Result<Model, CampaignError> {
        self.model_on(policy, |running| &running.predicting)
    }
}

impl PolicyModels for ServedHost {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(reasoning: Option<ReasoningHost>) -> HostSettings {
        HostSettings {
            brain: PathBuf::new(),
            models: PathBuf::new(),
            base: PathBuf::new(),
            work: PathBuf::new(),
            state: PathBuf::new(),
            store: PathBuf::new(),
            adapters: PathBuf::new(),
            log: PathBuf::new(),
            device: String::new(),
            kv_cache: None,
            slots: 1,
            runtime_dirs: Vec::new(),
            predict_temperature: 0.7,
            reasoning,
        }
    }

    #[test]
    fn without_reasoning_every_handle_asks_what_it_always_asked() {
        let s = settings(None);
        assert_eq!(s.predicting_asked(), Asked::sampled(0.7));
        assert_eq!(s.exam_asked(), Asked::sampled(0.0));
    }

    #[test]
    fn with_reasoning_the_predicting_and_exam_handles_think_and_the_exam_decodes_as_stated() {
        let host = |exam| {
            settings(Some(ReasoningHost {
                top_p: 0.95,
                top_k: 20,
                exam,
            }))
        };
        let greedy = host(ExamDecoding::Greedy);
        assert_eq!(
            greedy.predicting_asked(),
            Asked {
                temperature: 0.7,
                top_p: 0.95,
                top_k: 20,
                thinking: true,
                seed: None
            }
        );
        assert_eq!(
            (
                greedy.exam_asked().temperature,
                greedy.exam_asked().thinking,
                greedy.exam_asked().seed
            ),
            (0.0, true, None)
        );
        // seed: it is the value every request must carry
        let sampled = host(ExamDecoding::Sampled { seed: 9 }).exam_asked();
        assert_eq!(
            (sampled.temperature, sampled.thinking, sampled.seed),
            (0.7, true, Some(9))
        );
    }
}
