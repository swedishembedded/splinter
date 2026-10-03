// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that turn agent experience
// into better local models, for its clients. If your team needs expertise in
// continual learning, agent evaluation or embedding such a system in a
// product, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The [`Splinter`] handle and its builder.

use std::sync::Arc;

use serde::Serialize;
use splinter_orchestrator::runs::{record, Recorded, Recorder};
use splinter_orchestrator::{Config, Context, OrchestratorError, Runtime};
use splinter_pipelines::learn::{learn, LearnRequest, Learned};
use splinter_pipelines::train::{BrainTrainer, Trainer};

/// Receives each stage's summary as it finishes: `(stage, summary)`.
pub type Progress = Arc<dyn Fn(&str, &serde_json::Value) + Send + Sync>;

/// What a process shares across its commands.
pub struct Splinter {
    runtime: Arc<Runtime>,
    trainer: Arc<dyn Trainer + Send + Sync>,
    progress: Option<Progress>,
}

/// How a [`Splinter`] is put together.
pub struct SplinterBuilder {
    config: Config,
    allow_remote: bool,
    trainer: Arc<dyn Trainer + Send + Sync>,
    progress: Option<Progress>,
}

impl SplinterBuilder {
    /// Opts into models reached over the network, as `--allow-remote` does.
    #[must_use]
    pub fn allow_remote(mut self, allow: bool) -> Self {
        self.allow_remote = allow;
        self
    }

    /// Trains with `trainer` instead of brain's.
    #[must_use]
    pub fn trainer(mut self, trainer: Arc<dyn Trainer + Send + Sync>) -> Self {
        self.trainer = trainer;
        self
    }

    /// Reports each finished stage of every command to `progress`.
    #[must_use]
    pub fn progress(mut self, progress: Progress) -> Self {
        self.progress = Some(progress);
        self
    }

    /// The handle, with its runtime started.
    pub fn build(self) -> Result<Splinter, OrchestratorError> {
        Ok(Splinter {
            runtime: Arc::new(Runtime::new(self.config, self.allow_remote)?),
            trainer: self.trainer,
            progress: self.progress,
        })
    }
}

impl Splinter {
    /// A builder over `config`.
    #[must_use]
    pub fn builder(config: Config) -> SplinterBuilder {
        SplinterBuilder {
            config,
            allow_remote: false,
            trainer: Arc::new(BrainTrainer),
            progress: None,
        }
    }

    /// A handle configured from the environment (see [`Config::from_env`]),
    /// training with brain.
    pub fn from_env() -> Result<Self, OrchestratorError> {
        Self::builder(Config::from_env()).build()
    }

    /// The runtime this handle shares across commands.
    #[must_use]
    pub fn runtime(&self) -> &Arc<Runtime> {
        &self.runtime
    }

    /// The trainer this handle trains with.
    #[must_use]
    pub fn trainer(&self) -> &dyn Trainer {
        self.trainer.as_ref()
    }

    /// The context of one command: its own pin of each policy alias on the
    /// shared runtime, reporting to this handle's progress receiver.
    #[must_use]
    pub fn context(&self) -> Context {
        let context = self.runtime.context();
        match &self.progress {
            Some(progress) => {
                let progress = Arc::clone(progress);
                context.with_progress(Box::new(move |stage, summary| progress(stage, summary)))
            }
            None => context,
        }
    }

    /// Runs `work` as a recorded run of `command` with `arguments`, in a
    /// context of its own: the run's stages, how it ended and what it
    /// produced are kept, readable and cancellable from any process.
    pub fn run<R: Serialize>(
        &self,
        command: &str,
        arguments: &impl Serialize,
        work: impl FnOnce(&Context, &mut Recorder<'_>) -> Result<R, OrchestratorError>,
    ) -> Result<Recorded<R>, OrchestratorError> {
        let ctx = self.context();
        record(&ctx, command, arguments, |run| work(&ctx, run))
    }

    /// `learn`, training with this handle's trainer.
    pub fn learn(&self, request: &LearnRequest) -> Result<Learned, OrchestratorError> {
        learn(&self.context(), request, self.trainer())
    }
}
