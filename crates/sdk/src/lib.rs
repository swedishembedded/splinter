// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that turn agent experience
// into better local models, for its clients. If your team needs expertise in
// continual learning, agent evaluation or embedding such a system in a
// product, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The one way to embed Splinter.
//!
//! A [`Splinter`] owns what a process shares - the runtime with its loaded
//! models, the trainer, where progress is reported - and hands out a context
//! per command, so an alias that moves between two commands is seen by the
//! second. Every command that writes state runs as a recorded run:
//! [`Splinter::run`] is the one place that wrapping lives.
//!
//! ```no_run
//! use splinter_sdk::learn::LearnRequest;
//! use splinter_sdk::Splinter;
//!
//! # fn main() -> Result<(), splinter_sdk::Error> {
//! let splinter = Splinter::from_env()?;
//! let request = LearnRequest {
//!     sources: vec!["./manual.md".into()],
//!     goal: Some("the console and power limits".into()),
//!     ..LearnRequest::default()
//! };
//! let learned = splinter.learn(&request)?;
//! # let _ = learned;
//! # Ok(())
//! # }
//! ```
//!
//! # What a caller works with
//!
//! * The commands, one module each - [`learn`], [`ask`], [`sources`],
//!   [`tasks`], [`timeline`], [`solving`], [`verify`], [`critique`], [`datasets`],
//!   [`train`], [`release`], [`eval`], [`lineage`], [`status`], [`state`],
//!   [`router`] and the rest - with their requests and reports.
//! * What a command works with: [`Config`], [`Context`], [`Runtime`], the
//!   roles ([`roles`]) and the recorded run ([`runs`]), and the pipeline
//!   engine ([`pipeline`]) a new pipeline is defined on.
//! * The layers a caller reaches into, each under one name: [`vocabulary`]
//!   (what Splinter talks about), [`store`] (durable state), [`measure`]
//!   (verifiers, the release gate), [`data`] (training data from
//!   experience), [`knowledge`] (intake and task generation), [`agent`]
//!   (the sven adapter), [`model`] (the brain adapter) and [`sandbox`].
//!
//! The command line, and every sample, depend on this crate and on nothing
//! else of Splinter's: if they need more, this crate is missing something.

#![warn(missing_docs)]

mod splinter;

pub use crate::splinter::{Progress, Splinter, SplinterBuilder};

// What a command works with.
pub use splinter_orchestrator::{
    answers, concurrency, config, context, ids, model_ref, pipeline, releases, roles, runs,
};
pub use splinter_orchestrator::{Config, Context, OrchestratorError as Error, Runtime};

// The commands.
pub use splinter_pipelines::{
    ask, budget, critique, curriculum, datasets, dialogue, eval, exam, experiences, judge, learn,
    lineage, plan, raft, rehearsal, release, rerank, retrieval, router, solving, sources, state,
    status, tasks, timeline, train, variants, verify,
};

// The layers, each under one name.
pub use splinter_agent as agent;
pub use splinter_core as vocabulary;
pub use splinter_data as data;
pub use splinter_eval as measure;
pub use splinter_knowledge as knowledge;
pub use splinter_model as model;
pub use splinter_sandbox as sandbox;
pub use splinter_store as store;
