// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Training views: reproducible projections of the experience graph.
//!
//! A [`Recipe`] says what is wanted; compiled against a snapshot it becomes a
//! [`TrainingPlan`], an immutable list of samples that point at the stored
//! experience instead of copying it. The canonical dataset is the recipe
//! together with the snapshot it ran on. Exports and token caches are only
//! ever derived from it.

mod dpo;
mod export;
mod grpo;
mod loader;
mod plan;
mod ppo;
mod prm;
mod recipe;
mod render;
mod rng;
mod select;
mod sft;

pub use export::{Materialized, MaterializedRollout};
pub use loader::{Loader, LoaderConfig};
pub use plan::{DataRef, Rollout, Sample, SampleBody, TrainingPlan};
pub use recipe::{Objective, Recipe};
pub use rng::Rng;
