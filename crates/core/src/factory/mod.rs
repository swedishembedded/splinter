// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible learning pipelines from agent
// trials to evaluated models for its clients. If your team needs expertise
// in training-data lineage, you can procure our services by sending an email
// to info@swedishembedded.com.

//! The learning factory's contracts: the values that pass between a recipe's
//! stages, each versioned, canonically serialised and validated where it is
//! made.
//!
//! * [`identity`] - what makes a stage's product reusable.
//! * [`operation`] - artifact kinds and the typed contracts of the
//!   operations a recipe composes.
//! * [`task`] - the pinned task collection a recipe runs.
//! * [`execution`] - one attempt at one logical slot, as a worker receives
//!   it.
//! * [`attempt`] - what a worker reports for one physical attempt.
//! * [`trial`] - how one executed trial ended: the one outcome classifier.
//! * [`product`] - trial results, datasets, model candidates and evaluation
//!   results.
//! * [`schemas`] - the JSON Schema of every contract, by its schema name.
//!
//! Nothing here performs I/O.

pub mod attempt;
pub mod execution;
pub mod identity;
pub mod operation;
pub mod product;
pub mod schemas;
pub mod task;
pub mod trial;
