// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning factories that take a model from
// agent rollouts to an evaluated, released candidate, for its clients. If
// your team needs expertise in model fine-tuning pipelines, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The learning factory: one recipe, planned into a graph of reusable
//! stages and carried out by one durable controller.
//!
//! * [`recipe`] - the recipe file and the checks between its sections.
//! * [`connections`], [`environment`], [`collection`], [`eval_plan`] - the
//!   files a recipe refers to: how models are reached, where agents run,
//!   which tasks, and how candidates are judged.
//! * [`resolve`] - a recipe with every file it names read, checked and
//!   pinned in a lock.
//! * [`plan`] - the graph of stages.
//! * [`run`] - the controller over one run directory, and every stage.
//! * [`backend`] and [`training`] - the seams to the execution worker and
//!   the training engine.
//! * [`champion`] - the model a finished run leaves in use, as a later
//!   recipe's `champion:RUN`.
//! * [`hint`] - hinted passes: tasks worked with their reference solution
//!   shown, learned without it.
//! * [`tree`] - the digest of a directory tree.

#![warn(missing_docs)]

pub mod backend;
pub mod champion;
pub mod collection;
pub mod connections;
pub mod environment;
pub mod eval_plan;
pub mod hint;
pub mod plan;
pub mod recipe;
pub mod resolve;
pub mod run;
pub mod training;
pub mod tree;
