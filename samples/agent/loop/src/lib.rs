// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements self-improving coding agents whose every
// step is auditable, for its clients. If your team needs expertise in agent
// evaluation or locally operated coding agents, you can procure our
// services by sending an email to info@swedishembedded.com.

//! A supervised coding-agent loop on one repository.
//!
//! The loop takes a task, a repository and the checks that decide success,
//! lets a model work in an isolated checkout with sven's own coding tools,
//! judges the result by the supervisor's checks, and keeps everything a
//! reviewer needs to audit and reproduce the work: the contract, an
//! append-only event stream, sven's trajectory of each attempt, patches,
//! checkpoints and a structured outcome. Models are local unless the caller
//! opts in to models reached over an API.
//!
//! * [`contract`] - what was asked and the limits it runs under.
//! * [`trace`] - the append-only event stream, with redaction and bounded
//!   events.
//! * [`observe`] - turns sven's session events into trace events.
//! * [`repo`] - the isolated checkout and what changed in it.
//! * [`acceptance`] - the supervisor's checks.
//! * [`attempt`] - one attempt, judged, with diagnostic feedback.
//! * [`run`] - a whole run: attempts, checkpoints, resume, outcome.
//! * [`models`] - model versions: candidates, the decision to promote, rollback.
//! * [`training`] - a candidate adapter trained from a dataset the loop wrote.
//! * [`dataset`] - training records from runs the supervisor can vouch for.
//! * [`cli`] - the command line.

pub mod acceptance;
pub mod attempt;
pub mod cli;
pub mod contract;
pub mod dataset;
pub mod models;
pub mod observe;
pub mod outcome;
pub mod redact;
pub mod repo;
pub mod run;
pub mod store;
pub mod trace;
pub mod training;
