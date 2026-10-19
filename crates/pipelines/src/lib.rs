// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning agents that acquire a capability
// from a document or a tool and prove it with evidence, for its clients. If
// your team needs expertise in continual learning or agent evaluation, you
// can procure our services by sending an email to info@swedishembedded.com.

//! The stages and the pipelines: what each command does, composed from the
//! lower crates. One module per pipeline stage, each reading and writing the
//! content-addressed stores under the state root, and `learn` composing
//! them as a pipeline the orchestrator's engine runs. What a run is carried
//! out with - the configuration, the runtime and context, the roles, the
//! recorded run - is `splinter-orchestrator`'s.
//!
//! * [`sources`] - capture and inspect sources.
//! * [`tasks`] - sources become a task set.
//! * [`sessions`], [`claims`] - what a person taught an agent in recorded
//!   sessions: the sessions taken in with their secrets removed, claims
//!   proposed by a model, and the rulings of code on them.
//! * [`variants`] - the tasks kept for training asked again in other
//!   words, to be measured and never trained on.
//! * [`solving`] - a task set becomes an experience set.
//! * [`verify`] - verdicts appended, by each task kind's verifiers;
//!   [`judge`] - a judge's calibration.
//! * [`curriculum`] - budget spent where learning happens: pass@k frontier
//!   selection, concept mastery across releases, concepts the gate saw
//!   forgotten queued for new tasks, and diversity quotas on a training
//!   set.
//! * [`critique`] - failures critiqued and retried.
//! * [`datasets`] - experience sets projected into a stored dataset.
//! * [`train`] - datasets become a candidate adapter, continuing the
//!   champion with a replay of earlier releases' data.
//! * [`release`] - the gate a candidate passes to become the policy, the
//!   immutable releases and the aliases pointing at them, and rollback;
//!   [`eval`] - one model graded on the gate's suites.
//! * [`learn`] - every stage above, as one run; [`absorb`] - the sessions a
//!   person held with an agent, taught the same way; [`practice`] - a
//!   repository's tests made into tasks an agent solves in copies of its
//!   tree, trained on what its own checks confirm.
//! * [`runs`] - every command's run record, and cancelling one.
//! * [`doctor`] - whether a run can start here, before one does;
//! * [`status`], [`experiences`], [`ask`] - inspection and questions;
//!   [`compare`] - two experience sets of the same tasks compared, a family
//!   counted once;
//!   [`answers`] - every answer `ask` gave, and what gave it.
//! * [`lineage`] - from any artifact, where it came from and what came
//!   from it.
//! * [`router`] - a sentence becomes a command: a classifier reads it, code decides.

#![warn(missing_docs)]

pub mod absorb;
pub mod abstain;
pub mod ask;
pub mod author;
pub mod brain_server;
pub mod budget;
pub mod canaries;
pub mod checkpoints;
pub mod claim_quality;
pub mod claims;
pub mod compare;
pub mod critique;
pub mod curriculum;
pub mod dataset_ref;
mod dataset_views;
pub mod datasets;
pub mod describe;
pub mod dialogue;
pub mod doctor;
pub mod eval;
pub mod exam;
pub mod exam_seal;
pub mod exam_set;
pub mod experiences;
pub mod experiment;
pub mod family;
pub mod graded;
pub mod grouping;
pub mod harbor;
pub mod harness_opt;
pub mod index;
pub mod judge;
pub mod judging;
pub mod learn;
pub mod lineage;
pub mod plan;
pub mod powered;
pub mod generations;
pub mod practice;
pub mod raft;
pub mod rehearsal;
pub mod release;
pub mod rerank;
pub mod improve;
pub mod reserve;
pub mod retrieval;
pub mod router;
pub mod sessions;
pub mod solving;
pub mod sources;
pub mod state;
pub mod status;
pub mod tasks;
pub mod timeline;
pub mod train;
pub mod variants;
pub mod verify;
mod workspace_solve;
mod workspace_verify;
