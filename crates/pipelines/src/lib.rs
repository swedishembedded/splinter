// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning agents that acquire a capability
// from a document or a tool and prove it with evidence, for its clients. If
// your team needs expertise in continual learning or agent evaluation, you
// can procure our services by sending an email to info@swedishembedded.com.

//! The stages and the pipelines: what each command does, composed from the
//! lower crates. One module per pipeline stage, each reading and writing the
//! content-addressed stores under the state root, and `absorb` composing
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
//! * [`episodes`] - practice task folders run as episodes: a policy attempts
//!   each task again and again in fresh workspaces, a verifier it cannot
//!   reach grades each attempt, and every attempt is kept as experience with
//!   its reward, its typed outcome and its policy trace.
//! * [`verify`] - verdicts appended, by each task kind's verifiers;
//!   [`judge`] - a judge's calibration.
//! * [`curriculum`] - budget spent where learning happens: pass@k frontier
//!   selection, concept mastery across releases, concepts the gate saw
//!   forgotten queued for new tasks.
//! * [`goal`] - goal-directed learning: probe the participant, find its
//!   gaps, choose the raw material expected to close them under a budget
//!   with a reason for every section, learn it, probe again; and the
//!   comparison against reading everything, nothing or the same words at
//!   random.
//! * [`improve`] - which way of improving an agent on a target capability
//!   (more samples, another harness, a trained candidate, no change) is
//!   worth its cost, decided from what each was measured to do.
//! * [`critique`] - failures critiqued and retried.
//! * [`datasets`] - experience sets projected into a stored dataset.
//! * [`train`] - datasets become a candidate adapter, continuing the
//!   champion with a replay of earlier releases' data.
//! * [`release`] - the gate a candidate passes to become the policy, the
//!   immutable releases and the aliases pointing at them, and rollback;
//!   [`eval`] - one model graded on the gate's suites.
//! * [`absorb`] - the sessions a person held with an agent, taught with the
//!   stages above as one run.
//! * [`runs`] - every command's run record, and cancelling one.
//! * [`doctor`] - whether a run can start here, before one does;
//! * [`serving`] - the brain servers kept for other programs until interrupted;
//! * [`status`], [`experiences`], [`ask`] - inspection and questions;
//!   [`compare`] - two experience sets of the same tasks compared, a family
//!   counted once; [`arm_compare`] - graded trials of several arms (Harbor
//!   jobs or a trials file) each against a reference arm;
//!   [`answers`] - every answer `ask` gave, and what gave it.
//! * [`lineage`] - from any artifact, where it came from and what came
//!   from it.
//! * [`router`] - a sentence becomes a command: a classifier reads it, code decides.

#![warn(missing_docs)]

pub mod absorb;
pub mod abstain;
pub mod arm_compare;
pub mod ask;
mod attempt_store;
pub mod bench;
mod block;
pub use splinter_model::server as brain_server;
pub mod budget;
pub mod canaries;
pub mod checkpoints;
pub mod claim_quality;
pub mod claims;
pub mod compare;
pub mod conversation_admission;
pub mod conversation_policy_gradient;
pub mod critique;
pub mod curriculum;
pub mod dataset_ref;
mod dataset_views;
pub mod datasets;
pub mod decision_fixtures;
pub mod describe;
pub mod dialogue;
pub mod doctor;
pub mod episodes;
pub mod eval;
pub mod exam;
pub mod exam_seal;
pub mod exam_set;
pub mod experiences;
pub mod experiment;
pub mod factory_index;
pub mod family;
pub mod goal;
pub mod graded;
pub mod grouping;
pub mod harbor;
pub mod history;
pub mod hub;
pub mod improve;
pub mod index;
pub mod judge;
pub mod judging;
pub mod lineage;
pub mod objects;
pub mod policy_used;
pub mod powered;
pub mod preflight;
mod promotion;
pub mod raft;
pub mod rehearsal;
pub mod release;
pub mod rerank;
pub mod reserve;
pub mod retrieval;
mod rl_build;
pub mod router;
pub mod sealed;
pub mod served;
pub mod serving;
pub mod sessions;
pub mod set_import;
pub mod skill_library;
pub mod solving;
pub mod sources;
pub mod state;
pub mod status;
pub mod tasks;
pub mod timeline;
pub mod train;
pub mod variants;
pub mod verify;
mod workspace_held;
mod workspace_solve;
mod workspace_verify;
