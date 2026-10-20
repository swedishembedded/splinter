// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! A git history replayed by an agent, commit by commit, into verified
//! supervised fine-tuning data.
//!
//! * [`git`], [`tree`], [`select`], [`generated`] - reading the history and
//!   choosing the commits that can be replayed.
//! * [`plan_prompt`], [`plan`] - the task of a commit, written without the
//!   answer in it.
//! * [`grounding_prompt`], [`attempt`] - the replay: an agent in a tree that
//!   holds only the parent.
//! * [`gate`], [`trace`] - the deterministic judgement.
//! * [`pipeline`], [`ledger`] - the run, its bookkeeping and the bias report.
//! * [`dataset`], [`serving_prompt`], [`scrub`] - what leaves the run.
//! * [`fixture`], [`scripted`], [`audit`] - the sample checking itself with no model.

#![warn(missing_docs)]

pub mod attempt;
pub mod audit;
pub mod budget;
pub mod compare;
pub mod control;
pub mod dataset;
pub mod fixture;
pub mod gate;
pub mod generated;
pub mod git;
pub mod grounding_prompt;
pub mod justify;
pub mod ledger;
pub mod mask;
pub mod pipeline;
pub mod plan;
pub mod plan_only_prompt;
pub mod plan_prompt;
pub mod scripted;
pub mod scrub;
pub mod select;
pub mod serving_prompt;
pub mod splits;
pub mod timed;
pub mod trace;
pub mod tree;
