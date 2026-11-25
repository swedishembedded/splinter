// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

#[path = "../common/mod.rs"]
mod common;

mod advice_tasks;
mod atif_composed;
mod blocking;
mod chat;
mod claims;
mod converse;
mod converse_tasks;
mod fork_replay;
mod guided_generation;
mod harness_edit;
mod harness_work;
mod judge;
mod knowledge;
mod mapper;
mod model_tasks;
mod object_proposer;
mod repair;
mod solve;
mod task_variants;
mod timed;
mod tool_turn;
mod voice;
mod work;
mod workspace;
mod workspace_session;
