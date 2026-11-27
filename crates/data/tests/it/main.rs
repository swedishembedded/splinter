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

mod attempt_pairs;
mod chat_tools;
mod corrections;
mod dataset;
mod dataset_store;
mod external_projection;
mod fuzz;
mod human_context;
mod judgement_views;
mod metamorphic;
mod online_calls;
mod online_sessions;
mod pass_curation;
mod policy_gradient;
mod projection_refusal;
mod recorded_preferences;
mod retrieval_triples;
mod session_dialogue;
mod sft_admission;
mod sft_final;
mod sft_step_controls;
mod sft_step_curation;
mod sft_step_packed;
mod sft_step_reasoning;
mod sft_step_requests;
mod sft_step_serialised;
mod source_views;
mod split;
mod split_bands;
mod strip;
mod supervision_enumeration;
mod supervision_mask;
mod timeline_projection;
mod timeline_store;
mod trajectory_views;
mod transition;
mod workspace_hygiene;
mod workspace_views;
