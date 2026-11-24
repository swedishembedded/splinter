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

mod capture;
mod capture_git;
mod capture_tree;
mod claim_agreement;
mod claim_entailment;
mod claim_forms;
mod claim_gates;
mod claim_supersession;
mod claim_task;
mod denoise_task;
mod external_bm25;
mod external_skill_order;
mod factory_admission;
mod factory_documents;
mod factory_firmware;
mod factory_repositories;
mod factory_store;
mod generation_calls;
mod history_tasks;
mod redact;
mod retrieve;
mod sections;
mod session_import;
mod session_intake;
mod stance;
mod survey;
mod tabular;
mod workspace_harbor;
mod workspace_report_cap;
mod workspace_requests;
mod workspace_shared_helpers;
mod workspace_task_directory;
mod workspace_tasks;
mod workspace_tasks_cap;
