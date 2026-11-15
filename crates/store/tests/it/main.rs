// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

mod artifacts;
mod bundle;
mod campaign;
mod claim_store;
mod documents;
mod experience_store;
mod identity;
mod jsonl;
mod lineage;
mod longitudinal_import;
mod maintenance;
mod pointers;
mod projection;
mod recovery;
mod run_log;
mod source_store;
mod stage_ledger;
mod task_store;
