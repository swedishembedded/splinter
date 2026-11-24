// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The factory's integration tests, linked as one binary.

#![allow(clippy::expect_used, clippy::unwrap_used)]

mod agent_op;
mod brain_trainer;
mod closed_book;
mod dataset_extend;
mod dataset_import;
mod eval_given;
mod external;
mod factory_run;
mod fakes;
mod foreach;
mod fuzz;
mod generations;
mod golden;
mod graph;
mod harness_op;
mod harness_pass;
mod harness_search;
mod hinted;
mod import;
mod interventions;
mod knowledge;
mod machine_config;
mod native;
mod operator_security;
mod practice;
mod projections;
mod promotion;
mod qualify;
mod recompute;
mod registry;
mod repo;
mod resolve;
mod reuse;
mod template_runs;
mod templates;
mod text_dataset;
