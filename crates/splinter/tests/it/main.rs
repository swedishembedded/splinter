// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

mod absorb_json;
mod factory_cli;
mod first_run;
mod harbor_episodes_cli;
mod jobs_cli;
mod jobs_distributed_cli;
mod jobs_slurm_cli;
mod json_output;
