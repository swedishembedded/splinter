// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

mod confinement;
mod container_sandbox;
mod containment;
mod distributed_plan;
mod jobs;
mod process_runner;
mod process_sandbox;
mod runtime_registry;
mod scratch;
mod slurm;
mod workspace;
