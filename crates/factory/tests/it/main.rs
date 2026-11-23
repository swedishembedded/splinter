// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The factory's integration tests, linked as one binary.

#![allow(clippy::expect_used, clippy::unwrap_used)]

mod brain_trainer;
mod factory_run;
mod fakes;
mod foreach;
mod generations;
mod golden;
mod graph;
mod hinted;
mod native;
mod repo;
mod resolve;
mod reuse;
