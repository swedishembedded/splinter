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

mod analysis;
mod backend;
mod blobs;
mod clocks;
mod collector;
mod compaction;
mod concurrency;
mod crash;
mod derivations;
mod entities;
mod evaluations;
mod experiments;
mod families;
mod graph;
mod hardening;
mod ids;
mod indexes;
mod ordering;
mod overlay;
mod packed;
mod processes;
mod properties;
mod queries;
mod segments;
mod session;
mod skills;
mod snapshots;
mod states;
mod streams;
mod verdicts;
mod verify;
mod windows;
