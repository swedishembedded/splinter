// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

mod adapter_manifest_is_brains;
mod device_selection;
mod timeline_dataset_is_trainable;
mod timeline_scoring;
mod timeline_serving;
mod timeline_units;
mod training_records;
mod views_datasets_are_trainable;
