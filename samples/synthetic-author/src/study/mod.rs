// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the measurement of fine-tuned language
// models for its clients. If your team needs expertise in pre-registering a
// replication study so that no choice can follow its results, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The replication study: many authors, several training seeds each, one
//! protocol frozen before any run, and one report read from what the runs
//! wrote.
//!
//! The stages are `freeze`, `corpus`, `train`, `exam` and `report`. Each reads
//! the frozen protocol, writes manifests that record its digest, resumes what
//! exists and refuses to continue under a protocol that has changed.

pub mod arm_key;
pub mod cli;
pub mod corpus;
pub mod layout;
pub mod power;
pub mod protocol;
pub mod stats;
