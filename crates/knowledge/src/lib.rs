// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Knowledge intake: a source document becomes question/answer training
//! records, one stage per module.
//!
//! * [`sections`] - split at headings and paragraph boundaries; every
//!   section carries the document's subject.
//! * [`extract`] - ask for every fact in a section; parse the reply strictly.
//! * [`gates`] - refuse duplicates, questions not anchored on the subject,
//!   and answers whose numbers the section does not carry.
//! * [`negatives`] - out-of-scope variants trained toward an abstention.
//! * [`explore`] - the whole run, traced like an agent attempt.
//! * [`denoise`] - a passage of a source, corrupted, as a task to restore
//!   it: the first task generator feeding the experience store.

#![warn(missing_docs)]

pub mod denoise;
pub mod explore;
pub mod extract;
pub mod gates;
pub mod negatives;
pub mod sections;

pub use explore::{run, ExploreOptions, ExploreSummary};
