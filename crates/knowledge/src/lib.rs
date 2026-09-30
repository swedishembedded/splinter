// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Knowledge intake: what Splinter learns from is captured as a source, and
//! a source becomes tasks and training records, one stage per module.
//!
//! * [`capture`] - a document, a repository tree or a command run captured
//!   as an immutable source for the source store.
//! * [`sections`] - a text split into addressable byte ranges: at headings
//!   for Markdown, at paragraphs otherwise; the fact extractor's chunks
//!   each carry the document's subject.
//! * [`extract`] - ask for every fact in a section; parse the reply strictly.
//! * [`gates`] - refuse duplicates, questions not anchored on the subject,
//!   and answers whose numbers the section does not carry.
//! * [`negatives`] - out-of-scope variants trained toward an abstention.
//! * [`explore`] - the whole run, traced like an agent attempt.
//! * [`denoise`] - a passage of a source part, corrupted, as a task to
//!   restore it: the first task generator feeding the experience store.
//! * [`tasks`] - tasks of many kinds proposed by a generator model and
//!   admitted by code: grounded in the source, self-contained, checked by
//!   running them where the answer is computed, and new to their batch.

#![warn(missing_docs)]

pub mod capture;
pub mod denoise;
pub mod explore;
pub mod extract;
pub mod gates;
pub mod negatives;
pub mod sections;
pub mod tasks;

pub use explore::{run, ExploreOptions, ExploreSummary};
