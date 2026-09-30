// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Knowledge intake: what Splinter learns from is captured as a source, and
//! a source becomes tasks and training records, one stage per module.
//!
//! * [`capture`] - a document, a repository tree or a command run captured
//!   as an immutable source for the source store.
//! * [`sections`] - a text split into addressable byte ranges: at headings
//!   for Markdown, at paragraphs otherwise.
//! * [`concepts`] - the concepts a task exercises: those it declares, else
//!   the (source, section) pairs its evidence falls in, else its kind.
//! * [`gates`] - the text rules a generated task is held to: instruction
//!   normalisation for duplicate detection, and numbers traceable to the
//!   evidence.
//! * [`denoise`] - a passage of a source part, corrupted, as a task to
//!   restore it: the first task generator feeding the experience store.
//! * [`tasks`] - tasks of many kinds proposed by a generator model and
//!   admitted by code: grounded in the source, self-contained, checked by
//!   running them where the answer is computed, and new to their batch.

#![warn(missing_docs)]

pub mod capture;
pub mod concepts;
pub mod denoise;
pub mod gates;
pub mod sections;
pub mod tasks;
