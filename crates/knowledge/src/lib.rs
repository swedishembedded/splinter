// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Knowledge intake: what Splinter learns from is captured as a source, and
//! a source becomes tasks and training records, one stage per module.
//!
//! * [`capture`] - a document, a repository tree or a command run captured
//!   as an immutable source for the source store.
//! * [`session`] - a captured session read as text addressable by step.
//! * [`sections`] - a text split into addressable byte ranges: at headings
//!   for Markdown, at paragraphs otherwise.
//! * [`concepts`] - the concepts a task exercises: those it declares, else
//!   the (source, section) pairs its evidence falls in, else its kind.
//! * [`material`] - what a teacher is shown of a task: the source
//!   sections its evidence falls in, its passages and hints - never its
//!   reference.
//! * [`redact`] - secrets removed from text before it is stored or shown
//!   to a model.
//! * [`retrieve`] - the passages of the sources that bear on a query:
//!   lexical and semantic ranking, fused.
//! * [`gates`] - the text rules a generated task is held to: instruction
//!   normalisation for duplicate detection, and numbers traceable to the
//!   evidence.
//! * [`denoise`] - a passage of a source part, corrupted, as a task to
//!   restore it: the first task generator feeding the experience store.
//! * [`codebook`] - what a record file's variables mean, as published.
//! * [`harmonize`] - a variable mapped onto a shared concept, proposed by a
//!   model with the codebook words it relied on and admitted by code.
//! * [`tabular`] - record files as columns: the SAS transport format
//!   public survey and cohort data are distributed in.
//! * [`tasks`] - tasks of many kinds proposed by a generator model and
//!   admitted by code: grounded in the source, self-contained, checked by
//!   running them where the answer is computed, and new to their batch.
//! * [`rehearsal`] - general tasks for a base model to answer so its own
//!   answers can be replayed beside new training: sums and format requests
//!   built by code with fresh numbers, the brief a model writes general
//!   requests from, and the admission that keeps every anchor task and its
//!   near copies out.

#![warn(missing_docs)]

pub mod advice;
pub mod capture;
pub mod codebook;
pub mod concepts;
pub mod denoise;
pub mod gates;
pub mod harmonize;
pub mod material;
pub mod redact;
pub mod rehearsal;
pub mod retrieve;
pub mod sections;
mod seeded;
pub mod session;
pub mod survey;
pub mod tabular;
pub mod tasks;
