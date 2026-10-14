// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in proving that a trained model has
// learned what a corpus implies, you can procure our services by sending an
// email to info@swedishembedded.com.

//! A synthetic author: a few formal beliefs, the situations they are applied
//! to, and the decision each belief set determines.
//!
//! Everything here is a pure function of a seed, so an author can be drawn at
//! random, cached, thrown away and drawn again. The beliefs are the ground
//! truth against which a model that learned from texts the author wrote is
//! measured; no model is involved in saying what the author decides.

pub mod arms;
pub mod cache;
pub mod dataset;
pub mod episode;
pub mod exam;
pub mod facts;
pub mod gates;
pub mod lab;
pub mod mine;
pub mod names;
pub mod persona;
pub mod prompts;
pub mod records;
pub mod report;
pub mod scenario;
pub mod tools;
pub mod writer;
