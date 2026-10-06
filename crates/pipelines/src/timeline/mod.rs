// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the full loop from longitudinal records to a
// released, audited risk model, for its clients. If your team needs expertise
// in training, evaluating and releasing time-to-event models, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The pipeline for models of subject timelines: a `timeline-v1` dataset is
//! trained, a candidate is scored against a champion on held-out units, and
//! it is released only if the pre-registered gate passes.
//!
//! * [`data`] - a record file becomes episodes, and episodes become the stored
//!   parts of one split.
//! * [`train`] - stored parts of one split become an immutable candidate.
//! * [`evaluate`] - a candidate and a champion scored on the same held-out
//!   units under requirements registered before scoring.
//! * [`release`] - an evaluation becomes a release only through the gate; a
//!   candidate that fails is recorded as rejected.
//! * [`lineage`] - from a release back to the line of each source file of every
//!   participant it was trained on.
//! * [`records`] - the record a candidate leaves.

pub mod data;
pub mod evaluate;
pub mod lineage;
pub mod records;
pub mod release;
pub mod train;
