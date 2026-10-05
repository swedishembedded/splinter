// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Subject timelines: brain's continuous-time time-to-event model and the
//! evaluation arithmetic it is judged with, as the rest of Splinter reaches
//! them (this crate is the only one that touches brain).
//!
//! A campaign builds `timeline-v1` records from its sources, partitions them
//! (`splinter_data::partition`), trains [`TimelineModel`]s on the folds and
//! scores them with [`survival`] on the outcomes [`observed`] extracts; the
//! decision about which model is better is the measurement crate's.

pub use brain::survival;
pub use brain::timeline::{
    observed, read_jsonl, synthetic, AtRisk, Event, Observation, Prediction, Subject,
    TimelineConfig, TimelineModel, TimelineReport, TimelineSpec, Value,
};
