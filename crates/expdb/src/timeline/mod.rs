// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Reading experience by time: episodes and their streams, clocks resolved
//! across devices, windows over every stream, exact sample slices and
//! questions asked around events.

mod clock;
mod data;
mod samples;
mod views;
mod windows;

pub use clock::ClockTransform;
pub(crate) use data::TimelineData;
pub use samples::Samples;
pub use views::{ActionView, ChunkView, CorrespondenceView, EpisodeView, EventView, StreamView};
pub use windows::{EventQuery, EventWindow, StreamFilter, StreamWindow};
