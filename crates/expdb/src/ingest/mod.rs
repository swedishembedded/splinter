// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Getting experience in: a per-writer buffer that seals immutable segments,
//! an optional node-local spool with an aggregator, and the collector API
//! that turns an agent's run into records.

mod aggregator;
mod collector;
mod derive;
mod run;
mod writer;

pub use aggregator::{Aggregator, DrainReport};
pub use collector::Collector;
pub use run::{DecisionBuilder, DecisionRef, Run};
pub use writer::{Destination, Writer};
