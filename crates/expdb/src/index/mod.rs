// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Indexes: immutable projections of segments.
//!
//! A run is a sorted table of where every record lives and what links it to
//! its neighbours. Runs are written once, merged in the background, and never
//! updated in place. A reader assembles an [`Index`] from the runs that cover
//! a snapshot's segments and scans only the segments no run covers.

mod csr;
mod model;
mod persist;
mod read;
mod run;

pub use model::{Index, Loc};
pub use read::ScanStats;
pub(crate) use read::SnapshotCache;
