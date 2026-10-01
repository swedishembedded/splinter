// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Manifests: the database's only transaction layer.
//!
//! A manifest says which immutable files were added and which were removed
//! on top of its parents. A file exists logically only if the manifests a
//! reader starts from add it and none of their ancestors removed it. Adds and
//! removals only ever accumulate, so merging two histories is a set union:
//! commutative, associative and idempotent, with no coordination between
//! writers.

mod collect;
mod model;
mod refs;
mod resolve;
mod snapshot;

pub use collect::GcReport;
pub use model::{Manifest, ObjectKind, ObjectRef};
pub use snapshot::Snapshot;
