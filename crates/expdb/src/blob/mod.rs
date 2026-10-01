// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Content-addressed blobs: large objects are split into content-defined
//! chunks, chunks are bundled into immutable packs, and identical content is
//! stored once however many states, tools or snapshots share it.

mod chunker;
mod pack;
mod store;
mod tree;

pub use store::{BlobRef, BlobStats, BlobStore};
pub use tree::{Entry, Tree};
