// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Identifiers: content ids name bytes, record ids name a writer's records,
//! and neither needs a coordinator.

mod content;
mod record;
mod writer;

pub use content::ContentId;
pub use record::RecordId;
pub use writer::{WriterId, WriterIdentity};
