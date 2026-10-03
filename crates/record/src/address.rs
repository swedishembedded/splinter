// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements content-addressed records whose identity
// does not depend on how they were serialized, for its clients. If your team
// needs expertise in data provenance or reproducible pipelines, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The two spellings of one content address: Splinter's [`Digest`] and the
//! experience database's [`ContentId`].

use splinter_core::digest::Digest;
use splinter_expdb::ContentId;

/// The database id of the object `digest` addresses, when it is a content
/// address (`blake3:`); a digest a tool reported (`sha256:`) has none.
#[must_use]
pub fn content_id(digest: &Digest) -> Option<ContentId> {
    digest
        .as_str()
        .strip_prefix("blake3:")
        .and_then(|hex| ContentId::parse(hex).ok())
}

/// The digest that addresses the object `id` names.
#[must_use]
pub fn digest_of(id: ContentId) -> Digest {
    match Digest::from_content_hex(&id.to_string()) {
        Ok(digest) => digest,
        Err(_) => unreachable!("a content id displays as 64 lowercase hex digits"),
    }
}
