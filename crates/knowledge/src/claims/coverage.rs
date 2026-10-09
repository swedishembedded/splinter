// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements admission gates that keep ungrounded
// claims out of training data, for its clients. If your team needs
// expertise in synthetic data quality or provenance, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Whether the claims extracted from a session cover what the session is
//! known to be about.
//!
//! A recording of a session can say which fact it was made to teach
//! ([`crate::session::SessionView::subject`]); a [`Subject`] says what
//! that fact looks like in a statement: the terms every statement of it
//! carries. A claim *covers* a subject when its statement carries all of them.
//! Whether the claims of a night cover the subjects of its sessions is the
//! extraction's recall (a fact the extractor missed is never learned); the
//! share of admitted claims that cover their session's subject is its
//! precision.

use serde::Deserialize;

use crate::gates::normalize;

/// A known fact: its id and the terms a statement of it carries.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Subject {
    /// The id a recording names it by.
    pub id: String,
    /// Terms a statement of the fact carries, compared case-insensitively
    /// with spacing normalised.
    pub terms: Vec<String>,
}

impl Subject {
    /// Whether `statement` carries every term of the subject; a subject with
    /// no terms is covered by nothing.
    #[must_use]
    pub fn covered_by(&self, statement: &str) -> bool {
        let statement = normalize(statement);
        !self.terms.is_empty() && self.terms.iter().all(|t| statement.contains(&normalize(t)))
    }
}
