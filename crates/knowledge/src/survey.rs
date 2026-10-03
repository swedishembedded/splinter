// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that survey a body of writing and
// plan what to learn from it, for its clients. If your team needs expertise
// in turning an unfamiliar corpus into a learning plan, you can procure our
// services by sending an email to info@swedishembedded.com.

//! A survey of sources: what a set of captured sources holds, found by code.
//!
//! A planner model that chooses what to learn from a body of writing has to
//! be told what is in it. A survey is that account: counts, the share that
//! reads as advice, and a few excerpts and names. It is the same for the same
//! sources, costs one pass over the text, and stays small however large the
//! sources are, so it fits in a prompt.

use serde::Serialize;
use splinter_record::error::StoreError;
use splinter_record::source::SourceId;
use splinter_record::sources::SourceStore;

use crate::advice::{reads_as_advice, reads_as_judgment};
use crate::sections::sections;

/// The most excerpts a survey carries.
pub const MAX_EXCERPTS: usize = 6;

/// The most part names a survey carries.
pub const MAX_NAMES: usize = 8;

/// The most characters of one excerpt.
const EXCERPT_CHARS: usize = 300;

/// What a set of sources holds.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Survey {
    /// Text parts across the sources.
    pub parts: usize,
    /// Bytes of text in them.
    pub text_bytes: u64,
    /// Sections across the parts.
    pub sections: usize,
    /// Of those, the sections that read as advice.
    pub advice_sections: usize,
    /// Of those, the sections in which the writer judges: advises or states a
    /// position (advice included).
    pub judgment_sections: usize,
    /// The first few part names, in order.
    pub names: Vec<String>,
    /// A few short excerpts, advice first.
    pub excerpts: Vec<String>,
}

/// The first `EXCERPT_CHARS` characters of `text`, on one line.
fn excerpt(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(EXCERPT_CHARS)
        .collect()
}

/// A survey of the text parts of `sources`.
///
/// # Errors
/// A source or part that cannot be read from the store.
pub fn survey(store: &SourceStore, sources: &[SourceId]) -> Result<Survey, StoreError> {
    let mut found = Survey::default();
    let mut advice = Vec::new();
    let mut other = Vec::new();
    for id in sources {
        let source = store.get_source(id)?;
        for part in source
            .parts
            .iter()
            .filter(|p| p.media_type.starts_with("text/"))
        {
            let bytes = store.read_blob(&part.content)?;
            let text = String::from_utf8_lossy(&bytes);
            found.parts += 1;
            found.text_bytes += bytes.len() as u64;
            if found.names.len() < MAX_NAMES {
                found.names.push(part.name.clone());
            }
            for section in sections(&text, &part.media_type) {
                let Some(body) = section.text(&text) else {
                    continue;
                };
                found.sections += 1;
                found.judgment_sections += usize::from(reads_as_judgment(body));
                if reads_as_advice(body) {
                    found.advice_sections += 1;
                    if advice.len() < MAX_EXCERPTS {
                        advice.push(excerpt(body));
                    }
                } else if other.len() < MAX_EXCERPTS && body.split_whitespace().count() >= 12 {
                    other.push(excerpt(body));
                }
            }
        }
    }
    found.excerpts = advice.into_iter().chain(other).take(MAX_EXCERPTS).collect();
    Ok(found)
}
