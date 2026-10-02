// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Typed links between records, kept apart from the records themselves so a
//! relationship can be added later without rewriting anything.

use serde::{Deserialize, Serialize};

use crate::id::RecordId;

/// What a link means.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rel {
    /// The source was computed from the target.
    DerivedFrom,
    /// The source was extracted from the target.
    ExtractedFrom,
    /// The source was consumed by the target.
    UsedBy,
    /// The source was produced by the target.
    ProducedBy,
    /// The target is an alternative at the source counterfactual set.
    Alternative,
    /// The source was forked from the target decision.
    ForkedFrom,
    /// The source contains the target.
    Contains,
    /// The source experiment generated the target.
    Generated,
    /// The source was judged better than the target.
    PreferredOver,
    /// The source is a new attempt after the failed target.
    RetryOf,
    /// The source is a critique of the target.
    CritiqueOf,
    /// The source is a revision of the target made from a critique.
    RevisionOf,
    /// The source is the target asked in other words.
    VariantOf,
}

impl Rel {
    /// The stable number stored on disk.
    pub fn code(self) -> u8 {
        match self {
            Rel::DerivedFrom => 0,
            Rel::ExtractedFrom => 1,
            Rel::UsedBy => 2,
            Rel::ProducedBy => 3,
            Rel::Alternative => 4,
            Rel::ForkedFrom => 5,
            Rel::Contains => 6,
            Rel::Generated => 7,
            Rel::PreferredOver => 8,
            Rel::RetryOf => 9,
            Rel::CritiqueOf => 10,
            Rel::RevisionOf => 11,
            Rel::VariantOf => 12,
        }
    }

    /// The relation with stored number `code`.
    pub fn from_code(code: u8) -> Option<Self> {
        Some(match code {
            0 => Rel::DerivedFrom,
            1 => Rel::ExtractedFrom,
            2 => Rel::UsedBy,
            3 => Rel::ProducedBy,
            4 => Rel::Alternative,
            5 => Rel::ForkedFrom,
            6 => Rel::Contains,
            7 => Rel::Generated,
            8 => Rel::PreferredOver,
            9 => Rel::RetryOf,
            10 => Rel::CritiqueOf,
            11 => Rel::RevisionOf,
            12 => Rel::VariantOf,
            _ => return None,
        })
    }
}

/// A directed, typed link between two records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Edge {
    /// Where the link starts.
    pub from: RecordId,
    /// What the link means.
    pub rel: Rel,
    /// Where it points.
    pub to: RecordId,
}

impl Edge {
    /// A link from `from` to `to`.
    pub fn new(from: RecordId, rel: Rel, to: RecordId) -> Self {
        Self { from, rel, to }
    }
}
