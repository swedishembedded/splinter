// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Provenance: how a record came to exist, and everything that came of it.

use crate::error::Result;
use crate::id::RecordId;
use crate::manifest::Snapshot;
use crate::model::{Body, CreditAssignment, Derivation, RecordKind, Rel};

impl Snapshot {
    /// The derivation that produced a record, if it was derived.
    pub fn derivation_of(&self, id: RecordId) -> Result<Option<(RecordId, Derivation)>> {
        let index = self.index()?;
        for derivation_id in index.edges_from(id, Some(Rel::ProducedBy)) {
            if let Some(Body::Derivation(d)) = self.get(derivation_id)?.map(|r| r.body) {
                return Ok(Some((derivation_id, d)));
            }
        }
        Ok(None)
    }

    /// Everything a record rests on, following its links back to sources.
    pub fn lineage_back(&self, id: RecordId) -> Result<Vec<RecordId>> {
        Ok(self.index()?.reachable_from(id, None))
    }

    /// Everything that was made from a record, following links forward.
    pub fn lineage_forward(&self, id: RecordId) -> Result<Vec<RecordId>> {
        Ok(self.index()?.reaching(id, None))
    }

    /// Every record a given algorithm version produced.
    pub fn derived_by(&self, algorithm: &str, version: &str) -> Result<Vec<RecordId>> {
        let index = self.index()?;
        let mut produced = Vec::new();
        for id in index.by_kind(RecordKind::Derivation) {
            if let Some(Body::Derivation(d)) = self.get(id)?.map(|r| r.body) {
                if d.algorithm == algorithm && d.version == version {
                    produced.extend(index.edges_to(id, Some(Rel::ProducedBy)));
                }
            }
        }
        produced.sort_unstable();
        produced.dedup();
        Ok(produced)
    }

    /// Every credit any algorithm assigned to a decision. They stay side by
    /// side: a better algorithm adds an opinion, it replaces none.
    pub fn credits_for(&self, decision: RecordId) -> Result<Vec<CreditAssignment>> {
        let index = self.index()?;
        let mut credits = Vec::new();
        for id in index.edges_to(decision, Some(Rel::DerivedFrom)) {
            if let Some(Body::Credit(credit)) = self.get(id)?.map(|r| r.body) {
                credits.push(credit);
            }
        }
        Ok(credits)
    }
}
