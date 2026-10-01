// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Counterfactual sets: what was tried instead at one decision.

use std::collections::BTreeMap;

use crate::error::Result;
use crate::id::{ContentId, RecordId};
use crate::manifest::Snapshot;
use crate::model::{Body, RecordKind, Rel};

/// Every alternative recorded at one decision, however many forks made them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CounterfactualView {
    /// The decision the alternatives replace.
    pub origin_decision: RecordId,
    /// The world they all start in.
    pub origin_state: ContentId,
    /// The last record they all share.
    pub shared_prefix: Option<RecordId>,
    /// The set records, one per fork.
    pub sets: Vec<RecordId>,
    /// The origin decision and every alternative to it, in id order.
    pub alternatives: Vec<RecordId>,
}

impl Snapshot {
    /// The counterfactual sets, grouped by the decision they branch at.
    pub fn counterfactual_sets(&self) -> Result<Vec<CounterfactualView>> {
        let index = self.index()?;
        let mut by_origin: BTreeMap<RecordId, CounterfactualView> = BTreeMap::new();
        for id in index.by_kind(RecordKind::CounterfactualSet) {
            let Some(Body::CounterfactualSet(set)) = self.get(id)?.map(|r| r.body) else {
                continue;
            };
            let view = by_origin
                .entry(set.origin_decision)
                .or_insert_with(|| CounterfactualView {
                    origin_decision: set.origin_decision,
                    origin_state: set.origin_state,
                    shared_prefix: set.shared_prefix,
                    sets: Vec::new(),
                    alternatives: Vec::new(),
                });
            view.sets.push(id);
            view.alternatives
                .extend(index.edges_from(id, Some(Rel::Alternative)));
        }
        Ok(by_origin
            .into_values()
            .map(|mut view| {
                view.alternatives.sort_unstable();
                view.alternatives.dedup();
                view
            })
            .collect())
    }
}
