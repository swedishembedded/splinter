// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Relations: pairs of attempts the experience itself says belong together,
//! such as a retry and the attempt it retried.

use std::collections::HashMap;

use super::plan::{DataRef, Sample, SampleBody};
use super::recipe::Recipe;
use crate::error::{Error, Result};
use crate::id::{ContentId, RecordId};
use crate::manifest::Snapshot;
use crate::model::RecordKind;

impl Snapshot {
    pub(super) fn compile_relations(&self, recipe: &Recipe) -> Result<Vec<Sample>> {
        let rel = recipe.relation.ok_or_else(|| {
            Error::invalid("recipe", "a relation recipe names the relation to follow")
        })?;
        let index = self.index()?;
        let attempts: HashMap<RecordId, ContentId> = self
            .selected_attempts(recipe)?
            .into_iter()
            .map(|a| (a.attempt, a.family))
            .collect();
        let mut samples = Vec::new();
        for from in index.by_kind(RecordKind::Attempt) {
            let Some(family) = attempts.get(&from) else {
                continue;
            };
            for to in index.edges_from(from, Some(rel)) {
                if attempts.contains_key(&to) {
                    samples.push(Sample {
                        provenance: from,
                        family: Some(*family),
                        body: SampleBody::Related {
                            rel,
                            from: DataRef::Episode { attempt: from },
                            to: DataRef::Episode { attempt: to },
                        },
                    });
                }
            }
        }
        Ok(samples)
    }
}
