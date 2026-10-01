// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Experiments: the question an experience was produced to answer.

use crate::error::Result;
use crate::id::RecordId;
use crate::manifest::Snapshot;
use crate::model::{Body, Conclusion, Experiment, ExperimentResult, RecordKind, Rel};

/// An experiment, what it found and what it produced.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentView {
    /// The experiment's record id.
    pub id: RecordId,
    /// The question.
    pub experiment: Experiment,
    /// What was measured and concluded.
    pub results: Vec<ExperimentResult>,
    /// The records the experiment generated.
    pub generated: Vec<RecordId>,
}

impl Snapshot {
    fn experiment_view(&self, id: RecordId, experiment: Experiment) -> Result<ExperimentView> {
        let index = self.index()?;
        let mut results = Vec::new();
        for result_id in index.edges_to(id, Some(Rel::DerivedFrom)) {
            if let Some(Body::ExperimentResult(r)) = self.get(result_id)?.map(|r| r.body) {
                results.push(r);
            }
        }
        Ok(ExperimentView {
            id,
            experiment,
            results,
            generated: index.edges_from(id, Some(Rel::Generated)),
        })
    }

    /// One experiment with its results and what it generated.
    pub fn experiment(&self, id: RecordId) -> Result<Option<ExperimentView>> {
        match self.get(id)?.map(|r| r.body) {
            Some(Body::Experiment(experiment)) => Ok(Some(self.experiment_view(id, experiment)?)),
            _ => Ok(None),
        }
    }

    /// Experiments that have a result with this conclusion.
    pub fn experiments_concluding(&self, conclusion: Conclusion) -> Result<Vec<ExperimentView>> {
        let mut found = Vec::new();
        for id in self.index()?.by_kind(RecordKind::Experiment) {
            if let Some(view) = self.experiment(id)? {
                if view.results.iter().any(|r| r.conclusion == conclusion) {
                    found.push(view);
                }
            }
        }
        Ok(found)
    }
}
