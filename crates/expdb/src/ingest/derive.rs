// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Recording interpretations and training lineage. Every link points from
//! what was made to what it was made from, so lineage reads the same way
//! from a model back to a task as from an evaluation back to an attempt.

use super::collector::Collector;
use crate::error::Result;
use crate::id::RecordId;
use crate::model::{
    Body, CreditAssignment, DatasetNode, Derivation, Edge, EvaluatorRef, Experiment,
    ExperimentResult, ModelNode, Rel, Retraction, TrainingRun,
};

impl Collector {
    /// Records a derivation and the records it produced. Each output points
    /// to the derivation that made it and to every input the derivation read.
    pub fn derived(
        &mut self,
        derivation: Derivation,
        outputs: Vec<Body>,
    ) -> Result<(RecordId, Vec<RecordId>)> {
        let inputs = derivation.inputs.clone();
        let derivation_id = self.record(Body::Derivation(derivation))?;
        let mut ids = Vec::with_capacity(outputs.len());
        for body in outputs {
            let id = self.record(body)?;
            self.writer
                .link(Edge::new(id, Rel::ProducedBy, derivation_id))?;
            for input in &inputs {
                self.writer.link(Edge::new(id, Rel::DerivedFrom, *input))?;
            }
            ids.push(id);
        }
        Ok((derivation_id, ids))
    }

    /// Records a dataset made from records, usually skills or experiences.
    pub fn record_dataset(&mut self, dataset: DatasetNode, from: &[RecordId]) -> Result<RecordId> {
        let id = self.record(Body::Dataset(dataset))?;
        for source in from {
            self.writer.link(Edge::new(id, Rel::DerivedFrom, *source))?;
        }
        Ok(id)
    }

    /// Records a training run over a dataset, from a base model if it had one.
    pub fn record_training_run(&mut self, run: TrainingRun) -> Result<RecordId> {
        let (dataset, base) = (run.dataset, run.base_model);
        let id = self.record(Body::TrainingRun(run))?;
        self.writer.link(Edge::new(id, Rel::DerivedFrom, dataset))?;
        if let Some(base) = base {
            self.writer.link(Edge::new(id, Rel::DerivedFrom, base))?;
        }
        Ok(id)
    }

    /// Records a model, produced by a run and continuing a parent model.
    pub fn record_model(&mut self, model: ModelNode) -> Result<RecordId> {
        let (run, parent) = (model.run, model.parent);
        let id = self.record(Body::Model(model))?;
        if let Some(run) = run {
            self.writer.link(Edge::new(id, Rel::ProducedBy, run))?;
        }
        if let Some(parent) = parent {
            self.writer.link(Edge::new(id, Rel::DerivedFrom, parent))?;
        }
        Ok(id)
    }

    /// Records one algorithm's credit for a decision. Other algorithms'
    /// credits for the same decision stand beside it.
    pub fn assign_credit(&mut self, credit: CreditAssignment) -> Result<RecordId> {
        let target = credit.target;
        let id = self.record(Body::Credit(credit))?;
        self.writer.link(Edge::new(id, Rel::DerivedFrom, target))?;
        Ok(id)
    }

    /// Withdraws an evaluator version: its evaluations stop counting in
    /// views, and stay stored.
    pub fn retract(&mut self, evaluator: EvaluatorRef, reason: &str) -> Result<RecordId> {
        self.record(Body::Retraction(Retraction {
            evaluator,
            reason: reason.into(),
        }))
    }

    /// Records the question an experience will answer.
    pub fn start_experiment(&mut self, experiment: Experiment) -> Result<RecordId> {
        self.record(Body::Experiment(experiment))
    }

    /// Records what an experiment found.
    pub fn record_result(&mut self, result: ExperimentResult) -> Result<RecordId> {
        let experiment = result.experiment;
        let id = self.record(Body::ExperimentResult(result))?;
        self.writer
            .link(Edge::new(id, Rel::DerivedFrom, experiment))?;
        Ok(id)
    }
}
