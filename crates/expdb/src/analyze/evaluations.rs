// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Evaluations as records, and the retractions that withdraw them.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::error::Result;
use crate::id::RecordId;
use crate::manifest::Snapshot;
use crate::model::{
    Body, Epistemic, Evaluation, EvaluatorRef, RecordKind, Ruling, Target, Verdict,
};

/// The criterion that decides an attempt's reward.
pub const TASK_COMPLETION: &str = "task_completion";

/// An evaluation and the record it is stored in.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationView {
    /// The evaluation record.
    pub id: RecordId,
    /// What it says.
    pub evaluation: Evaluation,
}

/// Every evaluation of a snapshot and the evaluators withdrawn from it.
#[derive(Debug, Default)]
pub struct EvalSet {
    views: Vec<EvaluationView>,
    retracted: HashSet<EvaluatorRef>,
}

/// Which evaluations are wanted. Withdrawn evaluators are excluded unless
/// asked for.
#[derive(Debug, Clone, Default)]
pub struct EvalFilter {
    target: Option<Target>,
    criterion: Option<String>,
    evaluator: Option<String>,
    epistemic: Option<Epistemic>,
    min_confidence: Option<f64>,
    include_retracted: bool,
}

impl EvalFilter {
    /// Every standing evaluation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Only evaluations of this target.
    pub fn target(mut self, target: Target) -> Self {
        self.target = Some(target);
        self
    }

    /// Only evaluations of this criterion.
    pub fn criterion(mut self, criterion: &str) -> Self {
        self.criterion = Some(criterion.into());
        self
    }

    /// Only evaluations by this evaluator, any version.
    pub fn evaluator(mut self, name: &str) -> Self {
        self.evaluator = Some(name.into());
        self
    }

    /// Only evaluations of this epistemic class.
    pub fn epistemic(mut self, epistemic: Epistemic) -> Self {
        self.epistemic = Some(epistemic);
        self
    }

    /// Only evaluations at least this confident.
    pub fn min_confidence(mut self, confidence: f64) -> Self {
        self.min_confidence = Some(confidence);
        self
    }

    /// Include evaluations by withdrawn evaluators, for audit.
    pub fn include_retracted(mut self) -> Self {
        self.include_retracted = true;
        self
    }
}

/// What a target's verdicts add up to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolution {
    /// Whether the deciding evidence says the target passed.
    pub passed: bool,
    /// The rank of the deciding evidence.
    pub rank: u8,
}

/// What a set of verdicts decides: abstentions are ignored, only the highest
/// rank present is read, and it decides when every verdict at that rank
/// agrees. No verdict, only abstentions, or disagreement at the top rank
/// decide nothing: the answer is absent, never a zero.
pub fn resolve_verdicts(verdicts: &[Verdict]) -> Option<Resolution> {
    let top = verdicts
        .iter()
        .filter(|v| v.ruling != Ruling::Abstain)
        .map(|v| v.rank)
        .max()?;
    let mut at_top = verdicts
        .iter()
        .filter(|v| v.rank == top && v.ruling != Ruling::Abstain)
        .map(|v| v.ruling == Ruling::Pass);
    let first = at_top.next()?;
    at_top.all(|p| p == first).then_some(Resolution {
        passed: first,
        rank: top,
    })
}

impl Snapshot {
    /// What the standing verdicts of `criterion` on `target` decide.
    pub fn resolution(&self, target: &Target, criterion: &str) -> Result<Option<Resolution>> {
        let verdicts: Vec<Verdict> = self
            .evaluations(&EvalFilter::new().target(*target).criterion(criterion))?
            .iter()
            .filter_map(|v| v.evaluation.verdict)
            .collect();
        Ok(resolve_verdicts(&verdicts))
    }

    /// What the standing verdicts of `criterion` decide for every target that
    /// has a decision, in one pass.
    pub fn resolutions(&self, criterion: &str) -> Result<HashMap<Target, Resolution>> {
        let mut grouped: HashMap<Target, Vec<Verdict>> = HashMap::new();
        for view in self.evaluations(&EvalFilter::new().criterion(criterion))? {
            if let Some(verdict) = view.evaluation.verdict {
                grouped
                    .entry(view.evaluation.target)
                    .or_default()
                    .push(verdict);
            }
        }
        Ok(grouped
            .into_iter()
            .filter_map(|(target, verdicts)| resolve_verdicts(&verdicts).map(|r| (target, r)))
            .collect())
    }

    fn eval_set(&self) -> Result<Arc<EvalSet>> {
        let cache = self.cache();
        if let Some(found) = cache
            .evaluations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
        {
            return Ok(Arc::clone(found));
        }
        let index = self.index()?;
        let mut set = EvalSet::default();
        for id in index.by_kind(RecordKind::Evaluation) {
            if let Some(Body::Evaluation(evaluation)) = self.get(id)?.map(|r| r.body) {
                set.views.push(EvaluationView { id, evaluation });
            }
        }
        for id in index.by_kind(RecordKind::Retraction) {
            if let Some(Body::Retraction(retraction)) = self.get(id)?.map(|r| r.body) {
                set.retracted.insert(retraction.evaluator);
            }
        }
        let set = Arc::new(set);
        *cache
            .evaluations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Arc::clone(&set));
        Ok(set)
    }

    /// The evaluations matching a filter, in record order. A withdrawal
    /// covers an evaluator version's judgements whenever they were written.
    pub fn evaluations(&self, filter: &EvalFilter) -> Result<Vec<EvaluationView>> {
        let set = self.eval_set()?;
        Ok(set
            .views
            .iter()
            .filter(|v| {
                let e = &v.evaluation;
                (filter.include_retracted || !set.retracted.contains(&e.evaluator))
                    && filter.target.is_none_or(|t| e.target == t)
                    && filter.criterion.as_ref().is_none_or(|c| e.criterion == *c)
                    && filter
                        .evaluator
                        .as_ref()
                        .is_none_or(|n| e.evaluator.name == *n)
                    && filter.epistemic.is_none_or(|x| e.epistemic == x)
                    && filter.min_confidence.is_none_or(|c| e.confidence >= c)
            })
            .cloned()
            .collect())
    }

    /// The reward each attempt stands at according to its standing
    /// task-completion evaluations, weighted by confidence.
    pub(crate) fn attempt_rewards(&self) -> Result<HashMap<RecordId, f64>> {
        let mut sums: HashMap<RecordId, (f64, f64, usize)> = HashMap::new();
        for view in self.evaluations(&EvalFilter::new().criterion(TASK_COMPLETION))? {
            if let Target::Record(attempt) = view.evaluation.target {
                let entry = sums.entry(attempt).or_default();
                entry.0 += view.evaluation.score * view.evaluation.confidence;
                entry.1 += view.evaluation.confidence;
                entry.2 += 1;
            }
        }
        Ok(sums
            .into_iter()
            .map(|(attempt, (weighted, weight, n))| {
                let reward = if weight > 0.0 {
                    weighted / weight
                } else {
                    weighted / n as f64
                };
                (attempt, reward)
            })
            .collect())
    }
}
