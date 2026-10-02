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
    /// When it was stamped.
    pub timestamp_ns: u64,
}

/// Every evaluation of a snapshot and the evaluators withdrawn from it.
#[derive(Debug, Default)]
pub struct EvalSet {
    views: Vec<EvaluationView>,
    by_target: HashMap<Target, Vec<usize>>,
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

    /// Whether `e` passes every condition but the withdrawal of its
    /// evaluator, which the caller knows and passes as `retracted`.
    pub(crate) fn accepts(&self, e: &Evaluation, retracted: bool) -> bool {
        (self.include_retracted || !retracted)
            && self.target.is_none_or(|t| e.target == t)
            && self.criterion.as_ref().is_none_or(|c| e.criterion == *c)
            && self
                .evaluator
                .as_ref()
                .is_none_or(|n| e.evaluator.name == *n)
            && self.epistemic.is_none_or(|x| e.epistemic == x)
            && self.min_confidence.is_none_or(|c| e.confidence >= c)
    }

    pub(crate) fn wanted_target(&self) -> Option<Target> {
        self.target
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
                let timestamp_ns = index.timestamp_ns(id).unwrap_or(0);
                set.views.push(EvaluationView {
                    id,
                    evaluation,
                    timestamp_ns,
                });
            }
        }
        // In the order they were stamped, which every reader and every run
        // agrees on, and not in the order of the random ids of the writers.
        set.views.sort_by_key(|v| (v.timestamp_ns, v.id));
        for (position, view) in set.views.iter().enumerate() {
            set.by_target
                .entry(view.evaluation.target)
                .or_default()
                .push(position);
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
        let pick = |v: &EvaluationView| {
            filter
                .accepts(
                    &v.evaluation,
                    set.retracted.contains(&v.evaluation.evaluator),
                )
                .then(|| v.clone())
        };
        Ok(match filter.wanted_target() {
            Some(target) => set
                .by_target
                .get(&target)
                .into_iter()
                .flatten()
                .filter_map(|&i| pick(&set.views[i]))
                .collect(),
            None => set.views.iter().filter_map(pick).collect(),
        })
    }

    /// Whether every judgement by `evaluator` has been withdrawn.
    pub fn is_retracted(&self, evaluator: &EvaluatorRef) -> Result<bool> {
        Ok(self.eval_set()?.retracted.contains(evaluator))
    }

    /// The reward each attempt with standing task-completion evaluations
    /// stands at. Where any of them carries a ranked verdict, the strongest
    /// evidence decides, one for a pass and zero for a fail, and an attempt
    /// the evidence leaves undecided has `None`: unmeasured, not zero.
    /// Otherwise it is the mean of the scores, weighted by confidence.
    pub(crate) fn attempt_rewards(&self) -> Result<HashMap<RecordId, Option<f64>>> {
        struct Sums {
            weighted: f64,
            weight: f64,
            n: usize,
            verdicts: Vec<Verdict>,
        }
        let mut sums: HashMap<RecordId, Sums> = HashMap::new();
        for view in self.evaluations(&EvalFilter::new().criterion(TASK_COMPLETION))? {
            if let Target::Record(attempt) = view.evaluation.target {
                let entry = sums.entry(attempt).or_insert(Sums {
                    weighted: 0.0,
                    weight: 0.0,
                    n: 0,
                    verdicts: Vec::new(),
                });
                entry.weighted += view.evaluation.score * view.evaluation.confidence;
                entry.weight += view.evaluation.confidence;
                entry.n += 1;
                entry.verdicts.extend(view.evaluation.verdict);
            }
        }
        Ok(sums
            .into_iter()
            .map(|(attempt, s)| {
                let reward = if s.verdicts.is_empty() {
                    Some(if s.weight > 0.0 {
                        s.weighted / s.weight
                    } else {
                        s.weighted / s.n as f64
                    })
                } else {
                    resolve_verdicts(&s.verdicts).map(|r| if r.passed { 1.0 } else { 0.0 })
                };
                (attempt, reward)
            })
            .collect())
    }
}
