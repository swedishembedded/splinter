// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Interpretations of raw experience. None of these is ground truth about
//! what happened: each says who produced it and how, so it can be retracted
//! or recomputed without touching the experience.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::blob::BlobRef;
use crate::id::{ContentId, RecordId};

/// How much a statement should be believed, which decides what a training
/// view may treat as truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Epistemic {
    /// Observed directly: it happened.
    Fact,
    /// Computed from facts by a stated procedure.
    Derived,
    /// Someone's or something's judgement of a fact.
    Annotation,
    /// A conjecture awaiting evidence.
    Hypothesis,
}

/// What an evaluation, credit assignment or evidence points at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "of", content = "id", rename_all = "snake_case")]
pub enum Target {
    /// One record: a decision, a transition, an attempt.
    Record(RecordId),
    /// A content-addressed entity: a task instance, a state, a family.
    Entity(ContentId),
}

/// Who evaluated, and with which version, so a bad evaluator can be
/// withdrawn as a whole.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EvaluatorRef {
    /// The verifier, critic, human or test suite.
    pub name: String,
    /// Its version.
    pub version: String,
}

impl EvaluatorRef {
    /// An evaluator from its name and version.
    pub fn new(name: &str, version: &str) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
        }
    }
}

/// Evidence about how good, bad or useful something was.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Evaluation {
    /// What is being judged.
    pub target: Target,
    /// Who judged it.
    pub evaluator: EvaluatorRef,
    /// What was judged: `task_completion`, `reasoning_quality`, ...
    pub criterion: String,
    /// The score.
    pub score: f64,
    /// How sure the evaluator is, from 0 to 1.
    pub confidence: f64,
    /// The evidence behind the score.
    pub evidence: Option<BlobRef>,
    /// How far the score may be believed.
    pub epistemic: Epistemic,
    /// How the evaluation was computed, when it was derived.
    pub derivation: Option<RecordId>,
}

impl Evaluation {
    /// An evaluation, an annotation by default.
    pub fn new(
        target: Target,
        evaluator: EvaluatorRef,
        criterion: &str,
        score: f64,
        confidence: f64,
    ) -> Self {
        Self {
            target,
            evaluator,
            criterion: criterion.into(),
            score,
            confidence,
            evidence: None,
            epistemic: Epistemic::Annotation,
            derivation: None,
        }
    }

    /// Marks how far the score may be believed.
    pub fn epistemic(mut self, epistemic: Epistemic) -> Self {
        self.epistemic = epistemic;
        self
    }
}

/// Withdraws every evaluation by an evaluator version. The evaluations stay
/// in the database; views stop reading them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Retraction {
    /// The withdrawn evaluator.
    pub evaluator: EvaluatorRef,
    /// Why it was withdrawn.
    pub reason: String,
}

/// A decision point re-run with alternatives. The alternatives are linked by
/// `Alternative` edges, so the set can keep growing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CounterfactualSet {
    /// The world every alternative starts in.
    pub origin_state: ContentId,
    /// The decision whose alternatives these are.
    pub origin_decision: RecordId,
    /// The last record all alternatives share.
    pub shared_prefix: Option<RecordId>,
}

/// Credit one algorithm assigned to one decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreditAssignment {
    /// The decision credited.
    pub target: RecordId,
    /// The algorithm and version that produced the value.
    pub algorithm: String,
    /// The credit.
    pub value: f64,
    /// How sure the algorithm is.
    pub confidence: f64,
    /// The branches the value rests on.
    pub supporting_branches: Vec<RecordId>,
    /// How the value was computed.
    pub derivation: Option<RecordId>,
}

/// A hypothesis about behaviour, supported or contradicted by evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Skill {
    /// A short name.
    pub name: String,
    /// What the skill is.
    pub description: String,
    /// When it applies.
    pub trigger: String,
    /// What to do.
    pub action_pattern: String,
    /// What doing it should achieve.
    pub expected_effect: String,
    /// Skills this one specialises.
    pub parents: Vec<RecordId>,
    /// Skills needed first.
    pub prerequisites: Vec<RecordId>,
}

/// Which way evidence points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stance {
    /// The target shows the skill working.
    Supports,
    /// The target shows the skill failing or being unnecessary.
    Contradicts,
    /// A deliberate test of the skill on new tasks.
    Validates,
}

/// One piece of evidence about a skill.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillEvidence {
    /// The skill the evidence is about.
    pub skill: RecordId,
    /// The record that is the evidence.
    pub target: RecordId,
    /// Which way it points.
    pub stance: Stance,
    /// The domain it was observed in, to count transfer.
    pub domain: String,
    /// The measured effect, when one was measured.
    pub effect_size: Option<f64>,
    /// How strong the verification behind the evidence is, from 0 to 1.
    pub strength: f64,
}

/// How a derived record came to exist: enough to redo it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Derivation {
    /// The algorithm.
    pub algorithm: String,
    /// Its version.
    pub version: String,
    /// The records it read.
    pub inputs: Vec<RecordId>,
    /// The code revision that ran.
    pub code_ref: Option<String>,
    /// The model used, if one was.
    pub model: Option<String>,
    /// The parameters.
    pub params: serde_json::Value,
    /// The random seed.
    pub seed: Option<u64>,
}

/// A hypothesis tested by producing experience.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Experiment {
    /// A short name.
    pub name: String,
    /// The question being asked.
    pub hypothesis: String,
    /// The baseline arm.
    pub control: String,
    /// The arm being tried.
    pub treatment: String,
    /// The task instances used.
    pub tasks: Vec<ContentId>,
}

/// What an experiment found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Conclusion {
    /// The data support the hypothesis.
    Supported,
    /// The data contradict it.
    Refuted,
    /// The data do not decide.
    Inconclusive,
}

/// The measured outcome of an experiment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentResult {
    /// The experiment.
    pub experiment: RecordId,
    /// What was measured, by name.
    pub metrics: BTreeMap<String, f64>,
    /// What was concluded.
    pub conclusion: Conclusion,
}

/// A dataset materialised from a recipe at a snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetNode {
    /// The recipe as it was written.
    pub recipe: serde_json::Value,
    /// The manifest the recipe was run against.
    pub snapshot: ContentId,
    /// How many samples it produced.
    pub samples: u64,
}

/// A training run over a dataset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingRun {
    /// The dataset trained on.
    pub dataset: RecordId,
    /// The training objective.
    pub objective: String,
    /// The model it started from.
    pub base_model: Option<RecordId>,
}

/// A model produced by a training run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelNode {
    /// The model's name.
    pub name: String,
    /// The training run that produced it.
    pub run: Option<RecordId>,
    /// The model it continues.
    pub parent: Option<RecordId>,
}
