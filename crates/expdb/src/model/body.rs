// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The body of a record, and the kind that tells readers what it is without
//! parsing it.

use serde::{Deserialize, Serialize};

use super::entities::{
    Attempt, Decision, EpisodeFamily, Observation, Outcome, State, TaskDefinition, TaskInstance,
    Transition,
};
use super::interpret::{
    Conclusion, CounterfactualSet, CreditAssignment, DatasetNode, Derivation, Epistemic,
    Evaluation, Experiment, ExperimentResult, ModelNode, Retraction, Skill, SkillEvidence,
    TrainingRun,
};
use crate::id::RecordId;

macro_rules! kinds {
    ($($variant:ident = $code:expr),+ $(,)?) => {
        /// The kind of a record: a stable number stored in the record block,
        /// so a scan can skip whole blocks without parsing a body.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum RecordKind {
            $(
                #[doc = concat!("A `", stringify!($variant), "` record.")]
                $variant,
            )+
        }

        impl RecordKind {
            /// The stable number stored on disk (below 64, for block masks).
            pub fn code(self) -> u8 {
                match self { $(RecordKind::$variant => $code,)+ }
            }

            /// The kind with stored number `code`.
            pub fn from_code(code: u8) -> Option<Self> {
                match code { $($code => Some(RecordKind::$variant),)+ _ => None }
            }
        }
    };
}

kinds! {
    TaskDefinition = 0, TaskInstance = 1, Family = 2, Attempt = 3, State = 4, Observation = 5,
    Decision = 6, Transition = 7, AttemptEnd = 8, Evaluation = 9, Retraction = 10,
    CounterfactualSet = 11, Skill = 12, SkillEvidence = 13, Credit = 14, Derivation = 15,
    Experiment = 16, ExperimentResult = 17, Dataset = 18, TrainingRun = 19, Model = 20,
}

/// What a record says.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Body {
    /// A task definition.
    TaskDefinition(TaskDefinition),
    /// A task instance.
    TaskInstance(TaskInstance),
    /// An episode family.
    Family(EpisodeFamily),
    /// An attempt at a task instance.
    Attempt(Attempt),
    /// A world state.
    State(State),
    /// What an agent was shown.
    Observation(Observation),
    /// A decision.
    Decision(Decision),
    /// A transition caused by a decision.
    Transition(Transition),
    /// The end of an attempt.
    AttemptEnd {
        /// The attempt that ended.
        attempt: RecordId,
        /// How it ended.
        outcome: Outcome,
    },
    /// An evaluation.
    Evaluation(Evaluation),
    /// A withdrawn evaluator.
    Retraction(Retraction),
    /// A set of alternatives at a decision.
    CounterfactualSet(CounterfactualSet),
    /// A skill.
    Skill(Skill),
    /// Evidence about a skill.
    SkillEvidence(SkillEvidence),
    /// Credit assigned to a decision.
    Credit(CreditAssignment),
    /// How something was derived.
    Derivation(Derivation),
    /// An experiment.
    Experiment(Experiment),
    /// An experiment's result.
    ExperimentResult(ExperimentResult),
    /// A materialised dataset.
    Dataset(DatasetNode),
    /// A training run.
    TrainingRun(TrainingRun),
    /// A model.
    Model(ModelNode),
}

impl Body {
    /// The kind of this body.
    pub fn kind(&self) -> RecordKind {
        match self {
            Body::TaskDefinition(_) => RecordKind::TaskDefinition,
            Body::TaskInstance(_) => RecordKind::TaskInstance,
            Body::Family(_) => RecordKind::Family,
            Body::Attempt(_) => RecordKind::Attempt,
            Body::State(_) => RecordKind::State,
            Body::Observation(_) => RecordKind::Observation,
            Body::Decision(_) => RecordKind::Decision,
            Body::Transition(_) => RecordKind::Transition,
            Body::AttemptEnd { .. } => RecordKind::AttemptEnd,
            Body::Evaluation(_) => RecordKind::Evaluation,
            Body::Retraction(_) => RecordKind::Retraction,
            Body::CounterfactualSet(_) => RecordKind::CounterfactualSet,
            Body::Skill(_) => RecordKind::Skill,
            Body::SkillEvidence(_) => RecordKind::SkillEvidence,
            Body::Credit(_) => RecordKind::Credit,
            Body::Derivation(_) => RecordKind::Derivation,
            Body::Experiment(_) => RecordKind::Experiment,
            Body::ExperimentResult(_) => RecordKind::ExperimentResult,
            Body::Dataset(_) => RecordKind::Dataset,
            Body::TrainingRun(_) => RecordKind::TrainingRun,
            Body::Model(_) => RecordKind::Model,
        }
    }

    /// How far the record may be believed. What happened is fact;
    /// interpretations are not.
    pub fn epistemic(&self) -> Epistemic {
        match self {
            Body::Evaluation(evaluation) => evaluation.epistemic,
            Body::Credit(_)
            | Body::SkillEvidence(_)
            | Body::Derivation(_)
            | Body::ExperimentResult(_) => Epistemic::Derived,
            Body::Skill(_) => Epistemic::Hypothesis,
            _ => Epistemic::Fact,
        }
    }

    /// The content id of a content-addressed entity this body defines: a
    /// task definition, task instance, state or episode family. Other bodies
    /// have none.
    pub fn entity_id(&self) -> crate::error::Result<Option<crate::id::ContentId>> {
        Ok(match self {
            Body::TaskDefinition(d) => Some(d.id()?),
            Body::TaskInstance(i) => Some(i.id()?),
            Body::State(s) => Some(s.id()?),
            Body::Family(f) => Some(super::entities::family_key(
                &f.task_instance,
                &f.initial_state,
            )),
            _ => None,
        })
    }

    /// The conclusion of an experiment result, when this is one.
    pub fn conclusion(&self) -> Option<Conclusion> {
        match self {
            Body::ExperimentResult(result) => Some(result.conclusion),
            _ => None,
        }
    }
}
