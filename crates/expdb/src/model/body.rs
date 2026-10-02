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
    Attempt, Decision, Entity, EpisodeFamily, Observation, Outcome, State, TaskDefinition,
    TaskInstance, Transition,
};
use super::interpret::{
    Conclusion, CounterfactualSet, CreditAssignment, DatasetNode, Derivation, Epistemic,
    Evaluation, Experiment, ExperimentResult, ModelNode, Retraction, Skill, SkillEvidence,
    TrainingRun,
};
use super::stream::{
    ActionSegment, ClockDomain, ClockMapping, Correspondence, Episode, Event, ModalitySchema,
    Stream, StreamChunk, StreamOrigin,
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
    ModalitySchema = 21, ClockDomain = 22, ClockMapping = 23, Episode = 24, Stream = 25, StreamChunk = 26,
    Event = 27, ActionSegment = 28, Correspondence = 29, Entity = 30,
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
    /// A registered kind of sample.
    ModalitySchema(ModalitySchema),
    /// A source of timestamps.
    ClockDomain(ClockDomain),
    /// How two clocks relate.
    ClockMapping(ClockMapping),
    /// A stretch of synchronised experience.
    Episode(Episode),
    /// A stream of samples.
    Stream(Stream),
    /// A piece of a stream.
    StreamChunk(StreamChunk),
    /// Something that happened.
    Event(Event),
    /// An action over an interval.
    ActionSegment(ActionSegment),
    /// A claimed relationship between spans.
    Correspondence(Correspondence),
    /// An application-defined object.
    Entity(Entity),
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
            Body::ModalitySchema(_) => RecordKind::ModalitySchema,
            Body::ClockDomain(_) => RecordKind::ClockDomain,
            Body::ClockMapping(_) => RecordKind::ClockMapping,
            Body::Episode(_) => RecordKind::Episode,
            Body::Stream(_) => RecordKind::Stream,
            Body::StreamChunk(_) => RecordKind::StreamChunk,
            Body::Event(_) => RecordKind::Event,
            Body::ActionSegment(_) => RecordKind::ActionSegment,
            Body::Correspondence(_) => RecordKind::Correspondence,
            Body::Entity(_) => RecordKind::Entity,
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
            Body::Stream(stream) if stream.origin == StreamOrigin::Derived => Epistemic::Derived,
            Body::Event(event) => event.epistemic,
            Body::Correspondence(_) => Epistemic::Annotation,
            _ => Epistemic::Fact,
        }
    }

    /// The content id of a content-addressed entity this body defines: a
    /// task definition, task instance, state, episode family or application
    /// entity. Other bodies have none.
    pub fn entity_id(&self) -> crate::error::Result<Option<crate::id::ContentId>> {
        Ok(match self {
            Body::TaskDefinition(d) => Some(d.id()?),
            Body::TaskInstance(i) => Some(i.id()?),
            Body::State(s) => Some(s.id()?),
            Body::Entity(e) => Some(e.id()?),
            Body::Family(f) => Some(super::entities::family_key(
                &f.task_instance,
                &f.initial_state,
            )),
            _ => None,
        })
    }

    /// Refuses a value JSON cannot hold. The body is stored as JSON, which
    /// writes a NaN or an infinity as `null`: an optional number would read
    /// back as absent and a required one would make the whole segment
    /// unreadable. Better refused at the door with a message than changed.
    pub fn check_finite(&self) -> crate::error::Result<()> {
        let finite = |what: &'static str, value: f64| {
            if value.is_finite() {
                Ok(())
            } else {
                Err(crate::error::Error::invalid(
                    "record body",
                    format!("{what} is {value}, which cannot be stored"),
                ))
            }
        };
        let opt =
            |what: &'static str, value: Option<f64>| value.map_or(Ok(()), |v| finite(what, v));
        match self {
            Body::Decision(d) => {
                opt("old_logprob", d.old_logprob)?;
                opt("value_estimate", d.value_estimate)
            }
            Body::Transition(t) => opt("reward", t.reward),
            Body::Evaluation(e) => {
                finite("score", e.score)?;
                finite("confidence", e.confidence)
            }
            Body::Credit(c) => {
                finite("credit value", c.value)?;
                finite("credit confidence", c.confidence)
            }
            Body::SkillEvidence(e) => {
                opt("effect_size", e.effect_size)?;
                finite("strength", e.strength)
            }
            Body::ExperimentResult(r) => {
                r.metrics.values().try_for_each(|v| finite("a metric", *v))
            }
            Body::ClockMapping(m) => {
                finite("slope", m.slope)?;
                if m.slope <= 0.0 {
                    return Err(crate::error::Error::invalid(
                        "record body",
                        "a clock mapping needs a positive slope",
                    ));
                }
                Ok(())
            }
            Body::Stream(s) => opt("rate_hz", s.rate_hz),
            Body::Event(e) => opt("value", e.value),
            Body::Correspondence(c) => finite("confidence", c.confidence),
            _ => Ok(()),
        }
    }

    /// The conclusion of an experiment result, when this is one.
    pub fn conclusion(&self) -> Option<Conclusion> {
        match self {
            Body::ExperimentResult(result) => Some(result.conclusion),
            _ => None,
        }
    }
}
