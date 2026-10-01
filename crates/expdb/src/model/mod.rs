// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The experience graph's vocabulary: what a record is, and the kinds of
//! thing it can say.
//!
//! Raw experience (tasks, states, observations, decisions, transitions,
//! outcomes) is fact. Evaluations, credit, skills and training lineage are
//! interpretations that point at it and carry how they came to be, so they
//! can be replaced without touching what actually happened.

mod body;
mod context;
mod edge;
mod entities;
mod interpret;
mod record;
mod stream;

pub use body::{Body, RecordKind};
pub use context::ContextLog;
pub use edge::{Edge, Rel};
pub use entities::{
    family_key, Action, Attempt, Content, Decision, EpisodeFamily, Observation, Outcome, PolicyRef,
    ReproLevel, State, TaskDefinition, TaskInstance, Transition,
};
pub use interpret::{
    Conclusion, CounterfactualSet, CreditAssignment, DatasetNode, Derivation, Epistemic,
    Evaluation, EvaluatorRef, Experiment, ExperimentResult, ModelNode, Retraction, Skill,
    SkillEvidence, Stance, Target, TrainingRun,
};
pub use record::Record;
pub use stream::{
    ActionKind, ActionSegment, ClockDomain, ClockMapping, Correspondence, CorrespondenceRelation,
    Episode, EpisodeKind, Event, ModalitySchema, SpanRef, Stream, StreamChunk, StreamOrigin,
    StreamSpec, TimeRange, TimeSemantics,
};
