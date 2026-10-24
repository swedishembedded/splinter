// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The sft-step view: every action of a trajectory, supervised in the
//! context the solver had when it took it.
//!
//! The trajectory is rendered as `crate::trajectory` describes: the
//! student's turn (see [`Strip`]) in place of the solver's prompt, then
//! each agent step as an assistant message with its tool calls and the
//! observations that answered them. Each action - an agent step with tool
//! calls, or the final answer - yields one record: the conversation up to
//! it, with only the action supervised.
//!
//! In a workspace the agent is told how its work did (the visible checks'
//! output) and carries on: those later user turns are context in every
//! record after them, never supervised. A reply that ended a turn and was
//! answered by such feedback was premature, so it is context too and not an
//! action. The record's tools are the schemas the agent was offered, and its
//! system turn the prompt the harness pinned when the trajectory records one.
//! A trajectory is left out, counted by reason, when its context was
//! compacted, a tool call's arguments were not JSON, or an assistant turn
//! holds reasoning markup.
//!
//! Which actions:
//!
//! * of an experience decided pass at the minimum strength: every action
//!   except those whose step is labelled [`Label::Bad`];
//! * of any other experience (failed, too weakly decided, undecided): only
//!   actions whose step is labelled [`Label::Good`] and not also bad. A
//!   step label judges that step alone, and a failed attempt can hold
//!   correct steps; with no good label it yields nothing and is counted
//!   under why it did not pass;
//! * never a step ATIF marks as copied context;
//! * never an action that repeats a call of the trajectory whose result was
//!   the same ([`Exclusion::RepeatedAction`]): the tail of a loop stays in the
//!   context of what follows and is not learned.
//!
//! Candidates: an experience, until its actions are reached; then each
//! action (a bad one counts as [`Exclusion::BadStep`]).
//!
//! Packed ([`SftStep::packed`]), the actions an experience admits are one
//! record, not one each.

use splinter_core::annotation::{AnnotationBody, Label, Outcome, Strength};
use splinter_core::chat::WireMessage;
use splinter_core::rng::SplitMix64;

use crate::render::student_turn;
use crate::trajectory::conversation_serialised;
use crate::{
    require_pass, Corpus, Entry, Exclusion, Objective, Projection, Provenance, RecordBody, Strip,
    View, ViewError,
};

/// The name every sft-step record carries.
const NAME: &str = "sft-step";

/// How the view admits what it projects: normally only verified work; the
/// negative controls of the coding experiments change one thing each.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Control {
    /// Only experiences decided pass at the minimum strength.
    #[default]
    Verified,
    /// Only experiences decided pass at the minimum strength, and no action
    /// is admitted on its own label from one that was not: outcome-verified
    /// evidence. [`Control::Verified`], the default, also admits an action
    /// labelled good from an experience nobody verified or whose verdict
    /// failed (step-level credit, only as trustworthy as whoever labelled).
    OutcomeVerified,
    /// Trajectories whatever their verdict (a failed one, or none), except
    /// those an exposure audit failed: it tests whether verification matters.
    Unverified,
    /// The verified trajectories, each under the instruction of another
    /// task, paired by a seeded permutation with no fixed point: it tests
    /// that the gain comes from the pairing and not from the format.
    ShuffledInstructions {
        /// Seeds the permutation.
        seed: u64,
    },
}

/// Supervised fine-tuning on each action of a trajectory.
#[derive(Clone, Debug)]
pub struct SftStep {
    min_strength: Strength,
    strip: Strip,
    packed: bool,
    control: Control,
    serialise_calls: bool,
    only_actions: Option<Vec<String>>,
}

impl SftStep {
    /// The view, supervising every action of an experience decided pass at
    /// `min_strength` or stronger, showing the student only the
    /// instruction.
    #[must_use]
    pub fn new(min_strength: Strength) -> Self {
        Self {
            min_strength,
            strip: Strip::default(),
            packed: false,
            control: Control::Verified,
            serialise_calls: false,
            only_actions: None,
        }
    }

    /// The same view writing one record per trajectory, instead of one per
    /// action: the conversation up to its last admitted action with every
    /// admitted action supervised and every other message context. It
    /// supervises exactly what the per-action records supervise together,
    /// with the context once instead of once per action.
    ///
    /// A training step counts records, so a step of packed records spans
    /// many more tokens than one of per-action records; within a step each
    /// supervised token weighs the same in both forms.
    #[must_use]
    pub fn packed(self, packed: bool) -> Self {
        Self { packed, ..self }
    }

    /// The same view reading a run exported as one agent step as one call per
    /// turn ([`crate::trajectory`]): for trajectories recorded before the agent
    /// kept its turns apart.
    #[must_use]
    pub fn serialise_calls(self, serialise_calls: bool) -> Self {
        Self {
            serialise_calls,
            ..self
        }
    }

    /// The same view keeping only the actions whose tool calls are all named in `names`; an
    /// action that calls no tool (the final answer) is always kept. The rest are context and
    /// are counted as [`Exclusion::ActionNotWanted`]. This is how a corpus is made of the
    /// decisive actions - the edits and the conclusion - instead of the exploration around
    /// them, which is most of the tokens and teaches the least.
    #[must_use]
    pub fn only_actions(self, names: Vec<String>) -> Self {
        // An empty list is no request at all: it cannot mean "keep nothing".
        Self { only_actions: (!names.is_empty()).then_some(names), ..self }
    }

    /// The same view under a negative `control`.
    #[must_use]
    pub fn control(self, control: Control) -> Self {
        Self { control, ..self }
    }

    /// The same view under `strip`.
    #[must_use]
    pub fn with_strip(self, strip: Strip) -> Self {
        Self { strip, ..self }
    }

    fn project_entry(&self, entry: &Entry, projection: &mut Projection) {
        let outcome_verified = self.control == Control::OutcomeVerified;
        let passed = match self.control {
            Control::Unverified => admit_unverified(entry),
            _ => require_pass(entry, self.min_strength),
        };
        let any_good = entry.notes.iter().any(|n| {
            matches!(
                n.body,
                AnnotationBody::StepLabel {
                    label: Label::Good,
                    ..
                }
            )
        });
        if let Err(reason) = passed {
            if !any_good || outcome_verified {
                projection.exclude(reason);
                return;
            }
        }
        let turn = match student_turn(entry, &self.strip) {
            Ok(turn) => turn,
            Err(reason) => return projection.exclude(reason),
        };
        let conversation =
            match conversation_serialised(&entry.experience, &turn, self.serialise_calls) {
                Ok(conversation) => conversation,
                Err(reason) => return projection.exclude(reason),
            };
        if conversation.actions.is_empty() {
            return projection.exclude(Exclusion::NoAction);
        }
        let mut admitted = Vec::new();
        let wanted = |action: &crate::trajectory::Action| match &self.only_actions {
            None => true,
            Some(names) => {
                let calls = &conversation.messages[action.index].tool_calls;
                calls.is_empty() || calls.iter().all(|c| names.iter().any(|n| n == &c.function.name))
            }
        };
        for action in conversation.actions.iter().filter(|a| !a.copied) {
            if entry.labelled(action.step, Label::Bad) {
                projection.exclude(Exclusion::BadStep);
            } else if action.repeated {
                projection.exclude(Exclusion::RepeatedAction);
            } else if !wanted(action) {
                projection.exclude(Exclusion::ActionNotWanted);
            } else if passed.is_ok()
                || (!outcome_verified && entry.labelled(action.step, Label::Good))
            {
                admitted.push(action);
            }
        }
        let provenance = || Provenance::of(vec![entry.id.clone()]);
        if self.packed {
            if let Some(record) = conversation.packed_record(&admitted) {
                projection.push(record, provenance());
            }
        } else {
            for action in admitted {
                projection.push(conversation.record(action), provenance());
            }
        }
    }
}

impl View for SftStep {
    fn name(&self) -> &str {
        NAME
    }

    fn objective(&self) -> Objective {
        Objective::Sft
    }

    fn project(&self, corpus: &Corpus) -> Result<Projection, ViewError> {
        let mut projection = Projection::new(
            NAME,
            Objective::Sft,
            Some(self.strip.clone()),
            Some(self.min_strength),
        );
        projection.packed = self.packed;
        projection.outcome_verified_only = self.control == Control::OutcomeVerified;
        for entry in corpus.entries() {
            self.project_entry(entry, &mut projection);
        }
        if let Control::ShuffledInstructions { seed } = self.control {
            shuffle_instructions(&mut projection, seed)?;
        }
        Ok(projection)
    }
}

/// Admits `entry` whatever its verdict unless an exposure audit failed it: a
/// failed constraint means the trajectory touched material it must not.
fn admit_unverified(entry: &Entry) -> Result<(), Exclusion> {
    let exposed = entry.notes.iter().any(|n| {
        matches!(
            n.body,
            AnnotationBody::Verdict {
                outcome: Outcome::Fail,
                strength: Strength::Constraint,
                ..
            }
        )
    });
    if exposed {
        Err(Exclusion::Failed)
    } else {
        Ok(())
    }
}

/// The instruction of a record: its first user message.
fn first_user_message(body: &mut RecordBody) -> Option<&mut WireMessage> {
    match body {
        RecordBody::Chat { messages, .. } => messages.iter_mut().find(|m| m.role == "user"),
        _ => None,
    }
}

/// Replaces the instruction (the first user message) of every record with
/// another record's, by a seeded permutation of the distinct instructions
/// with no fixed point. Records of one instruction keep one donor.
fn shuffle_instructions(projection: &mut Projection, seed: u64) -> Result<(), ViewError> {
    let mut distinct: Vec<String> = Vec::new();
    for record in &mut projection.records {
        if let Some(text) = first_user_message(&mut record.body).map(|m| m.content.clone()) {
            if !distinct.contains(&text) {
                distinct.push(text);
            }
        }
    }
    if distinct.len() < 2 {
        return Err(ViewError::Parameter {
            name: "control",
            reason: "shuffling instructions needs at least two distinct instructions".into(),
        });
    }
    let mut order = distinct.clone();
    SplitMix64::new(seed).shuffle(&mut order);
    let donor = |own: &str| {
        let at = order.iter().position(|t| t == own)?;
        order.get((at + 1) % order.len()).cloned()
    };
    for record in &mut projection.records {
        if let Some(message) = first_user_message(&mut record.body) {
            if let Some(other) = donor(&message.content) {
                message.content = other;
            }
        }
    }
    Ok(())
}
