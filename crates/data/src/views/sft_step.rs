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
//! * never a step ATIF marks as copied context.
//!
//! Candidates: an experience, until its actions are reached; then each
//! action (a bad one counts as [`Exclusion::BadStep`]).
//!
//! Packed ([`SftStep::packed`]), the actions an experience admits are one
//! record, not one each.

use splinter_core::annotation::{AnnotationBody, Label, Strength};

use crate::render::student_turn;
use crate::trajectory::conversation;
use crate::{
    require_pass, Corpus, Entry, Exclusion, Objective, Projection, Provenance, Strip, View,
    ViewError,
};

/// The name every sft-step record carries.
const NAME: &str = "sft-step";

/// Supervised fine-tuning on each action of a trajectory.
#[derive(Clone, Debug)]
pub struct SftStep {
    min_strength: Strength,
    strip: Strip,
    packed: bool,
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

    /// The same view under `strip`.
    #[must_use]
    pub fn with_strip(self, strip: Strip) -> Self {
        Self { strip, ..self }
    }

    fn project_entry(&self, entry: &Entry, projection: &mut Projection) {
        let passed = require_pass(entry, self.min_strength);
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
            if !any_good {
                projection.exclude(reason);
                return;
            }
        }
        let turn = match student_turn(entry, &self.strip) {
            Ok(turn) => turn,
            Err(reason) => return projection.exclude(reason),
        };
        let conversation = match conversation(&entry.experience, &turn) {
            Ok(conversation) => conversation,
            Err(reason) => return projection.exclude(reason),
        };
        if conversation.actions.is_empty() {
            return projection.exclude(Exclusion::NoAction);
        }
        let mut admitted = Vec::new();
        for action in conversation.actions.iter().filter(|a| !a.copied) {
            if entry.labelled(action.step, Label::Bad) {
                projection.exclude(Exclusion::BadStep);
            } else if passed.is_ok() || entry.labelled(action.step, Label::Good) {
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
        for entry in corpus.entries() {
            self.project_entry(entry, &mut projection);
        }
        Ok(projection)
    }
}
