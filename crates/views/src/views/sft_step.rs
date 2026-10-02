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

use splinter_record::annotation::{AnnotationBody, Label, Strength};

use crate::render::student_turn;
use crate::trajectory::conversation;
use crate::{
    require_pass, Corpus, Entry, Exclusion, Objective, Projection, Provenance, RecordBody, Strip,
    View, ViewError,
};

/// The name every sft-step record carries.
const NAME: &str = "sft-step";

/// Supervised fine-tuning on each action of a trajectory.
#[derive(Clone, Debug)]
pub struct SftStep {
    min_strength: Strength,
    strip: Strip,
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
        }
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
        let Some(conversation) = conversation(&entry.experience.trajectory, &turn) else {
            return projection.exclude(Exclusion::Unrepresentable);
        };
        if conversation.actions.is_empty() {
            return projection.exclude(Exclusion::NoAction);
        }
        for action in conversation.actions.iter().filter(|a| !a.copied) {
            if entry.labelled(action.step, Label::Bad) {
                projection.exclude(Exclusion::BadStep);
            } else if passed.is_ok() || entry.labelled(action.step, Label::Good) {
                projection.push(
                    RecordBody::Chat {
                        messages: conversation.up_to(action),
                    },
                    Provenance::of(vec![entry.id.clone()]),
                );
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
        for entry in corpus.entries() {
            self.project_entry(entry, &mut projection);
        }
        Ok(projection)
    }
}
