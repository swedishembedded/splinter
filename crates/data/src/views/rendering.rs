// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! How a trajectory becomes chat records, for the views that train on each
//! action of an agent: [`SftStep`](super::SftStep) imitates the actions of
//! verified work, [`PolicyGradient`](super::PolicyGradient) weighs the actions
//! of every graded attempt by its advantage. Which experiences and which
//! actions a view wants is its own; how a wanted conversation is rendered
//! (from the requests its model was sent, else from its trajectory), which of
//! its actions can be records at all, and how they are packed is the same for
//! both and lives here once.

use splinter_core::chat::WireMessage;
use splinter_core::harness::{HarnessConversation, RecordedCalls};

use crate::images::MAX_IMAGES_PER_RECORD;
use crate::render::student_turn;
use crate::requests::conversation_of_requests;
use crate::trajectory::{conversation_serialised, Action, Conversation};
use crate::{Entry, Exclusion, Projection, RecordBody, Strip};

/// What a view makes of one action that can be a record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Judgement {
    /// A step labelled bad: counted as [`Exclusion::BadStep`].
    Bad,
    /// An action the view trains on.
    Admit,
    /// An action the view does not train on, which is context and not counted.
    Skip,
}

/// The rendering options two views share.
#[derive(Clone, Debug, Default)]
pub(crate) struct Rendering {
    pub(crate) strip: Strip,
    pub(crate) packed: bool,
    pub(crate) serialise_calls: bool,
    pub(crate) only_actions: Option<Vec<String>>,
    pub(crate) only_roles: Option<Vec<String>>,
    pub(crate) images: bool,
}

impl Rendering {
    /// Whether the roles asked for (if any) include the model role that
    /// answered `conversation`; an unknown role matches none.
    pub(crate) fn wants_role(&self, conversation: Option<&HarnessConversation>) -> bool {
        self.only_roles.as_ref().is_none_or(|wanted| {
            conversation
                .and_then(|c| c.role.as_deref())
                .is_some_and(|role| wanted.iter().any(|w| w == role))
        })
    }

    /// The conversation `entry` is: read from the requests its model was sent
    /// when it recorded them, else from its trajectory. A conversation whose
    /// model was shown images is the first only, and only when images are
    /// wanted.
    pub(crate) fn conversation_of(
        &self,
        entry: &Entry,
        harness: Option<&HarnessConversation>,
    ) -> Result<Conversation, Exclusion> {
        let shown_images = harness.is_some_and(|c| c.shown_images);
        if shown_images && !self.images {
            return Err(Exclusion::ShownImages);
        }
        if let Some(recorded) = RecordedCalls::of(&entry.experience.trajectory) {
            let calls = recorded.calls().ok_or(Exclusion::Unrepresentable)?;
            return conversation_of_requests(&entry.experience, &calls);
        }
        if shown_images {
            return Err(Exclusion::RequestsNotRecorded);
        }
        let turn = student_turn(entry, &self.strip)?;
        conversation_serialised(&entry.experience, &turn, self.serialise_calls)
    }

    /// The records of the actions of `conversation` the view `judge`s
    /// admissible: one each, or one for all of them when packed. Every action
    /// that cannot be a record is counted in `projection` under why, whatever
    /// the judge thinks of it: never one ATIF marks as copied context, never a
    /// repeat of a call whose result was the same, never one the view was
    /// asked to leave out ([`Exclusion::ActionNotWanted`]), never one whose
    /// context holds more images than a record may.
    pub(crate) fn records(
        &self,
        conversation: &Conversation,
        projection: &mut Projection,
        judge: impl Fn(&Action) -> Judgement,
    ) -> Vec<RecordBody> {
        let images_before = images_before_each_message(conversation);
        let wanted = |action: &Action| match &self.only_actions {
            None => true,
            Some(names) => {
                let calls = &conversation.messages[action.index].tool_calls;
                calls.is_empty()
                    || calls
                        .iter()
                        .all(|c| names.iter().any(|n| n == &c.function.name))
            }
        };
        let mut admitted = Vec::new();
        for action in conversation.actions.iter().filter(|a| !a.copied) {
            let judgement = judge(action);
            if judgement == Judgement::Bad {
                projection.exclude(Exclusion::BadStep);
            } else if action.repeated {
                projection.exclude(Exclusion::RepeatedAction);
            } else if !wanted(action) {
                projection.exclude(Exclusion::ActionNotWanted);
            } else if images_before[action.index] > MAX_IMAGES_PER_RECORD {
                projection.exclude(Exclusion::TooManyImages);
            } else if judgement == Judgement::Admit {
                admitted.push(action);
            }
        }
        // One record has one list of tools: actions taken under another list
        // than the last one's cannot be packed with it.
        if self.packed && admitted.iter().any(|a| a.tools != admitted[0].tools) {
            projection.exclude(Exclusion::Unrepresentable);
            return Vec::new();
        }
        if self.packed {
            conversation.packed_record(&admitted).into_iter().collect()
        } else {
            admitted
                .into_iter()
                .map(|action| conversation.record(action))
                .collect()
        }
    }
}

/// For each message of `conversation`, the images in the messages up to and
/// including it.
fn images_before_each_message(conversation: &Conversation) -> Vec<usize> {
    conversation
        .messages
        .iter()
        .scan(0, |seen, message: &WireMessage| {
            *seen += message.images.len();
            Some(*seen)
        })
        .collect()
}
