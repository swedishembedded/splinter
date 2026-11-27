// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! How a trajectory becomes chat records, for the views that train on each
//! action of an agent: [`SftStep`](super::SftStep) imitates the actions of
//! verified work, [`PolicyGradient`](super::PolicyGradient) weighs the actions
//! of every graded attempt by its advantage. Which experiences and which
//! actions a view wants is its own; how a wanted conversation is rendered
//! (from the requests its model was sent, one conversation or request by
//! request, else, when it recorded none, from its trajectory), which of its
//! actions can be records at all, and how they are packed is the same for
//! both and lives here once.

use splinter_core::chat::WireMessage;
use splinter_core::harness::{HarnessConversation, ModelCall, RecordedCalls};

use crate::curation::{reads_repository_history, HistoryCuration};
use crate::images::MAX_IMAGES_PER_RECORD;
use crate::render::student_turn;
use crate::requests::{conversation_of_requests, conversations_per_request};
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
    /// What becomes of an action that reads the repository's history
    /// ([`crate::curation`]).
    pub(crate) history: HistoryCuration,
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

    /// The conversation `entry` is, read whole: from the requests its model
    /// was sent when it recorded them, else from its trajectory. A
    /// conversation whose model was shown images is the first only, and only
    /// when images are wanted. Requests that do not read as one conversation
    /// (the context was rewritten between two of them) are refused: no single
    /// list of messages holds what the model saw, and the trajectory's is a
    /// context it was never sent. So is a conversation holding a step that is
    /// no action (an answer whose arguments are not JSON).
    pub(crate) fn conversation_of(
        &self,
        entry: &Entry,
        harness: Option<&HarnessConversation>,
    ) -> Result<Conversation, Exclusion> {
        self.check_images(harness)?;
        let Some(recorded) = RecordedCalls::of(&entry.experience.trajectory) else {
            return self.trajectory_conversation(entry, harness);
        };
        let conversation =
            conversation_of_requests(&entry.experience, &recorded_calls(&recorded)?)?;
        // Read whole, a step that is no action would leave the conversation
        // holding a turn the model did not write (a call the agent repaired)
        // as if it had.
        match conversation.left_out.first() {
            Some(reason) => Err(*reason),
            None => Ok(conversation),
        }
    }

    /// The conversations `entry`'s actions were taken in, for the views that
    /// train on each action: the one conversation its requests are (its steps
    /// that are no action left out and counted, [`Conversation::left_out`]),
    /// else its trajectory's when it recorded none, or, for a single
    /// agent's requests that do not read as one conversation, each request on
    /// its own as it was sent ([`conversations_per_request`]). A context the
    /// model was not sent is never one of them: a step no request answered is
    /// left out, and requests no agent step pairs with are refused
    /// ([`Exclusion::RequestsUnpaired`]), never rebuilt from the trajectory. A
    /// harness conversation's are refused.
    pub(crate) fn conversations_of(
        &self,
        entry: &Entry,
        harness: Option<&HarnessConversation>,
    ) -> Result<Vec<Conversation>, Exclusion> {
        self.check_images(harness)?;
        let Some(recorded) = RecordedCalls::of(&entry.experience.trajectory) else {
            return Ok(vec![self.trajectory_conversation(entry, harness)?]);
        };
        let calls = recorded_calls(&recorded)?;
        match conversation_of_requests(&entry.experience, &calls) {
            Ok(conversation) => Ok(vec![conversation]),
            Err(refusal) if harness.is_some() => Err(refusal),
            Err(_) => conversations_per_request(&entry.experience, &calls),
        }
    }

    fn check_images(&self, harness: Option<&HarnessConversation>) -> Result<(), Exclusion> {
        if harness.is_some_and(|c| c.shown_images) && !self.images {
            return Err(Exclusion::ShownImages);
        }
        Ok(())
    }

    /// The conversation an entry that recorded no requests is, from its
    /// trajectory: never one whose model was shown images, which are not in
    /// it.
    fn trajectory_conversation(
        &self,
        entry: &Entry,
        harness: Option<&HarnessConversation>,
    ) -> Result<Conversation, Exclusion> {
        if harness.is_some_and(|c| c.shown_images) {
            return Err(Exclusion::RequestsNotRecorded);
        }
        let turn = student_turn(entry, &self.strip)?;
        conversation_serialised(&entry.experience, &turn, self.serialise_calls)
    }

    /// The records of the actions of `conversations` the view `judge`s
    /// admissible: one each, or one for all of a conversation's when packed.
    /// Every action that cannot be a record is counted in `projection` under
    /// why, whatever the judge thinks of it: never one ATIF marks as copied
    /// context, never one that reads the repository's history when the
    /// curation drops it ([`Exclusion::RepositoryHistory`]), nor one after it
    /// when the curation truncates there ([`Exclusion::AfterRepositoryHistory`]),
    /// never a repeat of a call whose result was the same, never one the view
    /// was asked to leave out ([`Exclusion::ActionNotWanted`]), never one
    /// whose context holds more images than a record may, never a step the
    /// conversation leaves out (an answer whose arguments are not JSON, a step
    /// no recorded request answered).
    pub(crate) fn records(
        &self,
        conversations: &[Conversation],
        projection: &mut Projection,
        judge: impl Fn(&Action) -> Judgement,
    ) -> Vec<RecordBody> {
        self.records_of_actions(conversations, projection, judge)
            .into_iter()
            .map(|(body, _, _)| body)
            .collect()
    }

    /// [`Self::records`], each with the actions it supervises, in order, and
    /// the conversation it is of.
    pub(crate) fn records_of_actions<'c>(
        &self,
        conversations: &'c [Conversation],
        projection: &mut Projection,
        judge: impl Fn(&Action) -> Judgement,
    ) -> Vec<(RecordBody, Vec<Action>, &'c Conversation)> {
        let mut records = Vec::new();
        // A truncating curation ends the demonstration at the first history
        // read, whichever request it was in.
        let mut truncated = false;
        for conversation in conversations {
            for reason in &conversation.left_out {
                projection.exclude(*reason);
            }
            records.extend(
                self.records_of_one(conversation, projection, &judge, &mut truncated)
                    .into_iter()
                    .map(|(body, actions)| (body, actions, conversation)),
            );
        }
        records
    }

    fn records_of_one(
        &self,
        conversation: &Conversation,
        projection: &mut Projection,
        judge: &impl Fn(&Action) -> Judgement,
        truncated: &mut bool,
    ) -> Vec<(RecordBody, Vec<Action>)> {
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
        let reads_history = |action: &Action| {
            self.history != HistoryCuration::Keep
                && conversation.messages[action.index]
                    .tool_calls
                    .iter()
                    .any(|call| reads_repository_history(&call.function.arguments))
        };
        let mut admitted = Vec::new();
        for action in conversation.actions.iter().filter(|a| !a.copied) {
            let judgement = judge(action);
            let after_cut = *truncated;
            let flagged = reads_history(action);
            *truncated |= flagged && self.history == HistoryCuration::Truncate;
            if judgement == Judgement::Bad {
                projection.exclude(Exclusion::BadStep);
            } else if after_cut {
                projection.exclude(Exclusion::AfterRepositoryHistory);
            } else if flagged {
                projection.exclude(Exclusion::RepositoryHistory);
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
            conversation
                .packed_record(&admitted)
                .map(|body| (body, admitted.into_iter().copied().collect()))
                .into_iter()
                .collect()
        } else {
            admitted
                .into_iter()
                .map(|action| (conversation.record(action), vec![*action]))
                .collect()
        }
    }
}

/// The calls `recorded` holds, or why they cannot be read.
fn recorded_calls(recorded: &RecordedCalls) -> Result<Vec<ModelCall>, Exclusion> {
    recorded.calls().ok_or(Exclusion::Unrepresentable)
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
