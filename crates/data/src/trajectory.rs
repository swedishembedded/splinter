// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! An ATIF trajectory as the conversation a student is trained on.
//!
//! * The student's turn replaces the trajectory's first user step: the
//!   solver may have been prompted with privileged context, the student is
//!   shown what the strip policy allows (see [`crate::Strip`]). A second
//!   user step is not representable: an experience records one
//!   instruction.
//! * System steps are left out: every record starts with the one system
//!   turn every solve runs under (`splinter_core::prompt::SYSTEM_PROMPT`), which
//!   the projection puts first.
//! * An agent step is one assistant message - its text and every tool
//!   call it made, in `generic-messages-v2`'s tool-call form with the
//!   arguments JSON-encoded - followed by one `tool` message per
//!   observation result, answering its call by id. Reasoning content is
//!   not part of the wire format and is left out.
//! * An agent step with tool calls is an action; so is the last agent step
//!   when it answers without one (the final answer). An agent step with
//!   neither text nor calls adds nothing and is skipped.
//! * A dialogue - a trajectory with a second user step and no tool call -
//!   is the exception: [`dialogue`] renders it whole, the later user steps
//!   (the other speaker's own words) as user turns and every reply
//!   supervised.
//! * Image content and an observation that answers no call of its step are
//!   not representable, and the whole trajectory is refused: a record with
//!   a hole in its context would teach from a state the solver never saw.

use atif::{ContentSegment, MessageBody, StepOrigin, TraceStep, Trajectory};
use splinter_core::chat::{WireFunction, WireMessage, WireToolCall};

use crate::render::message;

/// Why a trajectory cannot be rendered as training conversations; every
/// variant names the ATIF `step_id` that decided it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Unprojectable {
    /// The step holds image content, which a chat record cannot carry.
    #[error("step {step} holds image content, which a training record cannot represent")]
    Image {
        /// The step.
        step: u64,
    },
    /// The step's observation answers no tool call of the same step.
    #[error("step {step} has an observation that answers no tool call of its own")]
    UnansweredObservation {
        /// The step.
        step: u64,
    },
    /// A second user step beside tool calls: a trajectory is rendered either
    /// as one instruction with its actions or as a dialogue of replies.
    #[error(
        "step {step} makes a tool call in a trajectory with more than one user step; it can be \
         rendered as a dialogue or as a tool-call record, not both"
    )]
    ToolCallInDialogue {
        /// The agent step with the call or observation.
        step: u64,
    },
    /// A second user step where one instruction with its actions is wanted.
    #[error("step {step} is a second user step; an instruction with actions has one")]
    SecondUserStep {
        /// The step.
        step: u64,
    },
    /// A dialogue step copied from another trajectory, which is never
    /// supervised.
    #[error("step {step} is copied from another trajectory and cannot be part of a dialogue")]
    CopiedInDialogue {
        /// The step.
        step: u64,
    },
    /// Two user steps, or two agent steps, in a row.
    #[error("step {step} does not alternate user and agent turns")]
    NotAlternating {
        /// The step.
        step: u64,
    },
    /// The dialogue ends on a user step, which no reply answers.
    #[error("the dialogue ends on user step {step}, which no agent reply answers")]
    EndsUnanswered {
        /// The last step.
        step: u64,
    },
}

/// Why `trajectory` is refused by the projection, or `None` when it can be
/// rendered: as a dialogue when it has two or more user steps, otherwise as
/// one instruction with its actions.
#[must_use]
pub fn projection_refusal(trajectory: &Trajectory) -> Option<Unprojectable> {
    let users = trajectory
        .steps
        .iter()
        .filter(|s| s.source == StepOrigin::User)
        .count();
    if users >= 2 {
        try_dialogue(trajectory, "", &|_, _| true).err()
    } else {
        try_conversation(trajectory, "").err()
    }
}

/// One action of a trajectory: the assistant message at `index`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Action {
    /// Its index in [`Conversation::messages`].
    pub(crate) index: usize,
    /// The ATIF `step_id` of the step it is.
    pub(crate) step: u64,
    /// Whether ATIF marks the step as copied from another trajectory, which
    /// must never be supervised.
    pub(crate) copied: bool,
}

/// A trajectory as messages, none supervised, and where its actions are.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Conversation {
    /// The messages.
    pub(crate) messages: Vec<WireMessage>,
    /// Its actions, in order.
    pub(crate) actions: Vec<Action>,
}

impl Conversation {
    /// The messages up to and including `action`, with only the action
    /// supervised.
    pub(crate) fn up_to(&self, action: &Action) -> Vec<WireMessage> {
        let mut messages: Vec<WireMessage> = self
            .messages
            .iter()
            .take(action.index + 1)
            .cloned()
            .collect();
        if let Some(last) = messages.last_mut() {
            last.train = true;
        }
        messages
    }
}

/// `trajectory` rendered with `student_turn` as its user turn; `None` when
/// it holds something a chat record cannot represent.
pub(crate) fn conversation(trajectory: &Trajectory, student_turn: &str) -> Option<Conversation> {
    try_conversation(trajectory, student_turn).ok()
}

/// [`conversation`], naming what makes it impossible.
fn try_conversation(
    trajectory: &Trajectory,
    student_turn: &str,
) -> Result<Conversation, Unprojectable> {
    let mut messages = vec![message("user", student_turn, false)];
    let mut actions = Vec::new();
    let mut users = 0usize;
    let last_agent = trajectory
        .steps
        .iter()
        .rposition(|s| s.source == StepOrigin::Agent);
    for (position, step) in trajectory.steps.iter().enumerate() {
        match step.source {
            StepOrigin::System => {}
            StepOrigin::User => {
                users += 1;
                if users > 1 {
                    return Err(Unprojectable::SecondUserStep { step: step.step_id });
                }
            }
            StepOrigin::Agent => {
                let text =
                    text_of(&step.message).ok_or(Unprojectable::Image { step: step.step_id })?;
                let calls = wire_calls(step);
                if text.is_empty() && calls.is_empty() {
                    continue;
                }
                let is_action = !calls.is_empty() || Some(position) == last_agent;
                let ids: Vec<String> = calls.iter().filter_map(|c| c.id.clone()).collect();
                let mut assistant = message("assistant", &text, false);
                assistant.tool_calls = calls;
                if is_action {
                    actions.push(Action {
                        index: messages.len(),
                        step: step.step_id,
                        copied: step.is_excluded_from_sft(),
                    });
                }
                messages.push(assistant);
                for result in step.observation.iter().flat_map(|o| &o.results) {
                    let call = result
                        .source_call_id
                        .as_ref()
                        .filter(|id| ids.contains(id))
                        .ok_or(Unprojectable::UnansweredObservation { step: step.step_id })?;
                    let content = match &result.content {
                        Some(body) => {
                            text_of(body).ok_or(Unprojectable::Image { step: step.step_id })?
                        }
                        None => String::new(),
                    };
                    let mut tool = message("tool", &content, false);
                    tool.tool_call_id = Some(call.clone());
                    messages.push(tool);
                }
            }
        }
    }
    Ok(Conversation { messages, actions })
}

/// `trajectory` as one conversation of alternating turns, every reply
/// supervised, with `student_turn` as the first user turn and the later user
/// steps as they were said; `None` when it is not a dialogue (fewer than two
/// user steps), holds a tool call, an image or a step copied from another
/// trajectory, or does not alternate user then agent.
pub(crate) fn dialogue(trajectory: &Trajectory, student_turn: &str) -> Option<Vec<WireMessage>> {
    let users = trajectory
        .steps
        .iter()
        .filter(|s| s.source == StepOrigin::User)
        .count();
    if users < 2 {
        return None;
    }
    try_dialogue(trajectory, student_turn, &|_, _| true).ok()
}

/// A recorded session as the conversation it was, from its own first user
/// step. The agent's reply at step `step` with text `text` is supervised
/// when `supervise(step, text)` says so, and context only otherwise - what
/// the agent said before it was corrected is shown and never trained on. A
/// session with one user step is a dialogue of one exchange. Refused, naming
/// the step, when it holds a tool call, an image or a copied step, or does
/// not alternate.
pub fn session_dialogue(
    trajectory: &Trajectory,
    supervise: impl Fn(u64, &str) -> bool,
) -> Result<Vec<WireMessage>, Unprojectable> {
    let opening = trajectory
        .steps
        .iter()
        .find(|s| s.source == StepOrigin::User)
        .map(|s| text_of(&s.message).ok_or(Unprojectable::Image { step: s.step_id }))
        .transpose()?
        .unwrap_or_default();
    try_dialogue(trajectory, &opening, &supervise)
}

/// [`dialogue`], naming what makes it impossible. A reply is supervised
/// where `supervise` says so.
fn try_dialogue(
    trajectory: &Trajectory,
    student_turn: &str,
    supervise: &dyn Fn(u64, &str) -> bool,
) -> Result<Vec<WireMessage>, Unprojectable> {
    let mut messages = Vec::new();
    let mut users = 0usize;
    let mut last_step = 0;
    for step in &trajectory.steps {
        last_step = step.step_id;
        match step.source {
            StepOrigin::System => {}
            StepOrigin::User => {
                if messages
                    .last()
                    .is_some_and(|m: &WireMessage| m.role == "user")
                {
                    return Err(Unprojectable::NotAlternating { step: step.step_id });
                }
                users += 1;
                let text = if users == 1 {
                    student_turn.to_string()
                } else {
                    text_of(&step.message).ok_or(Unprojectable::Image { step: step.step_id })?
                };
                messages.push(message("user", &text, false));
            }
            StepOrigin::Agent => {
                if !wire_calls(step).is_empty() || step.observation.is_some() {
                    return Err(Unprojectable::ToolCallInDialogue { step: step.step_id });
                }
                if step.is_excluded_from_sft() {
                    return Err(Unprojectable::CopiedInDialogue { step: step.step_id });
                }
                if messages.last().map(|m| m.role.as_str()) != Some("user") {
                    return Err(Unprojectable::NotAlternating { step: step.step_id });
                }
                let text =
                    text_of(&step.message).ok_or(Unprojectable::Image { step: step.step_id })?;
                let supervised = supervise(step.step_id, &text);
                messages.push(message("assistant", &text, supervised));
            }
        }
    }
    if messages.last().is_some_and(|m| m.role == "user") {
        return Err(Unprojectable::EndsUnanswered { step: last_step });
    }
    Ok(messages)
}

/// The tool calls `step` made, in `generic-messages-v2`'s tool-call form
/// with the arguments JSON-encoded.
fn wire_calls(step: &TraceStep) -> Vec<WireToolCall> {
    step.tool_calls
        .iter()
        .flatten()
        .map(|call| WireToolCall {
            id: Some(call.tool_call_id.clone()),
            kind: "function".into(),
            function: WireFunction {
                name: call.function_name.clone(),
                arguments: call.arguments.to_string(),
            },
        })
        .collect()
}

/// The tool calls `trajectory`'s last agent step made, as
/// [`conversation`] renders them; empty when it made none.
pub(crate) fn final_calls(trajectory: &Trajectory) -> Vec<WireToolCall> {
    trajectory
        .steps
        .iter()
        .rfind(|s| s.source == StepOrigin::Agent)
        .map(wire_calls)
        .unwrap_or_default()
}

/// The text of `body`; `None` when it holds an image.
fn text_of(body: &MessageBody) -> Option<String> {
    match body {
        MessageBody::Text(text) => Some(text.clone()),
        MessageBody::Segments(segments) => segments
            .iter()
            .map(|segment| match segment {
                ContentSegment::Text { text } => Some(text.as_str()),
                ContentSegment::Image { .. } => None,
            })
            .collect::<Option<Vec<&str>>>()
            .map(|parts| parts.concat()),
    }
}

/// Whether `a` and `b` say the same thing, ignoring call ids (a fresh run
/// draws fresh ones) and supervision.
pub(crate) fn same_message(a: &WireMessage, b: &WireMessage) -> bool {
    a.role == b.role
        && a.content == b.content
        && a.tool_calls.len() == b.tool_calls.len()
        && a.tool_calls
            .iter()
            .zip(&b.tool_calls)
            .all(|(x, y)| x.function == y.function)
}
