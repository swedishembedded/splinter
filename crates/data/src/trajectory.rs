// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! An ATIF trajectory as the conversation a student is trained on.
//!
//! * The student's turn replaces the trajectory's first user step: the
//!   solver may have been prompted with privileged context, the student is
//!   shown what the strip policy allows (see [`crate::Strip`]). A second
//!   user step is not representable: an experience records one
//!   instruction. In a workspace, the exception is feedback: the later user
//!   steps are the visible checks' output sent back to the agent, and they
//!   are rendered as user turns that are context and never supervised.
//! * System steps are left out: every record starts with the one system
//!   turn every solve runs under (`splinter_core::prompt::SYSTEM_PROMPT`), which
//!   the projection puts first. A workspace run is held under the prompt its
//!   harness pinned, which the trajectory records as a leading system step;
//!   that prompt opens the record instead, because the agent is served under
//!   it. A system step that marks the context being compacted refuses the
//!   trajectory: what the agent saw after it is not the steps before it.
//! * An assistant turn is trimmed: a model leaves whitespace around an empty
//!   reasoning block, which a chat template renders differently alone and in
//!   context, so no loss mask could be fixed for the turn. Text that holds
//!   reasoning markup refuses the trajectory, and so does a tool call whose
//!   arguments the model sent as text that is not JSON.
//! * The tool schemas the agent was offered are the record's `tools`.
//! * An agent step is one assistant message - its text and every tool
//!   call it made, in `generic-messages-v2`'s tool-call form with the
//!   arguments JSON-encoded - followed by one `tool` message per
//!   observation result, answering its call by id. Reasoning content is
//!   not part of the wire format and is left out.
//! * Results are paired to calls by id within their step. A step in which two
//!   calls share an id is refused (the pairing is ambiguous); ids repeat across
//!   steps (a model numbers each response's calls from zero), so the record
//!   names each call `s<step>-<id>`, unique in the conversation. The projection
//!   conserves its source: as many tool calls and tool results out as the steps
//!   it projects hold in.
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

use std::collections::HashSet;

use atif::{ContentSegment, MessageBody, StepOrigin, TraceStep, Trajectory};
use splinter_core::chat::{WireFunction, WireMessage, WireToolCall};
use splinter_core::experience::{Environment, Experience};

use crate::markers::{
    chat_tools, has_unparsed_arguments, is_compaction, is_harness_prompt, is_incomplete,
};
use crate::render::message;
use crate::{Exclusion, RecordBody};

/// Why a trajectory cannot be rendered as training conversations; every
/// variant names the ATIF `step_id` that decided it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Unprojectable {
    /// The capture says the run did not end cleanly (see
    /// `atif::persist::CAPTURE_STATUS_KEY`): a prefix of a run is not a record
    /// of one, and its steps are not a success whatever they end on.
    #[error("the trajectory is marked incomplete: the run it records did not finish")]
    Incomplete,
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
    /// An assistant turn holds `<think>` markup.
    #[error("step {step} holds reasoning markup in its text, which a chat record cannot carry")]
    ReasoningInText {
        /// The step.
        step: u64,
    },
    /// The agent's context was compacted before this step.
    #[error("step {step} marks the context being compacted; the turns before it are not what the agent saw after it")]
    Compacted {
        /// The system step that marks it.
        step: u64,
    },
    /// A tool call whose arguments are a stand-in for text that is not JSON.
    #[error(
        "step {step} makes a tool call whose arguments the model sent as text that is not JSON"
    )]
    UnparsedArguments {
        /// The agent step with the call.
        step: u64,
    },
    /// A tool call id used by two calls of one step: results are paired to
    /// calls by id, so every call would be given every result.
    #[error("step {step} makes two tool calls with the same id")]
    DuplicateCallId {
        /// The agent step with the two calls.
        step: u64,
    },
    /// The projection lost or invented a call or a result of its source.
    #[error(
        "the projection holds {calls} tool calls and {results} tool results where its source has \
         {source_calls} and {source_results}"
    )]
    NotConserved {
        /// Tool calls in the projected messages.
        calls: usize,
        /// Tool results in the projected messages.
        results: usize,
        /// Tool calls of the steps projected.
        source_calls: usize,
        /// Observation results of the steps projected.
        source_results: usize,
    },
    /// A tool schema that is neither an OpenAI function-calling entry nor
    /// a bare function schema.
    #[error("tool definition {index} is not a function schema")]
    ToolDefinition {
        /// Its index in the trajectory's `tool_definitions`.
        index: usize,
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
        try_conversation(trajectory, "", Setting::Closed, false).err()
    }
}

impl From<Unprojectable> for Exclusion {
    fn from(refusal: Unprojectable) -> Self {
        match refusal {
            Unprojectable::ReasoningInText { .. } => Self::ReasoningInText,
            Unprojectable::Incomplete => Self::Incomplete,
            Unprojectable::Compacted { .. } => Self::Compacted,
            Unprojectable::UnparsedArguments { .. } => Self::UnparsedArguments,
            Unprojectable::DuplicateCallId { .. } => Self::DuplicateCallId,
            Unprojectable::NotConserved { .. } => Self::NotConserved,
            _ => Self::Unrepresentable,
        }
    }
}

/// Where the agent worked, as far as rendering its trajectory is concerned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Setting {
    /// One instruction, then actions: a second user step is refused.
    Closed,
    /// A workspace: later user steps are feedback on the work so far, and
    /// the harness's own system prompt is the one the agent was served under.
    Workspace,
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
    /// The tool schemas the agent was offered, in a chat record's form.
    pub(crate) tools: Vec<serde_json::Value>,
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

    /// The chat record of all of `actions` together: the conversation up to
    /// the last of them with each of them supervised and every other message
    /// context, under the tools the agent was offered. `None` when there is
    /// no action.
    pub(crate) fn packed_record(&self, actions: &[&Action]) -> Option<RecordBody> {
        let last = actions.iter().map(|a| a.index).max()?;
        let mut messages: Vec<WireMessage> = self.messages.iter().take(last + 1).cloned().collect();
        for action in actions {
            messages[action.index].train = true;
        }
        Some(RecordBody::Chat {
            messages,
            tools: self.tools.clone(),
        })
    }

    /// The chat record of `action`: the conversation up to it with only it
    /// supervised, under the tools the agent was offered.
    pub(crate) fn record(&self, action: &Action) -> RecordBody {
        RecordBody::Chat {
            messages: self.up_to(action),
            tools: self.tools.clone(),
        }
    }
}

/// The conversation `experience`'s trajectory is, with `student_turn` as
/// its user turn; the reason it cannot be one otherwise. A workspace
/// experience renders as [`Setting::Workspace`] says, any other as one
/// instruction with its actions.
pub(crate) fn conversation(
    experience: &Experience,
    student_turn: &str,
) -> Result<Conversation, Exclusion> {
    conversation_serialised(experience, student_turn, false)
}

/// [`conversation`], and with `serialise_calls` a trajectory whose whole run
/// was exported as one agent step is taken as one call per turn: see
/// [`try_conversation`].
pub(crate) fn conversation_serialised(
    experience: &Experience,
    student_turn: &str,
    serialise_calls: bool,
) -> Result<Conversation, Exclusion> {
    let setting = if experience.environment.kind == Environment::WORKSPACE {
        Setting::Workspace
    } else {
        Setting::Closed
    };
    Ok(try_conversation(
        &experience.trajectory,
        student_turn,
        setting,
        serialise_calls,
    )?)
}

/// [`conversation`], naming what makes it impossible. A trajectory with a
/// single agent step that holds several tool calls is a whole run exported as
/// one step, whose turn boundaries are not recorded; with `serialise_calls` its
/// calls become one assistant turn each, followed by its result, and the step's
/// text one last assistant turn. A trajectory with several agent steps keeps
/// each step as one turn, parallel calls included.
fn try_conversation(
    trajectory: &Trajectory,
    student_turn: &str,
    setting: Setting,
    serialise_calls: bool,
) -> Result<Conversation, Unprojectable> {
    let one_run_step = trajectory
        .steps
        .iter()
        .filter(|s| s.source == StepOrigin::Agent)
        .count()
        == 1;
    if is_incomplete(trajectory) {
        return Err(Unprojectable::Incomplete);
    }
    let tools = chat_tools(trajectory)?;
    let mut messages = vec![message("user", student_turn, false)];
    let mut actions = Vec::new();
    let mut users = 0usize;
    let (mut source_calls, mut source_results) = (0usize, 0usize);
    let last_agent = trajectory
        .steps
        .iter()
        .rposition(|s| s.source == StepOrigin::Agent);
    for (position, step) in trajectory.steps.iter().enumerate() {
        match step.source {
            StepOrigin::System if is_compaction(step) => {
                return Err(Unprojectable::Compacted { step: step.step_id });
            }
            StepOrigin::System => {
                let leads = messages.len() == 1;
                if setting == Setting::Workspace && leads && is_harness_prompt(step) {
                    let prompt = text_of(&step.message)
                        .ok_or(Unprojectable::Image { step: step.step_id })?;
                    messages.insert(0, message("system", &prompt, false));
                }
            }
            StepOrigin::User => {
                users += 1;
                if users > 1 {
                    if setting == Setting::Closed {
                        return Err(Unprojectable::SecondUserStep { step: step.step_id });
                    }
                    let feedback = text_of(&step.message)
                        .ok_or(Unprojectable::Image { step: step.step_id })?;
                    messages.push(message("user", &feedback, false));
                }
            }
            StepOrigin::Agent => {
                let text =
                    text_of(&step.message).ok_or(Unprojectable::Image { step: step.step_id })?;
                let text = text.trim().to_string();
                if text.contains("<think>") || text.contains("</think>") {
                    return Err(Unprojectable::ReasoningInText { step: step.step_id });
                }
                if step.tool_calls.iter().flatten().any(has_unparsed_arguments) {
                    return Err(Unprojectable::UnparsedArguments { step: step.step_id });
                }
                let calls = wire_calls(step);
                if text.is_empty() && calls.is_empty() {
                    continue;
                }
                let mut in_step: HashSet<&String> = HashSet::new();
                if !calls
                    .iter()
                    .filter_map(|c| c.id.as_ref())
                    .all(|id| in_step.insert(id))
                {
                    return Err(Unprojectable::DuplicateCallId { step: step.step_id });
                }
                source_calls += calls.len();
                source_results += step.observation.iter().flat_map(|o| &o.results).count();
                if serialise_calls && one_run_step && calls.len() > 1 {
                    serialise_run_step(step, &text, calls, &mut messages, &mut actions)?;
                    continue;
                }
                let is_action = !calls.is_empty() || Some(position) == last_agent;
                let ids: Vec<String> = calls.iter().filter_map(|c| c.id.clone()).collect();
                let mut assistant = message("assistant", &text, false);
                assistant.tool_calls = calls.into_iter().map(|c| scoped(step, c)).collect();
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
                    tool.tool_call_id = Some(scoped_id(step, call));
                    messages.push(tool);
                }
            }
        }
    }
    let calls = messages.iter().map(|m| m.tool_calls.len()).sum();
    let results = messages.iter().filter(|m| m.role == "tool").count();
    if (calls, results) != (source_calls, source_results) {
        return Err(Unprojectable::NotConserved {
            calls,
            results,
            source_calls,
            source_results,
        });
    }
    Ok(Conversation {
        messages,
        actions,
        tools,
    })
}

/// A call id as the record names it: scoped to its step. A model numbers the
/// calls of each response from zero, so ids repeat across the steps of one
/// trajectory; results are paired to calls within their step, and the record
/// must name every call uniquely (brain refuses a conversation that does not).
fn scoped_id(step: &TraceStep, id: &str) -> String {
    format!("s{}-{id}", step.step_id)
}

fn scoped(step: &TraceStep, mut call: WireToolCall) -> WireToolCall {
    call.id = call.id.as_deref().map(|id| scoped_id(step, id));
    call
}

/// Appends the turns of a run exported as one agent step: each call as an
/// assistant turn with its result after it, then the step's text as the last
/// turn. Every assistant turn is an action of the step.
fn serialise_run_step(
    step: &TraceStep,
    text: &str,
    calls: Vec<WireToolCall>,
    messages: &mut Vec<WireMessage>,
    actions: &mut Vec<Action>,
) -> Result<(), Unprojectable> {
    let ids: Vec<&String> = calls.iter().filter_map(|c| c.id.as_ref()).collect();
    for result in step.observation.iter().flat_map(|o| &o.results) {
        if !result
            .source_call_id
            .as_ref()
            .is_some_and(|id| ids.contains(&id))
        {
            return Err(Unprojectable::UnansweredObservation { step: step.step_id });
        }
    }
    let mut action = |messages: &mut Vec<WireMessage>, assistant: WireMessage| {
        actions.push(Action {
            index: messages.len(),
            step: step.step_id,
            copied: step.is_excluded_from_sft(),
        });
        messages.push(assistant);
    };
    for call in calls {
        let id = call.id.clone();
        let mut assistant = message("assistant", "", false);
        assistant.tool_calls = vec![scoped(step, call)];
        action(messages, assistant);
        for result in step
            .observation
            .iter()
            .flat_map(|o| &o.results)
            .filter(|r| r.source_call_id == id)
        {
            let content = match &result.content {
                Some(body) => text_of(body).ok_or(Unprojectable::Image { step: step.step_id })?,
                None => String::new(),
            };
            let mut tool = message("tool", &content, false);
            tool.tool_call_id = id.as_deref().map(|id| scoped_id(step, id));
            messages.push(tool);
        }
    }
    if !text.is_empty() {
        action(messages, message("assistant", text, false));
    }
    Ok(())
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
    if is_incomplete(trajectory) {
        return Err(Unprojectable::Incomplete);
    }
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
