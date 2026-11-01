// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent-training pipelines that train only on
// what an agent really saw and said, for its clients. If your team needs
// expertise in turning agent trajectories into fine-tuning data, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Why a trajectory, or the requests of a conversation, cannot be rendered as
//! a training conversation.

use crate::Exclusion;

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
    /// A recorded request holds a message, or an image, a chat record cannot
    /// carry.
    #[error("the request answered at step {step} holds a message a chat record cannot represent")]
    UnreadableRequest {
        /// The agent step the request was answered at.
        step: u64,
    },
    /// The recorded requests do not extend one another: a later one does not
    /// hold the earlier one and the model's answer to it, or the log does not
    /// divide among the trajectory's agent steps. The context was rewritten
    /// between them, so no single conversation holds them all.
    #[error("the request answered at step {step} does not continue the requests before it")]
    RequestsDiverge {
        /// The agent step the request was answered at.
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
    /// The solve recorded a system prompt that only its digest is known by:
    /// the record cannot open with the prompt the run was under.
    #[error("the solve was held to a system prompt this build holds no text of")]
    UnknownSystemPrompt,
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

impl From<Unprojectable> for Exclusion {
    fn from(refusal: Unprojectable) -> Self {
        match refusal {
            Unprojectable::ReasoningInText { .. } => Self::ReasoningInText,
            Unprojectable::Incomplete => Self::Incomplete,
            Unprojectable::Compacted { .. } | Unprojectable::RequestsDiverge { .. } => {
                Self::Compacted
            }
            Unprojectable::UnparsedArguments { .. } => Self::UnparsedArguments,
            Unprojectable::DuplicateCallId { .. } => Self::DuplicateCallId,
            Unprojectable::NotConserved { .. } => Self::NotConserved,
            _ => Self::Unrepresentable,
        }
    }
}
