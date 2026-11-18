// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-set curation that projects
// verified agent experience into supervised datasets, for its clients. If
// your team needs expertise in dataset curation for fine-tuning, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What a view yields: a [`Record`], its [`RecordBody`] (each [`Objective`]
//! has one shape) and the [`RecordMetadata`] that says where it came from.

use serde::{Deserialize, Serialize};
use splinter_core::chat::WireMessage;
use splinter_core::digest::{digest_of, Digest};
use splinter_core::experience::ExperienceId;

use crate::Objective;

/// One privileged turn of a record's teacher context: text the student never
/// saw, shown to the frozen model that scores the student's own tokens.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TeacherMessage {
    /// `system` or `user`.
    pub role: String,
    /// The text.
    pub content: String,
}

/// The outcome of the rollout a chat record is one model call of, as brain's
/// training-record contract (version 2) takes it: the reward, the group of
/// rollouts it is compared with, and which rollout of the group it is.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RolloutOutcome {
    /// The verifier's reward for the rollout.
    pub reward: f64,
    /// The key of the group of rollouts the reward is compared within (the
    /// attempts at one task by one policy).
    pub group: String,
    /// Which rollout of the group: the records of one rollout share its
    /// reward and count once in the group. `None`: the record is a rollout
    /// of its own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollout: Option<String>,
}

/// The longest `group` or `rollout` key brain accepts, in bytes.
const MAX_KEY_BYTES: usize = 512;

impl RolloutOutcome {
    /// The outcome `reward` of rollout `rollout` in the group named by
    /// `group`'s parts (the task, its version, the policy, ...). The group's
    /// key is a digest of the parts, so two groups never share one whatever
    /// their names hold; the rollout's is its name when brain accepts it as
    /// a key (1 to 512 bytes, no control characters), else its digest.
    #[must_use]
    pub fn of(reward: f64, group: &[&str], rollout: &str) -> Self {
        let fits = !rollout.is_empty()
            && rollout.len() <= MAX_KEY_BYTES
            && !rollout.chars().any(char::is_control);
        Self {
            reward,
            group: digest_of(group).to_string(),
            rollout: Some(if fits {
                rollout.to_string()
            } else {
                digest_of(rollout).to_string()
            }),
        }
    }
}

/// What a record holds; each [`Objective`] has one shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "snake_case")]
pub enum RecordBody {
    /// A conversation in `generic-messages-v2` messages; `train` marks the
    /// supervised turns. SFT and classification.
    Chat {
        /// The conversation.
        messages: Vec<WireMessage>,
        /// The tool schemas the conversation was held under, in the form
        /// brain's chat template renders them: OpenAI function-calling
        /// entries, `{"type": "function", "function": {name, description,
        /// parameters}}`. Empty for a conversation with no tools, which is
        /// then written as it was before records had this field.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tools: Vec<serde_json::Value>,
        /// The outcome of the attempt the conversation is an action of: its
        /// reward and the group of attempts it is compared with, from which
        /// brain computes the advantage that weights every supervised token
        /// of the record in a policy-gradient run. `None` for a record that
        /// is imitated.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        outcome: Option<RolloutOutcome>,
        /// What only the teacher of a self-distillation run is shown, as
        /// extra turns right after the conversation's first user turn: the
        /// student's own conversation is [`Self::Chat::messages`] and never
        /// holds any of it. Empty for a record trained without a teacher.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        teacher_context: Vec<TeacherMessage>,
    },
    /// A shared prompt and two replies to it, the first preferred. DPO.
    Preference {
        /// The conversation before the reply.
        prompt: Vec<WireMessage>,
        /// The preferred reply.
        chosen: Box<WireMessage>,
        /// The reply it is preferred over.
        rejected: Box<WireMessage>,
    },
    /// A query, the text that answers it, and texts that do not. Contrastive.
    Contrastive {
        /// The query.
        query: String,
        /// The text it is grounded in.
        positive: String,
        /// Other texts of the same source it is not grounded in.
        negatives: Vec<String>,
    },
    /// A whole conversation and the reward it earned. Reward.
    Rewarded {
        /// The conversation; the solver's turns are marked `train`.
        messages: Vec<WireMessage>,
        /// The derived reward, always measured.
        reward: f64,
    },
    /// Raw text. Continued pretraining.
    Text {
        /// The text.
        text: String,
    },
}

impl RecordBody {
    /// A chat record of `messages` held under no tools.
    #[must_use]
    pub fn chat(messages: Vec<WireMessage>) -> Self {
        Self::Chat {
            messages,
            tools: Vec::new(),
            outcome: None,
            teacher_context: Vec::new(),
        }
    }

    /// This record carrying `outcome`, when it is a chat record; any other
    /// shape has no rollout to carry and is returned as it is.
    #[must_use]
    pub fn with_outcome(self, outcome: RolloutOutcome) -> Self {
        match self {
            Self::Chat {
                messages,
                tools,
                teacher_context,
                ..
            } => Self::Chat {
                messages,
                tools,
                outcome: Some(outcome),
                teacher_context,
            },
            other => other,
        }
    }

    /// This record shown to a teacher with `context` besides, when it is a
    /// chat record; any other shape has no teacher and is returned as it is.
    #[must_use]
    pub fn with_teacher_context(self, context: Vec<TeacherMessage>) -> Self {
        match self {
            Self::Chat {
                messages,
                tools,
                outcome,
                ..
            } => Self::Chat {
                messages,
                tools,
                outcome,
                teacher_context: context,
            },
            other => other,
        }
    }

    /// The shape's name, as it is serialized.
    #[must_use]
    pub fn shape(&self) -> &'static str {
        match self {
            Self::Chat { .. } => "chat",
            Self::Preference { .. } => "preference",
            Self::Contrastive { .. } => "contrastive",
            Self::Rewarded { .. } => "rewarded",
            Self::Text { .. } => "text",
        }
    }

    /// Whether this is the shape `objective`'s records have.
    #[must_use]
    pub fn serves(&self, objective: Objective) -> bool {
        matches!(
            (self, objective),
            (
                Self::Chat { .. },
                Objective::Sft | Objective::Classification | Objective::PolicyGradient
            ) | (Self::Preference { .. }, Objective::Dpo)
                | (Self::Contrastive { .. }, Objective::Contrastive)
                | (Self::Rewarded { .. }, Objective::Reward)
                | (Self::Text { .. }, Objective::Cpt)
        )
    }
}

/// One training record and where it came from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Record {
    /// What is trained on.
    pub body: RecordBody,
    /// Where it came from; carried in the dataset, never rendered into the
    /// prompt.
    pub metadata: RecordMetadata,
}

/// Which side of the training split a record is fixed to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Side {
    /// Always trained on: a record the run exists to teach, which no rule
    /// may hold out.
    Train,
    /// Never trained on: a record that only measures.
    HeldOut,
}

/// Where a record came from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordMetadata {
    /// The experiences it was projected from, in the view's order (for a
    /// pair: the chosen or critiquing one first).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub experiences: Vec<ExperienceId>,
    /// The task it was projected from, for a view that reads tasks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<Digest>,
    /// The content digests of the source text it holds, for a view that
    /// reads sources.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<Digest>,
    /// The group of overlapping source text it was projected from, when
    /// known: records of one group are held out or trained on together (see
    /// `crate::holdout`). Set by whoever has the source text, never
    /// by a view.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
    /// The side of the split the producer fixed for it, when it did; the
    /// holdout rule leaves such a record where it is (see
    /// `crate::holdout::Membership`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split: Option<Side>,
    /// The view that projected it.
    pub view: String,
    /// The objective it serves.
    pub objective: Objective,
}
