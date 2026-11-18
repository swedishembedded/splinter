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
    /// The record only sets its group's baseline: its reward counts among the
    /// group's and nothing of it is trained (brain's `baseline_only`); its
    /// conversation supervises no message.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub baseline_only: bool,
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
            baseline_only: false,
        }
    }

    /// The same outcome on a record that only sets its group's baseline.
    #[must_use]
    pub fn baseline(self) -> Self {
        Self {
            baseline_only: true,
            ..self
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
    /// A shared prompt and two continuations of it, the first preferred:
    /// preference optimisation (DPO, IPO, SimPO, ORPO).
    Preference {
        /// The conversation before the continuations: never trained on.
        prompt: Vec<WireMessage>,
        /// The preferred continuation: one assistant turn, or the turns of an
        /// agent's run from where the two parted (its assistant turns trained
        /// on, the tool results between them context). It opens and ends with
        /// an assistant turn.
        chosen: Vec<WireMessage>,
        /// The continuation it is preferred over, of the same form.
        rejected: Vec<WireMessage>,
        /// The tool schemas the conversation was held under (see
        /// [`Self::Chat::tools`]); empty for none.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tools: Vec<serde_json::Value>,
        /// The attempts the pair compares, when it was derived from attempts.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        identity: Option<RolloutIdentity>,
    },
    /// A prompt, one continuation of it and whether it is one to learn from
    /// (`desirable`) or to unlearn: KTO's judged completion.
    Feedback {
        /// The conversation before the continuation: never trained on.
        prompt: Vec<WireMessage>,
        /// The continuation, of [`Self::Preference::chosen`]'s form.
        completion: Vec<WireMessage>,
        /// Whether the continuation is desirable: its attempt verified as
        /// passed.
        desirable: bool,
        /// The tool schemas the conversation was held under; empty for none.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tools: Vec<serde_json::Value>,
        /// The attempt the completion is.
        identity: RolloutIdentity,
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
            Self::Feedback { .. } => "feedback",
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
            ) | (Self::Preference { .. }, Objective::Preference)
                | (Self::Feedback { .. }, Objective::Feedback)
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

impl RecordMetadata {
    /// The metadata of a record `view` made for `objective` from nothing it
    /// names: no experience, task or source, no group, side or draw.
    #[must_use]
    pub fn of(view: &str, objective: Objective) -> Self {
        Self {
            experiences: Vec::new(),
            task: None,
            sources: Vec::new(),
            group: None,
            split: None,
            split_draw: None,
            view: view.into(),
            objective,
        }
    }
}

/// What the holdout rule draws a record by: the seed of its dataset's split
/// and the stratum the record is drawn in, so held-out and monitoring sets
/// hold every stratum at its share.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitDraw {
    /// The seed of the split, the same for every record of a dataset.
    pub seed: u64,
    /// The stratum: the domain the record's task exercises, as its
    /// provenance records it; `None` is a stratum of its own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stratum: Option<String>,
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
    /// The seed and stratum the holdout rule draws it by (see
    /// `crate::holdout`); `None` for a record written before the split drew
    /// by seed, or by a producer that leaves the split to the newest end.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split_draw: Option<SplitDraw>,
    /// The view that projected it.
    pub view: String,
    /// The objective it serves.
    pub objective: Objective,
}

/// Which attempts a preference pair or a judged completion comes from: the
/// group of attempts they belong to, the attempts themselves and the
/// tokenizer their model encoded them with. Written into the line's
/// `metadata` (as `rollout`), which the trainer carries and never reads, so a
/// dataset can be traced to its attempts and checked for one tokenizer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RolloutIdentity {
    /// The key of the group: a digest of the task, its version and the policy
    /// that attempted it, as a policy-gradient record names its group.
    pub group: String,
    /// The attempts, by trial (else experience) name: for a pair the chosen
    /// one, then the rejected one; for a judged completion its own.
    pub rollouts: Vec<String>,
    /// The tokenizer the policy's server encoded the attempts with
    /// (`sha256:<hex>`, as its policy trace names it), when the trajectories
    /// recorded it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_digest: Option<String>,
    /// Where a pair comes from: grouped attempts of one task, or a repair (a
    /// failed attempt's prefix continued to a verified pass, against that
    /// attempt's own failed suffix); `None` for a judged completion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<PairSource>,
}

/// Where a preference pair comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PairSource {
    /// A verified pass and a verified fail of one task by one policy, from
    /// the same rendered prompt.
    Attempts,
    /// A failed attempt's prefix continued to a verified pass, against the
    /// failed attempt's own suffix: only where the two part is trained.
    Repair,
}
