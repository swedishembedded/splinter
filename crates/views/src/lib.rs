// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-set curation that projects
// verified agent experience into supervised datasets, for its clients. If
// your team needs expertise in dataset curation for fine-tuning, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Views: many training sets projected from one store of verified
//! experiences.
//!
//! A [`View`] reads a [`Corpus`] - experiences with their annotations, and
//! the tasks and sources beside them - and returns a [`Projection`]: the
//! [`Record`]s it yields for one [`Objective`], and how many candidates it
//! excluded for each [`Exclusion`] reason. Views never write to the store:
//! re-grading an experience (a new annotation) changes what a view yields,
//! while the experience stays as it was.
//!
//! | View | Objective | Projection |
//! |---|---|---|
//! | [`SftFinal`] | SFT | a passed experience's instruction and final answer |
//! | [`SftStep`] | SFT | each action of a trajectory, in the context before it |
//! | [`Critic`] | SFT | a task and candidate answer, and a verified critique of it |
//! | [`Preference`] | DPO | a chosen and a rejected answer to one task |
//! | [`VerifierView`] | classification | a task and candidate answer, and `pass` or `fail` |
//! | [`DecisionView`] | SFT | the passing action where a passing and a failed attempt part |
//! | [`Retrieval`] | contrastive | an instruction and the source span it is grounded in |
//! | [`OutcomeView`] | reward | a whole trajectory and its derived reward |
//! | [`DenoiseView`] | SFT | a denoise task's corrupted passage and its original |
//! | [`Cpt`] | continued pretraining | the raw text of source parts |
//!
//! Every conversation a view projects starts with [`SYSTEM_PROMPT`], the
//! system turn every model run on a task is sent: the policy is trained
//! under the prompt it answers under. What a student sees in its own turn
//! is decided by a [`Strip`] policy: by default only the instruction, never
//! what only the teacher saw - also when the experience is a teacher's
//! solve, prompted with the task's grounding material. A view drops
//! privileged context only from an instruction [`check_self_contained`]
//! accepts; see [`strip`](Strip) for the rules.
//!
//! [`write_dataset`] writes a projection in the one format its objective
//! maps to - brain's `generic-messages-v2` for chat records and
//! `generic-preference-v1` for preference pairs, Splinter's export-only
//! format on request for what brain cannot train - with a manifest beside
//! it.
//! [`DatasetStore`] keeps such datasets under the state root, each named by
//! its manifest's digest. [`replay_sample`] picks which earlier records are
//! replayed beside new ones.

#![warn(missing_docs)]

mod corpus;
mod dataset;
mod render;
mod replay;
mod store;
mod strip;
mod trajectory;
mod views;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use splinter_lab::{WireMessage, SYSTEM_PROMPT};
use splinter_record::annotation::{decide, Strength};
use splinter_record::digest::Digest;
use splinter_record::experience::{ExperienceError, ExperienceId};
use splinter_record::experiences::StoreError;

pub use corpus::{Corpus, Entry};
pub use dataset::{
    manifest_path, write_dataset, Counts, Dataset, Format, Manifest, WriteOptions, EXPORT_FORMAT,
};
pub use replay::replay_sample;
pub use store::{DatasetId, DatasetStore, StoredDataset, DATASET_FILE};
pub use strip::{
    check_self_contained, Fraction, NotSelfContained, Strip, MIN_QUOTED_CHARS, REFERRING_PHRASES,
};
pub use views::{
    Cpt, Critic, DecisionView, DenoiseView, OutcomeView, Preference, Retrieval, SftFinal, SftStep,
    VerifierView,
};

/// The training objective a view's records serve.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Objective {
    /// Supervised fine-tuning on chat records.
    Sft,
    /// A label predicted from an input. Rendered as SFT for now: the label
    /// is the supervised assistant turn, so brain's chat fine-tuning trains
    /// it, with no classification head.
    Classification,
    /// Direct preference optimisation on (chosen, rejected) pairs.
    Dpo,
    /// Contrastive retrieval on (query, positive, negatives).
    Contrastive,
    /// A whole trajectory with a scalar reward.
    Reward,
    /// Continued pretraining on raw text.
    Cpt,
}

impl Objective {
    /// The format brain's public SDK trains this objective from, when it
    /// has a trainer for it: chat fine-tuning trains SFT and classification
    /// rendered as SFT from [`Format::GenericMessagesV2`], preference
    /// fine-tuning trains DPO from [`Format::GenericPreferenceV1`]. brain
    /// has no trainer that reads contrastive triples for the policy model,
    /// rewarded trajectories or a raw text corpus as a dataset.
    #[must_use]
    pub fn brain_format(self) -> Option<Format> {
        match self {
            Self::Sft | Self::Classification => Some(Format::GenericMessagesV2),
            Self::Dpo => Some(Format::GenericPreferenceV1),
            Self::Contrastive | Self::Reward | Self::Cpt => None,
        }
    }

    /// Whether brain's public SDK trains this objective from a dataset
    /// file; see [`Objective::brain_format`].
    #[must_use]
    pub fn trainable_by_brain(self) -> bool {
        self.brain_format().is_some()
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
    },
    /// A shared prompt and two replies to it, the first preferred. DPO.
    Preference {
        /// The conversation before the reply.
        prompt: Vec<WireMessage>,
        /// The preferred reply.
        chosen: WireMessage,
        /// The reply it is preferred over.
        rejected: WireMessage,
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
                Objective::Sft | Objective::Classification
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
    /// The view that projected it.
    pub view: String,
    /// The objective it serves.
    pub objective: Objective,
}

/// Why a view left a candidate out. A view counts each candidate it
/// considers - an experience, a step, a relation, a pair, a source part,
/// as its documentation says - under the first reason that applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Exclusion {
    /// No pass/fail decision: no verdict, or the strongest ones conflict.
    Undecided,
    /// Decided below the view's minimum strength.
    TooWeak,
    /// Decided fail where the view needs a pass.
    Failed,
    /// No final output where the view needs one.
    NoFinalOutput,
    /// The instruction refers to privileged context the view would drop.
    NotSelfContained,
    /// A step labelled bad.
    BadStep,
    /// The trajectory holds something a chat record cannot represent (an
    /// image, a second user turn, an observation answering no call).
    Unrepresentable,
    /// No action to supervise or reward.
    NoAction,
    /// No reward could be derived.
    NoReward,
    /// A critique whose own decision is not a pass at the minimum strength.
    UnverifiedCritique,
    /// The experience a relation names is not in the corpus.
    RelatedMissing,
    /// A relation between experiences of different tasks.
    DifferentTask,
    /// Both sides of a pair gave the same output.
    IdenticalOutputs,
    /// A retry or revision chain without a pass on one side and a fail at
    /// the minimum strength on the other.
    NoPreferredPath,
    /// The two attempts part at no action both took a different one at.
    NoDivergence,
    /// No evidence span.
    NoEvidence,
    /// A source, part or content the store does not hold.
    SourceMissing,
    /// Content that is not UTF-8 text.
    NotText,
    /// Empty text.
    Empty,
    /// Content, a task or a pair already projected.
    Duplicate,
    /// A task with no single reference answer.
    NoReference,
}

/// A view's output: its records and what it left out.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Projection {
    /// The view's name.
    pub view: String,
    /// The objective its records serve.
    pub objective: Objective,
    /// The strip policy it applied; `None` for a view with no student
    /// input.
    pub strip: Option<Strip>,
    /// The minimum decision strength it required; `None` for a view that
    /// reads no verdict.
    pub min_strength: Option<Strength>,
    /// The records, in the corpus's order.
    pub records: Vec<Record>,
    /// Candidates left out, by reason.
    pub excluded: BTreeMap<Exclusion, usize>,
}

impl Projection {
    /// An empty projection of `view`.
    fn new(
        view: &str,
        objective: Objective,
        strip: Option<Strip>,
        min_strength: Option<Strength>,
    ) -> Self {
        Self {
            view: view.into(),
            objective,
            strip,
            min_strength,
            records: Vec::new(),
            excluded: BTreeMap::new(),
        }
    }

    /// Candidates excluded for `reason`.
    #[must_use]
    pub fn count(&self, reason: Exclusion) -> usize {
        self.excluded.get(&reason).copied().unwrap_or(0)
    }

    fn exclude(&mut self, reason: Exclusion) {
        *self.excluded.entry(reason).or_insert(0) += 1;
    }

    /// Adds a record with this projection's view and objective, a
    /// conversation starting with the system turn (see
    /// [`under_system_prompt`]).
    fn push(&mut self, body: RecordBody, provenance: Provenance) {
        self.records.push(Record {
            body: under_system_prompt(body),
            metadata: RecordMetadata {
                experiences: provenance.experiences,
                task: provenance.task,
                sources: provenance.sources,
                view: self.view.clone(),
                objective: self.objective,
            },
        });
    }

    /// Records `outcome`: its record, or its exclusion.
    fn take(&mut self, outcome: Result<(RecordBody, Provenance), Exclusion>) {
        match outcome {
            Ok((body, provenance)) => self.push(body, provenance),
            Err(reason) => self.exclude(reason),
        }
    }
}

/// `body` with [`SYSTEM_PROMPT`] as the first turn of its conversation -
/// a chat record's messages, a rewarded trajectory's, a preference pair's
/// prompt - the system turn every model run on a task is sent, so a record
/// shows the policy what it sees when it answers. Other shapes are not
/// conversations and are returned as they are.
fn under_system_prompt(body: RecordBody) -> RecordBody {
    let system = || render::message("system", SYSTEM_PROMPT, false);
    match body {
        RecordBody::Chat { mut messages } => {
            messages.insert(0, system());
            RecordBody::Chat { messages }
        }
        RecordBody::Rewarded {
            mut messages,
            reward,
        } => {
            messages.insert(0, system());
            RecordBody::Rewarded { messages, reward }
        }
        RecordBody::Preference {
            mut prompt,
            chosen,
            rejected,
        } => {
            prompt.insert(0, system());
            RecordBody::Preference {
                prompt,
                chosen,
                rejected,
            }
        }
        other @ (RecordBody::Contrastive { .. } | RecordBody::Text { .. }) => other,
    }
}

/// Where a record came from, before the view names itself on it.
#[derive(Clone, Debug, Default)]
struct Provenance {
    experiences: Vec<ExperienceId>,
    task: Option<Digest>,
    sources: Vec<Digest>,
}

impl Provenance {
    fn of(experiences: Vec<ExperienceId>) -> Self {
        Self {
            experiences,
            ..Self::default()
        }
    }
}

/// A projection of a corpus into training records for one objective.
pub trait View {
    /// The view's name, recorded in every record it yields.
    fn name(&self) -> &str;
    /// The objective its records serve.
    fn objective(&self) -> Objective;
    /// The records `corpus` yields, and what was left out.
    fn project(&self, corpus: &Corpus) -> Result<Projection, ViewError>;
}

/// Requires `entry` to be decided pass at `min_strength` or stronger.
fn require_pass(entry: &Entry, min_strength: Strength) -> Result<(), Exclusion> {
    let decision = decide(&entry.notes).ok_or(Exclusion::Undecided)?;
    if !decision.passed {
        return Err(Exclusion::Failed);
    }
    if decision.strength < min_strength {
        return Err(Exclusion::TooWeak);
    }
    Ok(())
}

/// Why a view or the dataset writer failed.
#[derive(Debug, thiserror::Error)]
pub enum ViewError {
    /// An annotation passed with an experience is about another one.
    #[error("an annotation of {annotation_of} was passed with experience {experience}")]
    ForeignAnnotation {
        /// The experience being added.
        experience: ExperienceId,
        /// The experience the annotation is about.
        annotation_of: ExperienceId,
    },
    /// An experience was added to a corpus twice.
    #[error("experience {0} is already in the corpus")]
    DuplicateExperience(ExperienceId),
    /// The experience's id cannot be computed.
    #[error(transparent)]
    Experience(#[from] ExperienceError),
    /// The experience or source store failed.
    #[error(transparent)]
    Store(#[from] StoreError),
    /// A view parameter is out of range.
    #[error("invalid view parameter {name}: {reason}")]
    Parameter {
        /// The parameter.
        name: &'static str,
        /// Why it is refused.
        reason: String,
    },
    /// No dataset is stored under this id.
    #[error("no dataset {0} in the store")]
    UnknownDataset(DatasetId),
    /// There are no records to write; a dataset that trains on nothing is
    /// a producer bug, not a dataset.
    #[error("no records to write")]
    Empty,
    /// Brain has no trainer for the objective, and the caller did not ask
    /// for an export-only file.
    #[error(
        "brain cannot train objective {objective:?}; write it with export_only to get \
         Splinter's export format instead"
    )]
    ObjectiveNotTrainable {
        /// The objective.
        objective: Objective,
    },
    /// A record does not have the shape its projection's objective needs.
    #[error("record {index} is a {shape} record, which does not serve objective {objective:?}")]
    Shape {
        /// The record's index.
        index: usize,
        /// Its shape.
        shape: &'static str,
        /// The projection's objective.
        objective: Objective,
    },
    /// A record or manifest cannot be serialized.
    #[error("cannot serialize a record: {0}")]
    Serialize(#[from] serde_json::Error),
    /// Brain's parser refused the dataset.
    #[error("{path} is refused by brain's dataset parser: {reason}")]
    Invalid {
        /// The dataset file.
        path: std::path::PathBuf,
        /// The parser's error.
        reason: String,
    },
    /// A file operation failed.
    #[error("{path}: {source}")]
    Io {
        /// The file.
        path: std::path::PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
}
