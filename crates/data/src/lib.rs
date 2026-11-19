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
//! | [`Preference`] | preference | a chosen and a rejected continuation of one prompt: a verified pass and a verified fail of one task by one policy, or a repair against the failure it continues |
//! | [`Feedback`] | feedback (KTO) | a verified attempt's continuation of its prompt, desirable when it passed |
//! | [`VerifierView`] | classification | a task and candidate answer, and `pass` or `fail` |
//! | [`DecisionView`] | SFT | the passing action where a passing and a failed attempt part |
//! | [`Retrieval`] | contrastive | an instruction and the source span it is grounded in |
//! | [`OutcomeView`] | reward | a whole trajectory and its derived reward |
//! | [`PolicyGradient`] | policy gradient | each action of every graded attempt, with the attempt's reward and the group of attempts of its task it is compared with |
//! | [`DenoiseView`] | SFT | a denoise task's corrupted passage and its original |
//! | [`Cpt`] | continued pretraining | the raw text of source parts |
//! | [`Voice`] | SFT | the writer's own text, a stretch of whole sections at a time, as the answer |
//! | [`Rehearsal`] | SFT | a model's own answers as given, unless a verifier decided one wrong |
//!
//! Every conversation a view projects starts with [`SYSTEM_PROMPT`], the
//! system turn every model run on a task is sent: the policy is trained
//! under the prompt it answers under. What a student sees in its own turn
//! is decided by a [`Strip`] policy: by default only the instruction, never
//! what only the teacher saw - also when the experience is a teacher's
//! solve, prompted with the task's grounding material. A view drops
//! privileged context only from an instruction [`splinter_core::selfcontained::check_self_contained`]
//! accepts; see [`strip`](Strip) for the rules.
//!
//! [`write_dataset`] writes a projection in the one format its objective
//! maps to - brain's `generic-messages-v2` for chat records and
//! `generic-preference-v1` for preference pairs, Splinter's export-only
//! format on request for what brain cannot train - with a manifest beside
//! it.
//! [`DatasetStore`] keeps such datasets under the state root, each named by
//! its manifest's digest. [`replay_sample`] picks which earlier records are
//! replayed beside new ones. [`wire`] makes chat records straight from an
//! agent's recorded model calls, the request as sent and its reply
//! supervised, for episodes whose requests were captured on the wire.
//!
//! Record datasets (one subject per line, such as `timeline-v1`) are not
//! projected from experiences; what this crate decides for them is how they
//! are divided for evaluation: [`partition`] holds out a locked test and
//! lays out repeated grouped stratified cross-validation over the rest,
//! pinned by digest before anything trains. [`split`] adds the temporal and
//! leave-one-source-out splits and the leakage gates a split passes: no group
//! in two parts, a held-out source held out whole, nothing after a cutoff in
//! training, nothing fitted outside the training units; [`overlap`] finds
//! records of different parts that repeat each other's content.
//! [`timeline_dataset`]
//! projects longitudinal episodes (kept by `splinter-store`) into
//! participant-safe `timeline-v1` records at a prediction point and writes
//! them, and the parts of a split, with manifests.

#![warn(missing_docs)]

mod block_split;
pub mod conversation_pg;
mod corpus;
pub mod curation;
mod dataset;
mod exclusion;
pub mod frozen;
pub mod holdout;
pub mod images;
mod markers;
pub mod overlap;
pub mod partition;
pub mod pii;
pub mod process;
mod record;
mod render;
mod replay;
mod requests;
mod sampled;
mod scrub;
pub mod split;
mod store;
mod strip;
mod system_turn;
pub mod timeline_dataset;
pub mod timeline_store;
mod trajectory;
mod unprojectable;
mod views;
pub mod wire;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use splinter_core::annotation::Strength;
use splinter_core::dataset::DatasetId;
use splinter_core::digest::Digest;
use splinter_core::experience::{ExperienceError, ExperienceId};
use splinter_core::prompt::SYSTEM_PROMPT;
use splinter_store::decision::decide;
use splinter_store::experiences::StoreError;

pub use corpus::{Corpus, Entry};
pub use dataset::{
    dataset_line, manifest_path, named_tokenizers, write_dataset, Counts, Dataset, DatasetCheck,
    Format, Manifest, Unchecked, EXPORT_FORMAT,
};
pub use exclusion::Exclusion;
pub use record::{
    PairSource, Record, RecordBody, RecordMetadata, RolloutIdentity, RolloutOutcome, Side,
    SplitDraw, TeacherMessage,
};
pub use replay::{replay_sample, select_replay, ReplayPolicy};
pub use sampled::{ProxyReason, SampledTurn, SampledTurns, TokenReport};
pub use store::{DatasetStore, StoredDataset};
pub use strip::{Fraction, Strip};
pub use trajectory::{projection_refusal, session_dialogue};
pub use unprojectable::Unprojectable;
pub use views::signals;
pub use views::{
    chars_as_tokens, description, Control, Cpt, Critic, DecisionView, DenoiseView, DescriptionOf,
    Feedback, FeedbackBalance, OutcomeView, PairingPolicy, PolicyGradient, Preference, Rehearsal,
    Retrieval, Sectioner, SftFinal, SftStep, VerifierView, Voice, DEFAULT_MAX_FAMILY_SHARE,
    PAIRING_RULE,
};
pub use views::{
    AdvantageFloor, AdvantageKind, AdvantageNorm, Distill, GroupKind, GroupReport, Overlong,
    RewardShape, TrialReport,
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
    /// Preference optimisation on (chosen, rejected) pairs, by the pair loss
    /// a run names (DPO, IPO, SimPO, ORPO). Read as `dpo` too, the name a
    /// dataset built before the other losses records.
    #[serde(alias = "dpo")]
    Preference,
    /// KTO on judged completions: each one desirable or undesirable, no
    /// second completion needed.
    Feedback,
    /// Contrastive retrieval on (query, positive, negatives).
    Contrastive,
    /// A whole trajectory with a scalar reward.
    Reward,
    /// Reinforcement learning from rollouts: chat records that each carry the
    /// reward of the attempt they come from and its group, trained with a
    /// clipped policy gradient on the advantages the trainer computes.
    PolicyGradient,
    /// Continued pretraining on raw text.
    Cpt,
}

impl Objective {
    /// The line format this objective's records are written in: chat lines
    /// for SFT and classification (rendered as SFT) and for continued
    /// pretraining (a text is one supervised assistant turn), preference
    /// pairs for preference optimisation, judged completions for KTO, and
    /// Splinter's own export format for the objectives
    /// whose records no trainer reads as a dataset file (contrastive
    /// triples, rewarded trajectories). Whether a backend trains a
    /// format is that backend's to say.
    #[must_use]
    pub fn line_format(self) -> Format {
        match self {
            Self::Sft | Self::Classification | Self::PolicyGradient | Self::Cpt => {
                Format::GenericMessagesV2
            }
            Self::Preference => Format::GenericPreferenceV1,
            Self::Feedback => Format::GenericFeedbackV1,
            Self::Contrastive | Self::Reward => Format::SplinterExportV1,
        }
    }
}

/// Which of items of `sizes` to keep so that they hold at most `budget`
/// between them, evenly spread over the items: every item when they fit,
/// else the share of them the budget is of the total, taken at a steady
/// stride rather than from the front. The first item taken is kept even when
/// it alone is over the budget: a budget of something is never nothing.
pub(crate) fn spread(sizes: &[usize], budget: usize) -> Vec<bool> {
    let total: usize = sizes.iter().sum();
    if total <= budget {
        return vec![true; sizes.len()];
    }
    let share = budget as f64 / total as f64;
    let (mut owed, mut used) = (0.5f64, 0usize);
    let mut kept: Vec<bool> = sizes
        .iter()
        .map(|&size| {
            owed += share;
            let keep = owed >= 1.0 - 1e-9 && (used + size <= budget || used == 0);
            if keep {
                owed -= 1.0;
                used += size;
            }
            keep
        })
        .collect();
    if budget > 0 && !kept.contains(&true) {
        kept[sizes.len() / 2] = true;
    }
    kept
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
    /// The groups of attempts a policy-gradient view compared, with the
    /// reward of each attempt and whether it was written; empty for any
    /// other view.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<GroupReport>,
    /// How the records' rewards become advantages, for records that carry a
    /// reward and a group: the normalisation the trainer is asked for
    /// ([`AdvantageKind::norm`]). `None` for a view that writes no reward.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub advantage: Option<AdvantageKind>,
    /// Candidates left out, by reason.
    pub excluded: BTreeMap<Exclusion, usize>,
    /// How many rewarded actions carry the tokens the policy sampled, and why
    /// the others are trained against brain's start-of-run proxy; empty for a
    /// view that attaches none.
    #[serde(default, skip_serializing_if = "TokenReport::is_empty")]
    pub tokens: TokenReport,
    /// The system prompt every conversation opens with when it is not the
    /// default ([`SYSTEM_PROMPT`]): a person's, for a policy trained to be
    /// them. A dataset records it, so what a model was trained under is
    /// what it is asked under.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    /// Whether each record holds a whole trajectory with several supervised
    /// turns ([`SftStep::packed`]), which training must keep as one sequence.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub packed: bool,
    /// Whether every record comes from an experience an independent verdict
    /// passed ([`SftStep::outcome_verified_only`]); without it an action
    /// labelled good is admitted from an unverified or failed experience.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub outcome_verified_only: bool,
    /// The terms the records' data came under, combined over their sources
    /// ([`splinter_core::terms::combine_stated`]); `None` when no source
    /// stated any. Set by whoever resolved the sources, never by a view.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terms: Option<splinter_core::terms::Terms>,
    /// How [`Projection::within_block`] / [`Projection::split_over_block`]
    /// measured the records, when the caller bounded them; `None` when it did
    /// not. Set by the caller that supplied the measure, so a build reports
    /// whether the bound is the trainer's own count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_measure: Option<BlockMeasure>,
    /// The curation policy the view applied to its demonstrations
    /// ([`curation`]); `None` when it applied none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub curation: Option<curation::Curation>,
    /// The over-length penalty the records' rewards carry ([`Overlong`]);
    /// `None` when they carry none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overlong: Option<Overlong>,
    /// How a preference view chose its pairs ([`PairingPolicy`]); `None` for
    /// any other view.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pairing: Option<PairingPolicy>,
    /// The classes of a feedback view's records and their weights
    /// ([`FeedbackBalance`]); `None` for any other view.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feedback: Option<FeedbackBalance>,
}

/// What a record's size was counted by when a projection was bounded to a
/// training block.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockMeasure {
    /// The trainer's own count: the dataset line the record is written as,
    /// read and encoded by the training backend's chat reader. A record
    /// fitting the bound by it is not refused by the trainer's length check.
    Trainer {
        /// The checkpoint whose tokenizer, chat template and image
        /// preprocessing counted the records: the base the training starts
        /// from.
        #[serde(default)]
        base: std::path::PathBuf,
        /// The first reason a record could not be measured, if any could not:
        /// the trainer would refuse it too, so it was left out.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        first_refusal: Option<String>,
    },
    /// The record's JSON text through a tokenizer (else a character ratio)
    /// plus image rows: a floor on the trainer's count, which also counts the
    /// chat template's markup. A record fitting the bound by it can still be
    /// refused by the trainer.
    Floor {
        /// Why the trainer's count was not used.
        reason: String,
    },
}

impl Projection {
    /// An empty projection of `view`, for a producer outside this crate that
    /// builds its records itself.
    #[must_use]
    pub fn new(
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
            groups: Vec::new(),
            advantage: None,
            excluded: BTreeMap::new(),
            tokens: TokenReport::default(),
            system_prompt: None,
            packed: false,
            outcome_verified_only: false,
            terms: None,
            block_measure: None,
            curation: None,
            overlong: None,
            pairing: None,
            feedback: None,
        }
    }

    /// The same projection with `prompt` as the first turn of every
    /// conversation in place of the default - inserted where a conversation
    /// has no system turn - and recorded as the prompt the projection trains
    /// under.
    #[must_use]
    pub fn with_system_prompt(mut self, prompt: &str) -> Self {
        for record in &mut self.records {
            let messages = match &mut record.body {
                RecordBody::Chat { messages, .. } | RecordBody::Rewarded { messages, .. } => {
                    Some(messages)
                }
                RecordBody::Preference { prompt, .. } | RecordBody::Feedback { prompt, .. } => {
                    Some(prompt)
                }
                _ => None,
            };
            let Some(messages) = messages else { continue };
            match messages.first_mut().filter(|m| m.role == "system") {
                Some(first) => first.content = prompt.to_string(),
                None => messages.insert(0, render::message("system", prompt, false)),
            }
        }
        self.system_prompt = Some(prompt.to_string());
        self
    }

    /// The same projection with the system prompt its records open with
    /// recorded as the prompt it trains under, when it has none recorded and
    /// they all open with the same prompt that is not the default: a dataset
    /// of runs made under the coding-agent prompt says so, and a model trained
    /// on it is asked under it. Refused when its records open with different
    /// prompts, which only [`Projection::with_system_prompt`] can settle: a
    /// model cannot be trained to answer under two.
    pub fn settled_system_prompt(mut self) -> Result<Self, ViewError> {
        if self.system_prompt.is_some() {
            return Ok(self);
        }
        let mut opening: Vec<&str> = Vec::new();
        for record in &self.records {
            let messages = match &record.body {
                RecordBody::Chat { messages, .. } | RecordBody::Rewarded { messages, .. } => {
                    messages
                }
                RecordBody::Preference { prompt, .. } | RecordBody::Feedback { prompt, .. } => {
                    prompt
                }
                _ => continue,
            };
            if let Some(first) = messages.first().filter(|m| m.role == "system") {
                if !opening.contains(&first.content.as_str()) {
                    opening.push(&first.content);
                }
            }
        }
        match opening[..] {
            [] => Ok(self),
            [only] => {
                if only != SYSTEM_PROMPT {
                    self.system_prompt = Some(only.to_string());
                }
                Ok(self)
            }
            _ => Err(ViewError::Parameter {
                name: "system prompt",
                reason: format!(
                    "the records open with {} different system prompts (the runs were made under \
                     different ones); pass a system prompt to write them all under one",
                    opening.len()
                ),
            }),
        }
    }

    /// The same projection thinned to at most `limit` records, kept evenly
    /// spread over it - the first and the last among them - and the rest
    /// counted as [`Exclusion::OverLimit`].
    #[must_use]
    pub fn thinned_to(mut self, limit: usize) -> Self {
        let total = self.records.len();
        if limit >= total {
            return self;
        }
        let kept: std::collections::BTreeSet<usize> = match limit {
            0 => std::collections::BTreeSet::new(),
            1 => std::iter::once(0).collect(),
            _ => (0..limit).map(|i| i * (total - 1) / (limit - 1)).collect(),
        };
        let records = std::mem::take(&mut self.records);
        self.records = records
            .into_iter()
            .enumerate()
            .filter_map(|(n, record)| kept.contains(&n).then_some(record))
            .collect();
        *self.excluded.entry(Exclusion::OverLimit).or_insert(0) += total - self.records.len();
        self
    }

    /// The same projection thinned to at most `budget` tokens as `size`
    /// counts them, kept evenly spread over it: a budget reaches the whole
    /// of what the view projects rather than its first part. The records dropped are counted
    /// as [`Exclusion::OverLimit`].
    #[must_use]
    pub fn thinned_to_tokens(mut self, budget: usize, size: &dyn Fn(&Record) -> usize) -> Self {
        let sizes: Vec<usize> = self.records.iter().map(size).collect();
        let kept = spread(&sizes, budget);
        let total = self.records.len();
        let records = std::mem::take(&mut self.records);
        self.records = records
            .into_iter()
            .zip(kept)
            .filter_map(|(record, keep)| keep.then_some(record))
            .collect();
        *self.excluded.entry(Exclusion::OverLimit).or_insert(0) += total - self.records.len();
        self
    }

    /// The same projection without the records longer than `max_block`
    /// tokens as `size` counts them, which the trainer would refuse mid-run;
    /// they are counted as [`Exclusion::OverBlock`]. `size` sees the record
    /// as it stands, so a system prompt set before this call is counted; a
    /// bound set to a trainer's limit needs the trainer's own count as
    /// `size` ([`BlockMeasure::Trainer`]), not a floor on it.
    #[must_use]
    pub fn within_block(mut self, max_block: usize, size: &dyn Fn(&Record) -> usize) -> Self {
        let total = self.records.len();
        self.records.retain(|record| size(record) <= max_block);
        let dropped = total - self.records.len();
        if dropped > 0 {
            *self.excluded.entry(Exclusion::OverBlock).or_insert(0) += dropped;
        }
        self
    }

    /// The same projection with every record drawn by `seed` in the stratum
    /// `stratum_of` gives it (`metadata.split_draw`), so that the holdout rule
    /// splits it by seed and stratum ([`holdout`]) and its manifest records
    /// how ([`holdout::SplitPolicy`]).
    #[must_use]
    pub fn drawn_by(mut self, seed: u64, stratum_of: &dyn Fn(&Record) -> Option<String>) -> Self {
        for record in &mut self.records {
            record.metadata.split_draw = Some(SplitDraw {
                seed,
                stratum: stratum_of(record),
            });
        }
        self
    }

    /// How the records' split is drawn, when they carry a draw: the seed of
    /// the first that does, and the records of each stratum.
    #[must_use]
    pub fn split_policy(&self) -> Option<holdout::SplitPolicy> {
        let seed = self
            .records
            .iter()
            .find_map(|r| r.metadata.split_draw.as_ref())?
            .seed;
        let mut strata = BTreeMap::new();
        for draw in self
            .records
            .iter()
            .filter_map(|r| r.metadata.split_draw.as_ref())
        {
            let name = draw.stratum.as_deref().unwrap_or(holdout::NO_STRATUM);
            *strata.entry(name.to_string()).or_insert(0) += 1;
        }
        Some(holdout::SplitPolicy {
            rule: holdout::SEEDED_STRATIFIED.into(),
            seed,
            strata,
        })
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
    /// [`system_turn::under_system_prompt`]).
    fn push(&mut self, body: RecordBody, provenance: Provenance) {
        self.records.push(Record {
            body: system_turn::under_system_prompt(body),
            metadata: RecordMetadata {
                experiences: provenance.experiences,
                task: provenance.task,
                sources: provenance.sources,
                group: None,
                split: None,
                split_draw: None,
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
    /// A recorded model call cannot be made a training record.
    #[error("a recorded model call cannot be a record: {0}")]
    Call(String),
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
    #[error("no records to write (left out: {excluded})")]
    Empty {
        /// What the projection left out and why, as `reason: count` pairs.
        excluded: String,
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
    /// A dataset too small to hold a record out.
    #[error(
        "{path} holds {records} record(s); scoring needs at least {} so one can be held out",
        crate::holdout::MIN_SAMPLES
    )]
    TooFewRecords {
        /// The dataset file.
        path: std::path::PathBuf,
        /// Records in it.
        records: usize,
    },
    /// No whole group can be held out: the records come from too few
    /// independent source texts, and dividing one would train on a print of
    /// what is scored.
    #[error(
        "{records} record(s) come from {groups} independent source text(s) and none can be held out whole; \
         a held-out measurement needs at least two, so add sources"
    )]
    NoGroupToHoldOut {
        /// Records in the dataset.
        records: usize,
        /// The distinct groups they fall in; an ungrouped record is its own.
        groups: usize,
    },
    /// The dataset's validator refused it.
    #[error("{path} is refused by the dataset validator: {reason}")]
    Invalid {
        /// The dataset file.
        path: std::path::PathBuf,
        /// The validator's error.
        reason: String,
    },
    /// A split or a dataset about to be written fails a leakage gate.
    #[error(transparent)]
    Leakage(#[from] split::LeakageError),
    /// The records name two tokenizers: the attempts were encoded by two
    /// models' servers, and one dataset holds the attempts of one.
    #[error(
        "the records name two tokenizers, {first} and {second}: the attempts of one dataset are \
         encoded by one tokenizer, so build one dataset per policy"
    )]
    MixedTokenizerDigest {
        /// The tokenizer the first record names.
        first: String,
        /// Another tokenizer a later record names.
        second: String,
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

#[cfg(test)]
mod thinning_tests;
