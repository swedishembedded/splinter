// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that run model learning as one
// resumable, multi-generation campaign with a single promotion gate, for its
// clients. If your team needs expertise in self-improving training loops you
// can procure our services by sending an email to info@swedishembedded.com.

//! The two stages of a generation that reach outside Splinter: the champion
//! practising (rollout) and the examiner's verdicts on a candidate
//! (evaluate). Each is a trait so a campaign runs the same over the real
//! examiner and over a stand-in; ingesting, curating, training, the gate and
//! the champion's moves are Splinter's own and are not seams.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use splinter_agent::CancelToken;
use splinter_core::model_ref::ModelRef;
use splinter_core::outcome::FailureClass;
use splinter_eval::promotion::{Pair, PublicKey};
use splinter_orchestrator::context::Context;

use super::types::CampaignError;
use crate::bench::practice::TaskFacts;

/// A model of the campaign's lineage: the base (generation 0) or a
/// candidate trained in a generation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Model {
    /// The generation that trained it; 0 for the base.
    pub generation: u32,
    /// How it is asked for, as a model reference: the base, or the base
    /// with the adapter (`local:<checkpoint>+<adapter>`).
    pub model: String,
    /// The candidate's id; `None` for the base.
    pub candidate: Option<String>,
    /// Its adapter's identity as the examiner names it (`sha256:<hex>` of
    /// the adapter file); `None` for the base served as it is.
    pub adapter: Option<String>,
}

impl Model {
    /// The model reference it is asked for by.
    ///
    /// # Errors
    /// A reference this build cannot read.
    pub fn reference(&self) -> Result<ModelRef, CampaignError> {
        self.model
            .parse()
            .map_err(|e| CampaignError::Config(format!("model {:?}: {e}", self.model)))
    }

    /// The pair the examiner compares: this model against `predecessor`.
    #[must_use]
    pub fn against(&self, predecessor: &Model) -> Pair {
        Pair {
            candidate: self.adapter.clone(),
            predecessor: predecessor.adapter.clone(),
        }
    }
}

/// One rollout stage: the champion attempts the tasks it has not mastered.
#[derive(Clone, Copy, Debug)]
pub struct RolloutRequest<'a> {
    /// The generation.
    pub generation: u32,
    /// Who practises.
    pub champion: &'a Model,
    /// The practice tasks, by name.
    pub tasks: &'a [String],
    /// Attempts of each.
    pub attempts: u32,
    /// The generation's rollout seed.
    pub seed: u64,
    /// Fresh transfer exercises to draw and practise beside `tasks`, when
    /// the capability curriculum asks for them.
    pub transfer: Option<TransferRequest<'a>>,
    /// The stage's own directory, inside the campaign's: everything the
    /// rollout writes goes here. It may hold what an interrupted attempt
    /// left; a rollout either continues it or starts over.
    pub dir: &'a Path,
}

/// Fresh exercises for the weakest capabilities, drawn from the examiner's
/// practice generator.
#[derive(Clone, Copy, Debug)]
pub struct TransferRequest<'a> {
    /// The capability ids, as the examiner names them.
    pub capabilities: &'a [String],
    /// Tasks drawn, of which those that exercise one of them are kept.
    pub count: usize,
    /// The draw's seed.
    pub seed: u64,
}

/// One trial of a rollout, as it ended.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Trial {
    /// The task.
    pub task: String,
    /// Its reward when the examiner graded it; `None` for a trial the
    /// infrastructure stopped, which counts for nothing.
    pub reward: Option<f64>,
    /// Whether it passed, when graded.
    pub passed: Option<bool>,
    /// Why it failed, when it failed and was diagnosed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class: Option<FailureClass>,
    /// The capability ids the examiner labels its task with; empty when it
    /// names none.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<String>,
    /// The share of its task's weighted checks it passed
    /// (`functional_correctness`), when the verifier measured it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correctness: Option<f64>,
    /// What its requests cost: completion plus uncached prompt tokens, when
    /// counted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<u64>,
    /// The completion tokens it generated, when counted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
}

/// What a rollout produced.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rolled {
    /// The examiner's export of the graded trials, inside the stage's
    /// directory: what the ingest stage admits.
    pub export: PathBuf,
    /// Every trial.
    pub trials: Vec<Trial>,
    /// Tokens the policy's requests processed (prompt and completion), summed
    /// over the trials whose usage was counted; `None` when none was.
    pub tokens: Option<u64>,
}

/// Where the champion practises.
pub trait Rollouts {
    /// What the rollouts are run over and how, by content (the material's
    /// digests and the settings), never by path or time: part of every
    /// rollout's identity.
    fn identity(&self) -> Value;

    /// The practice tasks there are, by name: the train split only.
    ///
    /// # Errors
    /// The material cannot be read.
    fn tasks(&self) -> Result<Vec<String>, CampaignError>;

    /// The capability ids of each practice task, by name, as the examiner
    /// labels them; a task it labels with none is absent. None by default.
    ///
    /// # Errors
    /// The material cannot be read.
    fn capabilities(&self) -> Result<BTreeMap<String, Vec<String>>, CampaignError> {
        Ok(BTreeMap::new())
    }

    /// What the material says of each practice task beyond its name, by
    /// name: its naive floor and its lineage. None by default.
    ///
    /// # Errors
    /// The material cannot be read.
    fn facts(&self) -> Result<BTreeMap<String, TaskFacts>, CampaignError> {
        Ok(BTreeMap::new())
    }

    /// The practice material's directory, when the rollouts run staged
    /// material: what a `shaped-partial` dataset reads its floors from.
    fn material(&self) -> Option<&Path> {
        None
    }

    /// Runs `request`; stops when `cancel` fires.
    ///
    /// # Errors
    /// The rollout could not be run, or was cancelled.
    fn rollout(
        &mut self,
        ctx: &Context,
        request: &RolloutRequest<'_>,
        cancel: &CancelToken,
    ) -> Result<Rolled, CampaignError>;
}

/// One evaluate stage: the examiner compares the candidate with the
/// champion it would replace.
#[derive(Clone, Copy, Debug)]
pub struct ExamRequest<'a> {
    /// The campaign, as the examiner's series names it.
    pub campaign: &'a str,
    /// The generation.
    pub generation: u32,
    /// The candidate.
    pub candidate: &'a Model,
    /// The champion it is compared with.
    pub predecessor: &'a Model,
    /// The sealed slice drawn for this generation, by the examiner's name.
    pub slice: &'a str,
    /// The training records the candidate learned from, for the examiner's
    /// overlap audit.
    pub training: &'a Path,
    /// The stage's own directory: the verdicts are written here.
    pub dir: &'a Path,
}

/// The examiner's three signed verdicts on a candidate, as files in the
/// stage's directory. They are read by the gate and by nothing else.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Examined {
    /// Over the development split.
    pub dev: PathBuf,
    /// Over the retention suite.
    pub retention: PathBuf,
    /// Over the generation's sealed slice.
    pub sealed: PathBuf,
}

/// Who examines candidates.
pub trait Examiner {
    /// What the examiner examines over and how, by content: part of every
    /// evaluation's identity.
    fn identity(&self) -> Value;

    /// The sealed slices drawn for the campaign, one per generation, by the
    /// examiner's names, in the order they are spent.
    fn slices(&self) -> Vec<String>;

    /// The examiner's public key, its verdicts' signatures are verified
    /// with.
    fn key(&self) -> &PublicKey;

    /// Runs `request`; stops when `cancel` fires.
    ///
    /// # Errors
    /// The examination could not be run, or was cancelled.
    fn examine(
        &mut self,
        ctx: &Context,
        request: &ExamRequest<'_>,
        cancel: &CancelToken,
    ) -> Result<Examined, CampaignError>;
}

/// Seconds elapsed on whatever clock measures GPU time: wall time in a run,
/// a pinned value in a test.
pub trait Stopwatch {
    /// Seconds since the clock started.
    fn seconds(&self) -> f64;
}

/// The wall clock as a [`Stopwatch`].
pub struct WallClock(std::time::Instant);

impl WallClock {
    /// A stopwatch started now.
    #[must_use]
    pub fn start() -> Self {
        Self(std::time::Instant::now())
    }
}

impl Stopwatch for WallClock {
    fn seconds(&self) -> f64 {
        self.0.elapsed().as_secs_f64()
    }
}
