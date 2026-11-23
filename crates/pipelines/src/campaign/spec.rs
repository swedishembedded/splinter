// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that run model learning as one
// resumable, multi-generation campaign with a single promotion gate, for its
// clients. If your team needs expertise in self-improving training loops you
// can procure our services by sending an email to info@swedishembedded.com.

//! What a campaign is: its generations, its base, how each stage is run and
//! what it may spend. A campaign is continued as it began: everything here
//! but the budget is part of its identity, and a changed specification is
//! refused (start a new campaign to change it).

use serde::{Deserialize, Serialize};
use splinter_core::model_ref::ModelRef;

use crate::datasets::{RewardPreset, ViewName};
use crate::served::ServedError;
use crate::train::Recipe;

/// The tasks a champion has mastered at this pass rate are not rolled out
/// again: [`splinter_eval::campaign::pass_rate::RETIRE_AT`].
pub const DEFAULT_RETIRE_AT: f64 = splinter_eval::campaign::pass_rate::RETIRE_AT;

/// A campaign.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignSpec {
    /// Its name: letters, digits, `-`, `_` and `.`.
    pub name: String,
    /// Generations to run. Not part of its identity: a campaign that ran its
    /// generations continues when it is given more.
    pub generations: u32,
    /// The base every candidate is trained from: a `local:` checkpoint. It
    /// is generation 0's champion unless `champion` names another.
    pub base: String,
    /// Generation 0's champion when it is not the base: an adapter on the
    /// base (`local:<base>+<adapter>`), for a campaign that continues from a
    /// model trained before it. Candidates are still trained from the base.
    #[serde(default)]
    pub champion: Option<String>,
    /// The seed every generation's draws derive from; drawn, printed and
    /// recorded in the ledger when left out, and read back from it on resume.
    /// Written as an integer, or as a string (decimal or `0x` hex) for a
    /// seed above what a TOML integer holds.
    #[serde(default, deserialize_with = "seed_text")]
    pub seed: Option<u64>,
    /// How the champion practises.
    #[serde(default)]
    pub rollout: RolloutSpec,
    /// How the experience becomes a dataset.
    #[serde(default)]
    pub curate: CurateSpec,
    /// How a candidate is trained.
    #[serde(default)]
    pub train: TrainSpec,
    /// What the campaign may spend. Not part of its identity: a campaign
    /// stopped by its budget continues when it is raised.
    #[serde(default)]
    pub budget: Budget,
    /// Where the champion practises, for `splinter campaign`: the staged
    /// practice material. Its content, not its path, is part of the identity
    /// (through the rollouts it makes).
    #[serde(default)]
    pub practice: Option<super::practice::PracticeSpec>,
    /// Who examines, for `splinter campaign`: bench's checkout, key and
    /// collections.
    #[serde(default)]
    pub examine: Option<super::examiner::ExamineSpec>,
}

/// The rollout stage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RolloutSpec {
    /// Attempts of each task.
    pub attempts: u32,
    /// The pass rate at which a task counts as mastered by the champion and
    /// is not rolled out again.
    pub retire_at: f64,
    /// The fewest graded attempts a pass rate must rest on before a task is
    /// retired.
    pub retire_after: usize,
    /// How a generation's tasks are chosen.
    #[serde(default, skip_serializing_if = "RolloutCurriculum::is_all")]
    pub curriculum: RolloutCurriculum,
    /// With the capability or signal curriculum: the most tasks a
    /// generation practises; every task of positive weight when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tasks: Option<usize>,
    /// With the capability curriculum: fresh transfer exercises drawn from
    /// the examiner's practice generator for the weakest capabilities.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transfer: Option<TransferSpec>,
}

impl Default for RolloutSpec {
    fn default() -> Self {
        Self {
            attempts: 4,
            retire_at: DEFAULT_RETIRE_AT,
            retire_after: 4,
            curriculum: RolloutCurriculum::All,
            tasks: None,
            transfer: None,
        }
    }
}

/// How a generation's practice tasks are chosen.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RolloutCurriculum {
    /// Every practice task the champion has not mastered.
    #[default]
    All,
    /// Tasks drawn by capability gap: weighted by their capabilities'
    /// failure rate, frontier-band membership and share of the model's own
    /// failures; never a task the infrastructure stopped every time
    /// ([`crate::curriculum::capability`]).
    Capability,
    /// Tasks drawn by variance per cost: in proportion to the square root of
    /// their shrunk reward variance (under the curate stage's reward) over
    /// the mean tokens an attempt costs; never a task whose graded attempts
    /// all earned the same, nor one the infrastructure stopped every time
    /// ([`crate::curriculum::signal`]).
    Signal,
}

impl RolloutCurriculum {
    fn is_all(&self) -> bool {
        *self == Self::All
    }
}

/// Fresh transfer exercises for the weakest capabilities.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransferSpec {
    /// How many of the weakest measured capabilities exercises are drawn
    /// for.
    pub capabilities: usize,
    /// Tasks the examiner draws from its generator per generation, of which
    /// it keeps those that exercise one of them.
    pub count: usize,
}

/// The curate stage: every generation's dataset is built from the
/// experience of every generation so far.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurateSpec {
    /// The view the dataset is projected through.
    pub view: ViewName,
    /// The longest record in tokens; `None` keeps every record.
    #[serde(default)]
    pub max_block: Option<usize>,
    /// For the sft-step view: one record per trajectory.
    #[serde(default)]
    pub pack: bool,
    /// For the policy-gradient view: the named reward; `shaped-partial`
    /// reads the naive floors of the practice material
    /// ([`crate::datasets::RewardPreset`]). The signal curriculum measures
    /// attempts under it too.
    #[serde(default, skip_serializing_if = "is_verdict")]
    pub reward: RewardPreset,
}

fn is_verdict(reward: &RewardPreset) -> bool {
    *reward == RewardPreset::Verdict
}

impl Default for CurateSpec {
    fn default() -> Self {
        Self {
            view: ViewName::SftStep,
            max_block: None,
            pack: false,
            reward: RewardPreset::Verdict,
        }
    }
}

/// The train stage: each candidate is trained from the base on the
/// generation's cumulative dataset.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainSpec {
    /// LoRA rank.
    pub rank: u32,
    /// The step budget; `None` sizes it to the dataset.
    #[serde(default)]
    pub steps: Option<u32>,
    /// The peak learning rate; `None` is the trainer's default.
    #[serde(default)]
    pub learning_rate: Option<f32>,
    /// Records averaged into one step; `None` sizes it to the dataset.
    #[serde(default)]
    pub records_per_step: Option<u32>,
    /// The longest training row in tokens.
    #[serde(default)]
    pub max_block: Option<u32>,
    /// A named recipe whose defaults fill what is left unset
    /// ([`crate::train::Recipe`]); for a policy-gradient view only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipe: Option<Recipe>,
}

impl Default for TrainSpec {
    fn default() -> Self {
        Self {
            rank: 16,
            steps: None,
            learning_rate: None,
            records_per_step: None,
            max_block: None,
            recipe: None,
        }
    }
}

/// What a campaign may spend, summed over its generations; an absent limit
/// is no limit. A stage that would exceed one does not start.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    /// Rollouts (attempts of a task).
    #[serde(default)]
    pub rollouts: Option<u64>,
    /// GPU-hours of training.
    #[serde(default)]
    pub gpu_hours: Option<f64>,
    /// Tokens the policy generated in its rollouts.
    #[serde(default)]
    pub tokens: Option<u64>,
}

/// A seed as a TOML integer or as text: a TOML integer is signed, so a
/// drawn seed above `i64::MAX` is written as a string.
fn seed_text<'de, D: serde::Deserializer<'de>>(de: D) -> Result<Option<u64>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Written {
        Number(u64),
        Text(String),
    }
    match Option::<Written>::deserialize(de)? {
        None => Ok(None),
        Some(Written::Number(n)) => Ok(Some(n)),
        Some(Written::Text(text)) => text
            .parse::<splinter_core::rng::Seed>()
            .map(|s| Some(s.value()))
            .map_err(serde::de::Error::custom),
    }
}

impl CampaignSpec {
    /// The specification in `text` (TOML).
    ///
    /// # Errors
    /// Not TOML of this shape, or [`Self::validate`].
    pub fn parse(text: &str) -> Result<Self, ServedError> {
        let spec: Self = toml::from_str(text)
            .map_err(|e| ServedError::Config(format!("campaign specification: {e}")))?;
        spec.validate()?;
        Ok(spec)
    }

    /// The base as a model reference.
    ///
    /// # Errors
    /// Not a `local:` reference.
    pub fn base_ref(&self) -> Result<ModelRef, ServedError> {
        let base: ModelRef = self
            .base
            .parse()
            .map_err(|e| ServedError::Config(format!("base {:?}: {e}", self.base)))?;
        match &base {
            ModelRef::Local { adapter: None, .. } => Ok(base),
            _ => Err(ServedError::Config(format!(
                "base {:?}: a campaign trains from a local checkpoint with no adapter",
                self.base
            ))),
        }
    }

    /// Generation 0's champion as a model reference: `champion`, or the
    /// base when there is none.
    ///
    /// # Errors
    /// The base's, or a champion that is not an adapter on the base.
    pub fn champion_ref(&self) -> Result<ModelRef, ServedError> {
        let base = self.base_ref()?;
        let Some(champion) = &self.champion else {
            return Ok(base);
        };
        let parsed: ModelRef = champion
            .parse()
            .map_err(|e| ServedError::Config(format!("champion {champion:?}: {e}")))?;
        match (&parsed, &base) {
            (
                ModelRef::Local {
                    checkpoint,
                    adapter: Some(_),
                    ..
                },
                ModelRef::Local {
                    checkpoint: base_checkpoint,
                    ..
                },
            ) if checkpoint == base_checkpoint => Ok(parsed),
            _ => Err(ServedError::Config(format!(
                "champion {champion:?}: generation 0's champion is an adapter on the base {:?} \
                 (local:<base>+<adapter>)",
                self.base
            ))),
        }
    }

    /// Refuses a specification no campaign can run.
    ///
    /// # Errors
    /// The first setting out of range, named.
    pub fn validate(&self) -> Result<(), ServedError> {
        let refuse = |why: String| Err(ServedError::Config(why));
        let sound = !self.name.is_empty()
            && self.name.len() <= 64
            && !self.name.starts_with('.')
            && self
                .name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c));
        if !sound {
            return refuse(format!(
                "campaign name {:?}: 1 to 64 letters, digits, `-`, `_` and `.`, not starting with a dot",
                self.name
            ));
        }
        if self.generations == 0 {
            return refuse("generations: at least one".into());
        }
        if self.rollout.attempts == 0 {
            return refuse("rollout.attempts: at least one".into());
        }
        if !(self.rollout.retire_at > 0.0 && self.rollout.retire_at <= 1.0) {
            return refuse(format!(
                "rollout.retire_at is {}, not in (0, 1]",
                self.rollout.retire_at
            ));
        }
        let curriculum = self.rollout.curriculum;
        if curriculum == RolloutCurriculum::All && self.rollout.tasks.is_some() {
            return refuse(
                "rollout.tasks applies to curriculum = \"capability\" or \"signal\"".into(),
            );
        }
        if curriculum != RolloutCurriculum::Capability && self.rollout.transfer.is_some() {
            return refuse("rollout.transfer applies to curriculum = \"capability\"".into());
        }
        if self.curate.reward != RewardPreset::Verdict
            && !matches!(
                self.curate.view,
                ViewName::PolicyGradient | ViewName::SelfDistill
            )
        {
            return refuse(format!(
                "curate.reward applies to a view that rewards attempts, not {}",
                self.curate.view.as_str()
            ));
        }
        if self.rollout.tasks == Some(0) {
            return refuse("rollout.tasks: at least one".into());
        }
        if self
            .rollout
            .transfer
            .is_some_and(|t| t.capabilities == 0 || t.count == 0)
        {
            return refuse("rollout.transfer: at least one capability and one task".into());
        }
        if self.train.rank == 0 {
            return refuse("train.rank: at least one".into());
        }
        if self.budget.gpu_hours.is_some_and(|h| h.is_nan() || h < 0.0) {
            return refuse("budget.gpu_hours: not negative".into());
        }
        self.champion_ref().map(|_| ())
    }

    /// The part of the specification that is the campaign's identity: all
    /// of it but the generations and the budget, which say how far it may
    /// go, the seed, which the ledger records, and the practice and examine
    /// sections, whose content the rollouts and the examiner state by their
    /// own identity.
    #[must_use]
    pub fn identity(&self) -> serde_json::Value {
        let mut identity = serde_json::json!({
            "name": self.name,
            "base": self.base,
            "rollout": self.rollout,
            "curate": self.curate,
            "train": self.train,
        });
        // Only a campaign that names one carries it, so the campaigns begun
        // before keep their identity.
        if let Some(champion) = &self.champion {
            identity["champion"] = champion.clone().into();
        }
        identity
    }
}
