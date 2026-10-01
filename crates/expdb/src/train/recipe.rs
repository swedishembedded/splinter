// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A recipe: what training data is wanted, as data.

use serde::{Deserialize, Serialize};

/// The learning objective a plan is shaped for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Objective {
    /// Imitate the steps of successful attempts.
    Sft,
    /// Prefer the better of two decisions taken at one state.
    Dpo,
    /// Compare the attempts of one family against each other.
    Grpo,
    /// Learn from single transitions.
    Ppo,
    /// Learn a score for each step.
    Prm,
}

/// What is wanted from the experience. Settings an objective does not use
/// are ignored by it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Recipe {
    /// The objective.
    pub objective: Objective,
    /// Only attempts by this policy name and version.
    pub policy: Option<(String, String)>,
    /// Only attempts whose standing evaluations are at least this confident.
    pub min_confidence: Option<f64>,
    /// Only attempts a fact-level evaluation vouches for.
    pub verified_only: bool,
    /// At most this many samples from one family.
    pub max_per_family: Option<usize>,
    /// At most this many samples in all, chosen by the seed.
    pub max_samples: Option<usize>,
    /// Fixes every random choice.
    pub seed: u64,
    /// SFT: the least reward an attempt may have stood at.
    pub min_reward: f64,
    /// SFT: whole episodes instead of single steps.
    pub episodes: bool,
    /// DPO: the least reward difference worth a pair.
    pub min_gap: f64,
    /// DPO: rank by this algorithm's credit instead of the outcome.
    pub credit: Option<String>,
    /// GRPO: attempts per group, exactly.
    pub group_size: usize,
    /// GRPO: the least reward variance in a group.
    pub min_variance: f64,
    /// GRPO: only groups where some attempts pass and some fail.
    pub require_mixed: bool,
    /// PPO: give a final transition the attempt's reward if it has none.
    pub terminal_reward: bool,
    /// PRM: the evaluation criterion that labels steps.
    pub criterion: String,
}

impl Recipe {
    fn new(objective: Objective) -> Self {
        Self {
            objective,
            policy: None,
            min_confidence: None,
            verified_only: false,
            max_per_family: None,
            max_samples: None,
            seed: 0,
            min_reward: 1.0,
            episodes: false,
            min_gap: 0.5,
            credit: None,
            group_size: 8,
            min_variance: 0.0,
            require_mixed: true,
            terminal_reward: true,
            criterion: "reasoning_quality".into(),
        }
    }

    /// Imitation of successful attempts.
    pub fn sft() -> Self {
        Self::new(Objective::Sft)
    }

    /// Preferences between decisions taken at one state.
    pub fn dpo() -> Self {
        Self::new(Objective::Dpo)
    }

    /// Groups of attempts at one instance, compared against each other.
    pub fn grpo() -> Self {
        Self::new(Objective::Grpo)
    }

    /// Single transitions.
    pub fn ppo() -> Self {
        Self::new(Objective::Ppo)
    }

    /// Step labels from evaluations.
    pub fn prm() -> Self {
        Self::new(Objective::Prm)
    }

    /// Only attempts by this policy.
    pub fn policy(mut self, name: &str, version: &str) -> Self {
        self.policy = Some((name.into(), version.into()));
        self
    }

    /// Only attempts at least this confidently evaluated.
    pub fn min_confidence(mut self, confidence: f64) -> Self {
        self.min_confidence = Some(confidence);
        self
    }

    /// Only attempts vouched for by a fact-level evaluation.
    pub fn verified_only(mut self) -> Self {
        self.verified_only = true;
        self
    }

    /// At most `n` samples from any one family.
    pub fn max_per_family(mut self, n: usize) -> Self {
        self.max_per_family = Some(n);
        self
    }

    /// At most `n` samples, the seed choosing which.
    pub fn max_samples(mut self, n: usize) -> Self {
        self.max_samples = Some(n);
        self
    }

    /// Fixes every random choice.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// SFT: the least reward an attempt may have stood at.
    pub fn min_reward(mut self, reward: f64) -> Self {
        self.min_reward = reward;
        self
    }

    /// SFT: whole episodes, not steps.
    pub fn episodes(mut self) -> Self {
        self.episodes = true;
        self
    }

    /// DPO: the least reward difference worth a pair.
    pub fn min_gap(mut self, gap: f64) -> Self {
        self.min_gap = gap;
        self
    }

    /// DPO: rank by an algorithm's credit instead of the outcome.
    pub fn credit(mut self, algorithm: &str) -> Self {
        self.credit = Some(algorithm.into());
        self
    }

    /// GRPO: attempts per group, exactly.
    pub fn group_size(mut self, n: usize) -> Self {
        self.group_size = n;
        self
    }

    /// GRPO: the least reward variance in a group.
    pub fn min_reward_variance(mut self, variance: f64) -> Self {
        self.min_variance = variance;
        self
    }

    /// GRPO: accept groups where every attempt did the same.
    pub fn allow_uniform(mut self) -> Self {
        self.require_mixed = false;
        self
    }

    /// PPO: leave a final transition's reward as the environment gave it.
    pub fn without_terminal_reward(mut self) -> Self {
        self.terminal_reward = false;
        self
    }

    /// PRM: the criterion that labels steps.
    pub fn criterion(mut self, criterion: &str) -> Self {
        self.criterion = criterion.into();
        self
    }
}
