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
    /// Predict the future of the streams from their past and an action.
    WorldModel,
    /// Align two modalities of one moment, against nearby and far moments.
    Contrastive,
    /// Predict one hidden stream from everything around it.
    Masked,
    /// Produce the actions that follow, from observations and an instruction.
    ActionChunk,
    /// Follow a recorded relation between attempts.
    Relation,
}

/// What a multimodal objective windows over. Times are nanoseconds on the
/// episode's clock.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MmSpec {
    /// Observation streams by name; empty means every recorded stream.
    pub streams: Vec<String>,
    /// How much of the past is context.
    pub past_ns: i64,
    /// How much of the future is the target.
    pub future_ns: i64,
    /// The spacing of anchors when no action sets them.
    pub stride_ns: i64,
    /// The width of a contrastive or masked window.
    pub window_ns: i64,
    /// Contrastive: the two modalities, by stream name.
    pub pair: Option<(String, String)>,
    /// Contrastive: how far in time a hard negative is taken.
    pub hard_shift_ns: i64,
    /// Contrastive: negatives per sample, the hard one included.
    pub negatives: usize,
    /// Masked: the stream that is hidden.
    pub target_stream: Option<String>,
    /// Masked: how much of the surroundings is shown.
    pub context_ns: i64,
    /// Action chunk: how far ahead the actions reach.
    pub horizon_ns: i64,
    /// Action chunk: the event that carries the instruction.
    pub instruction_event: String,
}

impl Default for MmSpec {
    fn default() -> Self {
        const SECOND: i64 = 1_000_000_000;
        Self {
            streams: Vec::new(),
            past_ns: SECOND,
            future_ns: SECOND,
            stride_ns: SECOND,
            window_ns: SECOND,
            pair: None,
            hard_shift_ns: 5 * SECOND,
            negatives: 2,
            target_stream: None,
            context_ns: SECOND,
            horizon_ns: SECOND,
            instruction_event: "instruction".into(),
        }
    }
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
    /// Multimodal objectives: what to window over.
    pub mm: MmSpec,
    /// Only attempts whose evidence decides at this rank or stronger.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_rank: Option<u8>,
    /// DPO: pair the passing and failing attempts of one task instead of
    /// the decisions taken at one state.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub by_task: bool,
    /// DPO by task: at most this many pairs from one task.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_pairs_per_task: Option<usize>,
    /// Relation: the relation to follow.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relation: Option<crate::model::Rel>,
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
            mm: MmSpec::default(),
            min_rank: None,
            by_task: false,
            max_pairs_per_task: None,
            relation: None,
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

    /// Predicting the future of streams from their past and an action.
    pub fn world_model() -> Self {
        Self::new(Objective::WorldModel)
    }

    /// Aligning two modalities.
    pub fn contrastive() -> Self {
        Self::new(Objective::Contrastive)
    }

    /// Predicting a hidden stream from everything around it.
    pub fn masked() -> Self {
        Self::new(Objective::Masked)
    }

    /// Producing actions from observations and an instruction.
    pub fn action_chunk() -> Self {
        Self::new(Objective::ActionChunk)
    }

    /// The attempts joined by `rel`, as pairs of whole paths.
    pub fn relations(rel: crate::model::Rel) -> Self {
        Self {
            relation: Some(rel),
            ..Self::new(Objective::Relation)
        }
    }

    /// Only attempts whose evidence decides, at this rank or stronger.
    pub fn min_rank(mut self, rank: u8) -> Self {
        self.min_rank = Some(rank);
        self
    }

    /// DPO: pair each passing attempt of a task with each failing one decided
    /// at the same rank, instead of decisions taken at one state.
    pub fn by_task(mut self) -> Self {
        self.by_task = true;
        self
    }

    /// DPO by task: at most `n` pairs from any one task.
    pub fn max_pairs_per_task(mut self, n: usize) -> Self {
        self.max_pairs_per_task = Some(n);
        self
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

impl Recipe {
    /// Multimodal: the observation streams to window, by name.
    pub fn streams(mut self, names: &[&str]) -> Self {
        self.mm.streams = names.iter().map(|n| (*n).to_owned()).collect();
        self
    }

    /// Multimodal: how much of the past is context.
    pub fn past_ns(mut self, ns: i64) -> Self {
        self.mm.past_ns = ns;
        self
    }

    /// Multimodal: how much of the future is the target.
    pub fn future_ns(mut self, ns: i64) -> Self {
        self.mm.future_ns = ns;
        self
    }

    /// Multimodal: the spacing of anchors.
    pub fn stride_ns(mut self, ns: i64) -> Self {
        self.mm.stride_ns = ns;
        self
    }

    /// Multimodal: the width of a window.
    pub fn window_ns(mut self, ns: i64) -> Self {
        self.mm.window_ns = ns;
        self
    }

    /// Contrastive: the two modalities to align, by stream name.
    pub fn pair(mut self, a: &str, b: &str) -> Self {
        self.mm.pair = Some((a.into(), b.into()));
        self
    }

    /// Contrastive: how far in time a hard negative is taken.
    pub fn hard_shift_ns(mut self, ns: i64) -> Self {
        self.mm.hard_shift_ns = ns;
        self
    }

    /// Contrastive: negatives per sample, the hard one included.
    pub fn negatives(mut self, n: usize) -> Self {
        self.mm.negatives = n;
        self
    }

    /// Masked: the stream to hide.
    pub fn target_stream(mut self, name: &str) -> Self {
        self.mm.target_stream = Some(name.into());
        self
    }

    /// Masked: how much of the surroundings to show.
    pub fn context_ns(mut self, ns: i64) -> Self {
        self.mm.context_ns = ns;
        self
    }

    /// Action chunk: how far ahead the actions reach.
    pub fn horizon_ns(mut self, ns: i64) -> Self {
        self.mm.horizon_ns = ns;
        self
    }

    /// Action chunk: the event that carries the instruction.
    pub fn instruction_event(mut self, name: &str) -> Self {
        self.mm.instruction_event = name.into();
        self
    }
}
