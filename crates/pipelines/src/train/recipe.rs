// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reinforcement learning from the verified
// rewards of agent rollouts for its clients. If your team needs expertise in
// policy-gradient recipes that move a policy measurably and safely, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Named training recipes: the defaults a kind of run is known to need, laid
//! under whatever the command names itself.
//!
//! A recipe sets only what was left unset, records itself with the run, and
//! states the targets its run should reach, so the candidate says whether it
//! reached them ([`RecipeReading`]). Brain's own algorithm presets
//! (`--algorithm grpo|dr-grpo|dapo|rloo`) shape the objective; a recipe is
//! the run around it: the learning rate and how many passes over one batch of
//! rollouts it takes.

use serde::{Deserialize, Serialize};
use splinter_core::training::PolicyGradientSummary;
use splinter_model::policy_gradient::{Behaviour, LossAggregation};

use super::Tuning;

/// A named recipe.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Recipe {
    /// On-policy reinforcement learning on one batch of rollouts, sampled by
    /// the policy the run starts from: the exact recorded behaviour
    /// log-probabilities, every token of a step weighed alike (token sum, as
    /// Dr. GRPO argues against a per-sequence mean), a learning rate of
    /// [`ON_POLICY_RL_LEARNING_RATE`] and [`ON_POLICY_RL_PASSES`] passes over
    /// the batch. A run at a small learning rate over its rollouts once barely
    /// moved the policy and clipped almost none of its tokens; the targets
    /// ([`ON_POLICY_RL_KL`], [`ON_POLICY_RL_CLIP`]) say what moving it looks
    /// like.
    OnPolicyRl,
}

/// The learning rate of [`Recipe::OnPolicyRl`].
pub const ON_POLICY_RL_LEARNING_RATE: f32 = 1e-4;

/// The passes over a batch of rollouts of [`Recipe::OnPolicyRl`]: brain draws
/// every record once per pass, reshuffled; the recorded behaviour
/// log-probabilities keep the importance ratio exact in every pass.
pub const ON_POLICY_RL_PASSES: u32 = 4;

/// The mean KL to the reference [`Recipe::OnPolicyRl`] aims for: below it the
/// policy did not move, above it it moved further than one batch supports.
pub const ON_POLICY_RL_KL: (f32, f32) = (0.02, 0.05);

/// The mean clip fraction [`Recipe::OnPolicyRl`] aims for.
pub const ON_POLICY_RL_CLIP: (f32, f32) = (0.10, 0.20);

impl Recipe {
    /// The recipe's name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::OnPolicyRl => "on-policy-rl",
        }
    }

    /// `tuning` with the recipe's defaults wherever it named nothing, and the
    /// recipe recorded.
    #[must_use]
    pub fn apply(self, tuning: Tuning) -> Tuning {
        match self {
            Self::OnPolicyRl => {
                let mut options = tuning.policy_gradient;
                options.behaviour = options.behaviour.or(Some(Behaviour::Recorded));
                options.loss_agg = options.loss_agg.or(Some(LossAggregation::TokenSum));
                Tuning {
                    recipe: Some(self),
                    learning_rate: tuning.learning_rate.or(Some(ON_POLICY_RL_LEARNING_RATE)),
                    passes: tuning.passes.or(Some(ON_POLICY_RL_PASSES)),
                    policy_gradient: options,
                    ..tuning
                }
            }
        }
    }

    /// What the run `measured` came to against the recipe's targets.
    #[must_use]
    pub fn read(self, measured: Option<&PolicyGradientSummary>) -> RecipeReading {
        let (kl, clip) = match self {
            Self::OnPolicyRl => (ON_POLICY_RL_KL, ON_POLICY_RL_CLIP),
        };
        let kl_measured = measured.and_then(|m| m.mean_kl);
        let clip_measured = measured.and_then(|m| m.mean_clip_fraction);
        RecipeReading {
            recipe: self,
            kl_target: kl,
            kl: kl_measured,
            kl_reached: kl_measured.map(|v| Reached::of(v, kl)),
            clip_target: clip,
            clip_fraction: clip_measured,
            clip_reached: clip_measured.map(|v| Reached::of(v, clip)),
        }
    }
}

impl std::str::FromStr for Recipe {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        match text {
            "on-policy-rl" => Ok(Self::OnPolicyRl),
            other => Err(format!("{other:?} is not a recipe: on-policy-rl")),
        }
    }
}

/// Where a measured figure is against a target band.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reached {
    /// Under the band.
    Below,
    /// In it.
    Within,
    /// Over it.
    Above,
}

impl Reached {
    fn of(value: f32, (low, high): (f32, f32)) -> Self {
        if value < low {
            Self::Below
        } else if value > high {
            Self::Above
        } else {
            Self::Within
        }
    }
}

/// A run trained by a recipe against the recipe's targets; an unmeasured
/// figure is absent and reaches nothing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecipeReading {
    /// The recipe.
    pub recipe: Recipe,
    /// The mean KL to the reference it aims for, low and high.
    pub kl_target: (f32, f32),
    /// The mean KL measured.
    pub kl: Option<f32>,
    /// Where it is against the target.
    pub kl_reached: Option<Reached>,
    /// The mean clip fraction it aims for, low and high.
    pub clip_target: (f32, f32),
    /// The mean clip fraction measured.
    pub clip_fraction: Option<f32>,
    /// Where it is against the target.
    pub clip_reached: Option<Reached>,
}

impl RecipeReading {
    /// The reading as one line.
    #[must_use]
    pub fn line(&self) -> String {
        let figure = |v: Option<f32>, reached: Option<Reached>| match (v, reached) {
            (Some(v), Some(r)) => format!("{v:.4} ({r:?})").to_lowercase(),
            _ => "not measured".to_string(),
        };
        format!(
            "recipe {}: mean KL {} against [{}, {}], mean clip fraction {} against [{}, {}]",
            self.recipe.name(),
            figure(self.kl, self.kl_reached),
            self.kl_target.0,
            self.kl_target.1,
            figure(self.clip_fraction, self.clip_reached),
            self.clip_target.0,
            self.clip_target.1,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_recipe_fills_only_what_was_left_unset() {
        let tuned = Recipe::OnPolicyRl.apply(Tuning::default());
        assert_eq!(tuned.recipe, Some(Recipe::OnPolicyRl));
        assert_eq!(tuned.learning_rate, Some(ON_POLICY_RL_LEARNING_RATE));
        assert_eq!(tuned.passes, Some(ON_POLICY_RL_PASSES));
        assert_eq!(tuned.policy_gradient.behaviour, Some(Behaviour::Recorded));
        assert_eq!(
            tuned.policy_gradient.loss_agg,
            Some(LossAggregation::TokenSum)
        );
        let mut named = Tuning {
            learning_rate: Some(3e-5),
            passes: Some(2),
            ..Tuning::default()
        };
        named.policy_gradient.loss_agg = Some(LossAggregation::SeqMean);
        let kept = Recipe::OnPolicyRl.apply(named);
        assert_eq!(kept.learning_rate, Some(3e-5));
        assert_eq!(kept.passes, Some(2));
        assert_eq!(
            kept.policy_gradient.loss_agg,
            Some(LossAggregation::SeqMean)
        );
    }

    #[test]
    fn the_reading_places_the_measured_kl_and_clip_fraction_against_the_targets() {
        let measured = PolicyGradientSummary {
            mean_kl: Some(0.002),
            mean_clip_fraction: Some(0.15),
            ..PolicyGradientSummary::default()
        };
        let reading = Recipe::OnPolicyRl.read(Some(&measured));
        assert_eq!(reading.kl_reached, Some(Reached::Below));
        assert_eq!(reading.clip_reached, Some(Reached::Within));
        let unmeasured = Recipe::OnPolicyRl.read(None);
        assert_eq!((unmeasured.kl, unmeasured.kl_reached), (None, None));
    }
}
