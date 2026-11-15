// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that run model learning as one
// continuous, resumable campaign, for its clients. If your team needs
// expertise in the configuration of policy-gradient campaigns you can
// procure our services by sending an email to info@swedishembedded.com.

//! A campaign's configuration: one value, digested into every ledger entry.

use serde::{Deserialize, Serialize};
use splinter_core::digest::{digest_of, Digest};
use splinter_eval::campaign::evidence::EvidenceGate;
use splinter_eval::campaign::gate::DevGate;
use splinter_eval::campaign::pass_rate::{CALIBRATION_K, RETIRE_AT};
use splinter_eval::campaign::plateau::PlateauRule;
use splinter_eval::campaign::sampling::{AllFailStrategy, BatchPolicy};
use splinter_eval::campaign::signals::SignalBounds;
use splinter_knowledge::factory::alarm::DivergencePolicy;

use super::CampaignError;
use crate::preflight::Requirement;

/// How a group's attempts earn advantages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdvantageMode {
    /// The standardised advantage: better attempts are pushed up and worse
    /// ones down. Wrong attempts share most of their tokens with right ones,
    /// so on reused prompts the negatives memorise; on fresh prompts it is the
    /// published method.
    TwoSided,
    /// Only attempts that beat their group are reinforced and nothing is
    /// pushed down: the rejection-sampling limit.
    PositiveOnly,
}

impl AdvantageMode {
    /// The advantage rule this mode names.
    #[must_use]
    pub fn kind(self) -> splinter_data::AdvantageKind {
        match self {
            Self::TwoSided => splinter_data::AdvantageKind::Standardised,
            Self::PositiveOnly => splinter_data::AdvantageKind::Positive,
        }
    }
}

/// What an arm does with its batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LearningStrategy {
    /// Group-relative policy gradient on the informative groups, two-sided
    /// advantages.
    PolicyGradient,
    /// The same with positive-only advantages.
    PositiveOnly,
    /// Supervised repair: a wrong attempt, the verifier's feedback, and the
    /// fix a later attempt found, trained as imitation of the fix.
    Repair,
    /// Policy gradient plus per-token distillation of the groups that all
    /// failed: the policy's own tokens scored by the frozen model shown what
    /// the verifier knows.
    Distill,
}

impl LearningStrategy {
    /// The name the ledger records.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::PolicyGradient => "policy-gradient",
            Self::PositiveOnly => "positive-only",
            Self::Repair => "sft-repair",
            Self::Distill => "distill",
        }
    }
}

/// How an arm's candidate is accepted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum Acceptance {
    /// The dev gate: a net gain over every dev task within the retention
    /// margin; significance is reported, not required.
    #[default]
    Greedy,
    /// The evidence gate: each arm draws `draw` dev tasks (fewer than the dev
    /// set holds) and the candidate advances only when a noninferiority
    /// e-process over them crosses the arm's level, the false evolution rate
    /// held to `gate.q` over `horizon` arms.
    Evidence {
        /// The test and its budget.
        gate: EvidenceGate,
        /// Dev tasks an arm draws.
        draw: usize,
        /// Arms the budget is planned over; later arms test at level zero.
        horizon: u32,
    },
}

impl Acceptance {
    /// Whether this is the greedy gate, which a configuration does not
    /// write, so the digest of every configuration before the choice existed
    /// is unchanged.
    #[must_use]
    pub fn is_greedy(&self) -> bool {
        *self == Self::Greedy
    }
}

/// The bounded replay of earlier arms' verified data.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReplayConfig {
    /// Most records the buffer holds; the oldest arm's go first.
    pub capacity: usize,
    /// Replay records per batch record, drawn from earlier arms only.
    pub per_batch_record: f64,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            capacity: 256,
            per_batch_record: 0.25,
        }
    }
}

/// Everything that shapes a campaign.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CampaignConfig {
    /// The campaign's name; its state and ledger are kept under it.
    pub name: String,
    /// What the campaign is for, in a sentence, written before it runs; every
    /// arm's ledger entry repeats it with the strategy.
    pub hypothesis: String,
    /// Attempts per task of the offline calibration, `8..=16`.
    pub calibration_k: usize,
    /// Tasks per calibration request.
    pub calibration_batch: usize,
    /// Attempts per task in a training group.
    pub attempts_per_group: usize,
    /// How a batch is built.
    pub batch: BatchPolicy,
    /// The pass rate at which a task is dropped for good.
    pub retire_at: f64,
    /// What to do with a task whose group all failed.
    pub all_fail: AllFailStrategy,
    /// The strategies the campaign runs, in order; a plateau moves to the
    /// next, and the last repeats.
    pub ladder: Vec<LearningStrategy>,
    /// Bounds the signals of every step are held to.
    pub bounds: SignalBounds,
    /// The replay of earlier arms.
    pub replay: ReplayConfig,
    /// The dev gate: its retention rule applies under either acceptance.
    pub gate: DevGate,
    /// How a candidate is accepted.
    #[serde(default, skip_serializing_if = "Acceptance::is_greedy")]
    pub acceptance: Acceptance,
    /// The lead, in the share of dev units won, an arm is meant to produce.
    pub target_effect: f64,
    /// The share of dev units on which a candidate and its champion are
    /// expected to differ; replaced by the last gate's measurement once there
    /// is one.
    pub expected_discordance: f64,
    /// Discoveries per GPU-hour below which the campaign has plateaued.
    pub plateau: PlateauRule,
    /// Cards the campaign holds, for the GPU-hours it records.
    pub gpus: u32,
    /// Repairs (wrong attempt, fix) built per group under [`LearningStrategy::Repair`].
    pub repairs_per_group: usize,
    /// Attempts of an all-failed group distilled under the cliff strategy.
    pub distilled_per_group: usize,
    /// Disks the campaign writes to and what each must have free before an arm.
    pub preflight: Vec<Requirement>,
    /// The share of the retention set a candidate may lose that the probe
    /// must be able to detect; the retention set is refused when too small for
    /// it. Zero skips the check.
    pub retention_loss_to_detect: f64,
    /// Checkpoints kept beyond the champion and a pending candidate.
    pub keep_recent_checkpoints: usize,
    /// When an environment's training reward climbing while its dev measure
    /// does not is called a divergence (only for tasks that come from an
    /// environment factory).
    pub divergence: DivergencePolicy,
    /// Arms in a row that may fail to advance the champion once the strategy
    /// ladder is spent; the campaign then stops instead of burning GPU-hours.
    pub max_idle_arms: u32,
}

impl CampaignConfig {
    /// A configuration with the defaults a first campaign wants: calibration
    /// at k = 8, groups of 8, batches of 16 informative groups, dynamic
    /// sampling over-provisioned by half, two-sided advantages.
    #[must_use]
    pub fn new(name: &str, hypothesis: &str) -> Self {
        Self {
            name: name.to_string(),
            hypothesis: hypothesis.to_string(),
            calibration_k: CALIBRATION_K,
            calibration_batch: 32,
            attempts_per_group: 8,
            batch: BatchPolicy {
                target_groups: 16,
                overprovision: 1.5,
                max_rounds: 4,
                max_request: 128,
            },
            retire_at: RETIRE_AT,
            all_fail: AllFailStrategy::Skip,
            ladder: vec![LearningStrategy::PolicyGradient],
            bounds: SignalBounds::default(),
            replay: ReplayConfig::default(),
            gate: DevGate::default(),
            acceptance: Acceptance::Greedy,
            target_effect: 0.15,
            expected_discordance: 0.3,
            plateau: PlateauRule::default(),
            gpus: 1,
            repairs_per_group: 2,
            distilled_per_group: 2,
            preflight: Vec::new(),
            retention_loss_to_detect: 0.0,
            keep_recent_checkpoints: 1,
            divergence: DivergencePolicy::default(),
            max_idle_arms: 4,
        }
    }

    /// The digest of the configuration, which a ledger entry carries.
    #[must_use]
    pub fn digest(&self) -> Digest {
        digest_of(self)
    }

    /// Refuses a configuration that cannot run a campaign.
    ///
    /// # Errors
    /// The first thing wrong with it.
    pub fn validate(&self) -> Result<(), CampaignError> {
        let refuse = |why: String| Err(CampaignError::Config(why));
        if !(8..=16).contains(&self.calibration_k) {
            return refuse(format!(
                "calibration_k is {}; the offline calibration needs 8 to 16 attempts per task",
                self.calibration_k
            ));
        }
        if self.calibration_batch == 0 {
            return refuse("calibration_batch is 0".into());
        }
        if self.attempts_per_group < 2 {
            return refuse("a group needs at least 2 attempts to show a spread".into());
        }
        self.batch
            .validate()
            .map_err(|e| CampaignError::Config(e.to_string()))?;
        if self.ladder.is_empty() {
            return refuse("the strategy ladder is empty".into());
        }
        if self.gpus == 0 {
            return refuse("a campaign holds at least one GPU".into());
        }
        if !(self.replay.per_batch_record >= 0.0 && self.replay.per_batch_record.is_finite()) {
            return refuse("replay.per_batch_record is not a finite number of at least 0".into());
        }
        if let Acceptance::Evidence {
            gate,
            draw,
            horizon,
        } = self.acceptance
        {
            gate.validate()
                .map_err(|e| CampaignError::Config(e.to_string()))?;
            if draw == 0 || horizon == 0 {
                return refuse(
                    "the evidence gate draws at least one dev task over at least one arm".into(),
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_configuration_is_valid_and_its_digest_follows_every_setting() {
        let config = CampaignConfig::new("c", "dynamic sampling finds signal");
        config.validate().unwrap();
        let mut changed = config.clone();
        changed.attempts_per_group = 16;
        assert_ne!(config.digest(), changed.digest());
        assert_eq!(config.digest(), config.clone().digest());
    }

    #[test]
    fn a_configuration_that_cannot_run_is_refused_with_the_reason() {
        let ok = CampaignConfig::new("c", "h");
        for (mutate, mention) in [
            (
                (|c: &mut CampaignConfig| c.calibration_k = 4) as fn(&mut CampaignConfig),
                "8 to 16",
            ),
            (|c| c.attempts_per_group = 1, "2 attempts"),
            (|c| c.ladder.clear(), "ladder"),
            (|c| c.gpus = 0, "GPU"),
            (|c| c.batch.target_groups = 0, "at least one group"),
        ] {
            let mut bad = ok.clone();
            mutate(&mut bad);
            let refused = bad.validate().unwrap_err().to_string();
            assert!(refused.contains(mention), "{refused}");
        }
    }

    #[test]
    fn the_greedy_gate_is_not_written_so_earlier_digests_hold_and_an_evidence_gate_is() {
        let greedy = CampaignConfig::new("c", "h");
        let json = serde_json::to_value(&greedy).unwrap();
        assert!(json.get("acceptance").is_none(), "{json}");
        let mut evidence = greedy.clone();
        evidence.acceptance = Acceptance::Evidence {
            gate: EvidenceGate::new(0.02, 0.3, 0.01).unwrap(),
            draw: 60,
            horizon: 8,
        };
        evidence.validate().unwrap();
        assert_ne!(greedy.digest(), evidence.digest());
        let read: CampaignConfig =
            serde_json::from_value(serde_json::to_value(&evidence).unwrap()).unwrap();
        assert_eq!(read, evidence);
        evidence.acceptance = Acceptance::Evidence {
            gate: EvidenceGate::new(0.02, 0.3, 0.01).unwrap(),
            draw: 0,
            horizon: 8,
        };
        assert!(evidence.validate().is_err());
    }

    #[test]
    fn the_advantage_modes_name_the_views_rules() {
        assert_eq!(
            AdvantageMode::PositiveOnly.kind(),
            splinter_data::AdvantageKind::Positive
        );
        assert_eq!(
            AdvantageMode::TwoSided.kind(),
            splinter_data::AdvantageKind::Standardised
        );
    }
}
