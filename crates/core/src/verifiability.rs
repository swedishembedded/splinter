// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems whose reward can be
// trusted, for its clients. If your team needs expertise in verifier design
// or in keeping reward hacking out of reinforcement learning, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Where a reward comes from, and what a verdict built on it may claim.
//!
//! The source of a reward decides how much a high score proves. A program
//! run by an interpreter cannot be argued with; a rubric applied by a model
//! can; a simulated user stands in for a real one and can be wrong about
//! what a real one does. So every reward carries the [`VerifierClass`] it
//! came from, with the environment, split and seed that produced it
//! ([`RewardProvenance`]), and every verdict built on rewards carries the
//! label the weakest class among them forces ([`verdict_label`]): a verdict
//! resting on a proxy says "proxy only", never a bare "improved".
//!
//! [`Split`] names the two parts of a generator Splinter produces. A sealed
//! exam is held by the measurement authority and is never produced here, so
//! the type has no variant for it and parsing one is refused by name.

use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// How a reward is established, from the most to the least reliable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum VerifierClass {
    /// A program decides: an interpreter, a test suite, a state diff.
    Objective,
    /// A person's own evidence decides: a recorded outcome, a rating.
    HumanEvidence,
    /// A judge model decides against a rubric, and its agreement with
    /// humans on a calibration set is measured.
    CalibratedJudge,
    /// A stand-in decides: a keyword rule, a simulated user, a heuristic.
    /// What it rewards may differ from what it stands for.
    Proxy,
}

impl VerifierClass {
    /// Every class, most reliable first.
    pub const ALL: [Self; 4] = [
        Self::Objective,
        Self::HumanEvidence,
        Self::CalibratedJudge,
        Self::Proxy,
    ];

    /// The class's name as it is written in files and labels.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Objective => "objective",
            Self::HumanEvidence => "human-evidence",
            Self::CalibratedJudge => "calibrated-judge",
            Self::Proxy => "proxy",
        }
    }

    /// What a verdict resting on this class must say beside its number;
    /// `None` for an objective check, which needs no qualification.
    #[must_use]
    pub const fn label(self) -> Option<&'static str> {
        match self {
            Self::Objective => None,
            Self::HumanEvidence => Some("human evidence"),
            Self::CalibratedJudge => Some("judge-graded"),
            Self::Proxy => Some("proxy only"),
        }
    }

    /// Whether a model that learned to please this verifier has shown the
    /// skill it was meant to teach. Only an objective check or a person's
    /// evidence says so; a judge or a proxy shows agreement with itself.
    #[must_use]
    pub const fn supports_efficacy_claim(self) -> bool {
        matches!(self, Self::Objective | Self::HumanEvidence)
    }
}

impl fmt::Display for VerifierClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Text that names no verifier class.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
    "{text:?} is not a verifier class (one of objective, human-evidence, calibrated-judge, proxy)"
)]
pub struct ParseClassError {
    text: String,
}

impl FromStr for VerifierClass {
    type Err = ParseClassError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|class| class.as_str() == text.trim())
            .ok_or_else(|| ParseClassError {
                text: text.to_string(),
            })
    }
}

/// The class a verdict over rewards of `classes` is only as good as: the
/// least reliable one present. `None` for no rewards.
#[must_use]
pub fn weakest_class(classes: impl IntoIterator<Item = VerifierClass>) -> Option<VerifierClass> {
    classes.into_iter().max()
}

/// The label a verdict over rewards of `classes` must carry: that of the
/// weakest class present, `None` when every reward is an objective check (or
/// there are none).
#[must_use]
pub fn verdict_label(classes: impl IntoIterator<Item = VerifierClass>) -> Option<&'static str> {
    weakest_class(classes).and_then(VerifierClass::label)
}

/// Which part of an environment's generator an instance is drawn from.
///
/// There is no sealed-exam variant: the measurement authority seals and
/// holds exams, and nothing here may produce one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Split {
    /// Instances a policy trains on.
    Train,
    /// Instances that decide whether training continues; never trained on.
    Dev,
}

impl Split {
    /// The split's name as it is written in files.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Dev => "dev",
        }
    }
}

impl fmt::Display for Split {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Text that names no split Splinter produces.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ParseSplitError {
    /// A sealed or held-out exam split: held by the measurement authority.
    #[error("{0:?} is a sealed split: the measurement authority holds it and Splinter never produces it")]
    Sealed(String),
    /// Not a split at all.
    #[error("{0:?} is not a split (train or dev)")]
    Unknown(String),
}

impl FromStr for Split {
    type Err = ParseSplitError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text.trim() {
            "train" => Ok(Self::Train),
            "dev" => Ok(Self::Dev),
            sealed @ ("exam" | "test" | "sealed" | "heldout" | "held-out" | "holdout") => {
                Err(ParseSplitError::Sealed(sealed.to_string()))
            }
            other => Err(ParseSplitError::Unknown(other.to_string())),
        }
    }
}

/// Where one reward came from: enough to tell, from the reward alone, how
/// far to trust it and how to draw the same instance again.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RewardProvenance {
    /// How the reward was established.
    pub class: VerifierClass,
    /// The verifier that produced it, with its version.
    pub verifier: String,
    /// The environment, by its content version.
    pub environment: String,
    /// The part of the generator the instance was drawn from.
    pub split: Split,
    /// The seed the instance was drawn with.
    pub seed: u64,
}

/// A reward and where it came from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Reward {
    /// The scalar a trainer reads, in `0..=1`.
    pub value: f64,
    /// The parts it is made of, each in `0..=1`; empty for a scalar-only
    /// verifier. A trainer may weight them; the scalar is the default.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub components: BTreeMap<String, f64>,
    /// Where it came from.
    pub provenance: RewardProvenance,
}

/// A reward that is not a number in `0..=1`.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[error("reward {name} is {value}: a reward is a finite number in 0..=1")]
pub struct RewardRangeError {
    name: String,
    value: f64,
}

impl Reward {
    /// A reward of `value` made of `components`, from `provenance`.
    ///
    /// # Errors
    ///
    /// When the value or a component is not finite and within `0..=1`.
    pub fn new(
        value: f64,
        components: BTreeMap<String, f64>,
        provenance: RewardProvenance,
    ) -> Result<Self, RewardRangeError> {
        let in_range = |v: f64| v.is_finite() && (0.0..=1.0).contains(&v);
        if !in_range(value) {
            return Err(RewardRangeError {
                name: "value".into(),
                value,
            });
        }
        if let Some((name, &bad)) = components.iter().find(|(_, v)| !in_range(**v)) {
            return Err(RewardRangeError {
                name: name.clone(),
                value: bad,
            });
        }
        Ok(Self {
            value,
            components,
            provenance,
        })
    }

    /// The label a verdict resting on this reward alone must carry.
    #[must_use]
    pub fn label(&self) -> Option<&'static str> {
        self.provenance.class.label()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // a panic is the failure report
mod tests {
    use super::*;

    fn provenance(class: VerifierClass) -> RewardProvenance {
        RewardProvenance {
            class,
            verifier: "interpreter-output@1".into(),
            environment: "env-0".into(),
            split: Split::Train,
            seed: 7, // seed: the value is only recorded, never drawn from
        }
    }

    #[test]
    fn a_verdict_is_labelled_by_the_weakest_reward_it_rests_on() {
        use VerifierClass::*;
        assert_eq!(verdict_label([Objective, Objective]), None);
        assert_eq!(verdict_label([]), None);
        assert_eq!(verdict_label([Objective, Proxy]), Some("proxy only"));
        assert_eq!(verdict_label([Proxy, CalibratedJudge]), Some("proxy only"));
        assert_eq!(
            verdict_label([Objective, CalibratedJudge]),
            Some("judge-graded")
        );
        assert_eq!(verdict_label([HumanEvidence]), Some("human evidence"));
    }

    #[test]
    fn only_an_objective_check_or_a_persons_evidence_supports_an_efficacy_claim() {
        let supporting: Vec<_> = VerifierClass::ALL
            .into_iter()
            .filter(|c| c.supports_efficacy_claim())
            .collect();
        assert_eq!(
            supporting,
            [VerifierClass::Objective, VerifierClass::HumanEvidence]
        );
    }

    #[test]
    fn classes_round_trip_through_their_names() {
        for class in VerifierClass::ALL {
            assert_eq!(class.as_str().parse::<VerifierClass>().unwrap(), class);
            let json = serde_json::to_string(&class).unwrap();
            assert_eq!(json, format!("\"{}\"", class.as_str()));
        }
        assert!("judge".parse::<VerifierClass>().is_err());
    }

    #[test]
    fn a_sealed_split_is_refused_by_name_and_never_produced() {
        assert_eq!("train".parse::<Split>().unwrap(), Split::Train);
        assert_eq!("dev".parse::<Split>().unwrap(), Split::Dev);
        for sealed in ["exam", "test", "sealed", "heldout", "held-out", "holdout"] {
            assert_eq!(
                sealed.parse::<Split>().unwrap_err(),
                ParseSplitError::Sealed(sealed.into())
            );
        }
        assert!(matches!(
            "validation".parse::<Split>(),
            Err(ParseSplitError::Unknown(_))
        ));
    }

    #[test]
    fn a_reward_is_a_finite_number_in_the_unit_interval() {
        let ok = Reward::new(1.0, BTreeMap::new(), provenance(VerifierClass::Objective)).unwrap();
        assert_eq!(ok.label(), None);
        for bad in [-0.1, 1.5, f64::NAN, f64::INFINITY] {
            assert!(Reward::new(bad, BTreeMap::new(), provenance(VerifierClass::Proxy)).is_err());
        }
        let components = BTreeMap::from([("state".to_string(), 2.0)]);
        let err = Reward::new(0.5, components, provenance(VerifierClass::Proxy)).unwrap_err();
        assert!(err.to_string().contains("state"), "{err}");
    }

    #[test]
    fn a_proxy_reward_carries_its_label_and_provenance_through_json() {
        let reward = Reward::new(
            0.5,
            BTreeMap::from([("rubric".to_string(), 0.5)]),
            provenance(VerifierClass::Proxy),
        )
        .unwrap();
        assert_eq!(reward.label(), Some("proxy only"));
        let back: Reward = serde_json::from_str(&serde_json::to_string(&reward).unwrap()).unwrap();
        assert_eq!(back, reward);
    }
}
