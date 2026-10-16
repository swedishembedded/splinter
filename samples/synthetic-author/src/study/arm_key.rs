// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the measurement of fine-tuned language
// models for its clients. If your team needs expertise in sizing and
// analysing a paired evaluation with honest statistics, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The names a study's arms go by: the examination's arms, and the tuned arm
//! once per way its records were made.

use crate::arm::Arm;
use crate::dataset::Briefings;

/// The name of the policy that always takes the option listed first, as a
/// reference in a comparison.
pub use crate::report::FIRST_LISTED;
/// The name of the policy that always takes the option the beliefs determine
/// most often, as a reference in a comparison.
pub use crate::report::MAJORITY;

/// An arm of the study: an arm of the examination, and for the tuned arm which
/// briefings its adapter was trained on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArmKey {
    arm: Arm,
    briefings: Briefings,
}

impl ArmKey {
    /// An arm of the examination (the tuned arm trained on both briefings).
    #[must_use]
    pub fn of(arm: Arm) -> Self {
        Self {
            arm,
            briefings: Briefings::Both,
        }
    }

    /// The tuned arm whose adapter was trained on `briefings`.
    #[must_use]
    pub fn tuned(briefings: Briefings) -> Self {
        Self {
            arm: Arm::Tuned,
            briefings,
        }
    }

    /// The arm called `name`: an arm of the examination, or `tuned` followed
    /// by the suffix of a briefings setting.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        if let Some(arm) = Arm::parse(name) {
            return Some(Self::of(arm));
        }
        [Briefings::ReaderOnly, Briefings::FixedWordingOnly]
            .into_iter()
            .map(Self::tuned)
            .find(|key| key.name() == name)
    }

    /// The name used in the protocol, the manifests and the reports.
    #[must_use]
    pub fn name(self) -> String {
        match self.arm {
            Arm::Tuned => format!("tuned{}", self.briefings.suffix()),
            other => other.name().to_string(),
        }
    }

    /// The examination's arm this asks of a model.
    #[must_use]
    pub fn arm(self) -> Arm {
        self.arm
    }

    /// Which briefings the adapter was trained on; `Both` for every arm that
    /// has no adapter of its own.
    #[must_use]
    pub fn briefings(self) -> Briefings {
        self.briefings
    }

    /// Whether the arm is answered by an adapter, which each training seed
    /// makes anew.
    #[must_use]
    pub fn is_adapter(self) -> bool {
        self.arm == Arm::Tuned
    }

    /// The examination directory slot this arm's outcomes are kept in: one
    /// shared by every arm without an adapter, and one per adapter.
    #[must_use]
    pub fn slot(self, training_seed: u64) -> String {
        if self.is_adapter() {
            format!("{}-s{training_seed}", self.name())
        } else {
            SHARED_SLOT.to_string()
        }
    }
}

/// The slot of the arms that no training seed changes.
pub const SHARED_SLOT: &str = "shared";
