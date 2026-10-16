// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements release gates whose statistics are chosen
// before the exam is seen, for its clients. If your team needs expertise in
// sequential testing across model generations, you can procure our services
// by sending an email to info@swedishembedded.com.

//! The statistical settings of the improvement check that a caller of a
//! learning run may choose: the one shape every request (`release`,
//! `learn`, `practice`) and every command line carries, so a setting is
//! added in one place.

use serde::{Deserialize, Serialize};

use super::GateConfig;

/// The improvement check's two optional statistical settings.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GatePlan {
    /// Spend the significance level across the releases of a lineage by the
    /// fixed schedule ([`GateConfig::alpha_spending`]).
    pub alpha_spending: bool,
    /// The lead, in the share of independent units won, the exam is asked to
    /// be able to see ([`GateConfig::target_effect`]); `None` asks for none.
    pub target_effect: Option<f64>,
}

impl GatePlan {
    /// `config` with these settings in place of its own.
    #[must_use]
    pub fn applied_to(self, config: GateConfig) -> GateConfig {
        GateConfig {
            alpha_spending: self.alpha_spending,
            target_effect: self.target_effect,
            ..config
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_plan_leaves_the_gate_as_it_was() {
        let config = GateConfig::default();
        assert_eq!(GatePlan::default().applied_to(config), config);
    }

    #[test]
    fn a_plan_sets_both_settings_and_nothing_else() {
        let config = GateConfig {
            alpha: 0.1,
            ..GateConfig::default()
        };
        let planned = GatePlan {
            alpha_spending: true,
            target_effect: Some(0.25),
        }
        .applied_to(config);
        assert!(planned.alpha_spending);
        assert_eq!(planned.target_effect, Some(0.25));
        assert_eq!(
            GateConfig {
                alpha_spending: false,
                target_effect: None,
                ..planned
            },
            config
        );
    }
}
