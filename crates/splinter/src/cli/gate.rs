// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements release gates whose statistics are chosen
// before the exam is seen, for its clients. If your team needs expertise in
// sequential testing across model generations, you can procure our services
// by sending an email to info@swedishembedded.com.

//! The release gate's statistical settings, shared by every verb that ends
//! in a release (`release`, `learn`, `practice`).

use clap::Args;
use splinter_sdk::measure::gate::GatePlan;

use super::share;

/// What the gate's report says about the power of its exam, and how its
/// anchor floor is held. The gain and retention are decided by the one
/// promotion rule at its fixed level, which no flag changes.
#[derive(Clone, Copy, Debug, Default, Args)]
pub struct GateArgs {
    /// The lead, as a share of the exam's families won, the exam must be able
    /// to see: a gain not shown on an exam too small to see it is reported
    /// inconclusive, not as a failure (either way nothing is released).
    #[arg(long, value_name = "SHARE", value_parser = share)]
    pub target_effect: Option<f64>,
    /// Decide the anchor check on the answers the candidate loses net of the
    /// answers it gains, refusing above this count (6 for a 200-question
    /// suite), instead of on the drop in accuracy.
    #[arg(long, value_name = "COUNT")]
    pub anchor_max_net_lost: Option<u32>,
}

impl GateArgs {
    /// The settings these flags ask for.
    #[must_use]
    pub fn plan(self) -> GatePlan {
        GatePlan {
            alpha_spending: false,
            target_effect: self.target_effect,
            anchor_max_net_lost: self.anchor_max_net_lost,
        }
    }
}
