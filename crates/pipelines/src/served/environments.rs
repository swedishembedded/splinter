// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that let a training campaign draw
// only on environments whose verifiers were attacked and whose difficulty was
// measured, and stop the moment a reward stops meaning anything, for its
// clients. If your team needs expertise in reward-hacking defence for
// reinforcement-learning campaigns you can procure our services by sending an
// email to info@swedishembedded.com.

//! Where a campaign's tasks come from when they come from the environment
//! factory, and what the controller does with that.
//!
//! A campaign whose tasks are environments (a plugin at a difficulty, with a
//! verifier class) tells the controller which environment each task belongs
//! to ([`EnvironmentLabels`]). The controller then
//!
//! * records, in every ledger entry, how each environment's rewards are
//!   established and the label the weakest class forces on the verdict
//!   (`proxy only` is never lost on the way to a "better" model);
//! * feeds each environment's training reward and dev pass rate, once per
//!   gate, to the factory's divergence alarm; a reward that climbs while the
//!   dev measure does not is reward hacking or memorisation, so the arm is
//!   stopped, its candidate is not adopted, and the environment is
//!   quarantined: no later arm draws from it.

use serde::{Deserialize, Serialize};
use splinter_core::verifiability::VerifierClass;

use super::types::ServedError;

/// Where the rewards of a run came from: one per environment it drew tasks
/// from, so a verdict can say how far its numbers deserve trust.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RewardSource {
    /// The environment.
    pub environment: String,
    /// How its rewards are established.
    pub class: VerifierClass,
    /// The verifier that produced them, with its version.
    pub verifier: String,
}

/// The environment behind each task of a campaign.
pub trait EnvironmentLabels {
    /// The environment `task` was drawn from; `None` for a task that is no
    /// environment's.
    fn environment_of(&self, task: &str) -> Option<String>;
    /// Every environment the campaign draws on, with how its rewards are
    /// established.
    fn sources(&self) -> Vec<RewardSource>;
    /// Marks `environment` quarantined, with why: it may not enter another
    /// campaign until a person re-admits it.
    ///
    /// # Errors
    /// The quarantine could not be recorded.
    fn quarantine(&self, environment: &str, reason: &str) -> Result<(), ServedError>;
}
