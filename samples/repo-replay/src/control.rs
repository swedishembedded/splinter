// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! What the replaying agent is given besides the plan and the repository.

use serde::{Deserialize, Serialize};

/// The condition of a run. A state directory holds attempts of one condition.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Control {
    /// The agent is shown the masked change (`harness/replay.yaml`).
    #[default]
    Grounded,
    /// The agent gets the plan and the repository only (`--control plan-only`, the `control`
    /// input of `harness/replay.yaml`).
    PlanOnly,
}

impl Control {
    /// The name used on the command line and in reports.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Grounded => "grounded",
            Self::PlanOnly => "plan-only",
        }
    }

    /// The control named `name`.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "grounded" => Some(Self::Grounded),
            "plan-only" => Some(Self::PlanOnly),
            _ => None,
        }
    }
}
