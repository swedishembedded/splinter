// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The two granularities of a plan.
//!
//! A student sees both a terse request and a detailed one for the same
//! change. The prompts that ask a model for them live in
//! `harness/replay.yaml`; the plan checker ([`crate::plan::validate`]) holds
//! each style to its length bounds.

use serde::{Deserialize, Serialize};

/// How detailed a plan is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanStyle {
    /// A request in one to three sentences, the way a person types it.
    Terse,
    /// A request with goal, motivation, behaviour, areas and constraints.
    Detailed,
}

impl PlanStyle {
    /// The name used on the command line and in records.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Terse => "terse",
            Self::Detailed => "detailed",
        }
    }

    /// The style named `name`.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "terse" => Some(Self::Terse),
            "detailed" => Some(Self::Detailed),
            _ => None,
        }
    }
}
