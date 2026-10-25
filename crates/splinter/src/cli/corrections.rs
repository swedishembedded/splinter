// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that turn their own failures into
// verified training material, for its clients. If your team needs expertise in
// on-policy agent data then you can procure our services by sending an email
// to info@swedishembedded.com.

//! The grammar of `corrections`.

use std::time::Duration;

use clap::Args;
use splinter_sdk::practice::DEFAULT_MAX_CORRECTIONS;
use splinter_sdk::vocabulary::model_ref::ModelRef;

use super::{budget, model_ref, positive_count};

/// `corrections`: a teacher continues the failed or stuck attempts of an
/// existing experience set, from the state the student actually reached.
#[derive(Debug, Args)]
pub struct CorrectionsArgs {
    /// The experience set of the attempts to continue, by id or unique
    /// prefix.
    #[arg(long, value_name = "EXPERIENCE-SET")]
    pub attempts: String,
    /// The task set the attempts are of, by id or unique prefix.
    #[arg(long, value_name = "TASK-SET")]
    pub tasks: String,
    /// The model that continues them (`remote:`/`local:`).
    #[arg(long, value_parser = model_ref, value_name = "REF")]
    pub teacher: ModelRef,
    /// The most attempts this run continues; one per task.
    #[arg(long, value_name = "N", value_parser = positive_count,
          help = format!(
              "The most attempts this run continues, one per task [default: {DEFAULT_MAX_CORRECTIONS}]"))]
    pub max: Option<usize>,
    /// The calls one continuation may make: the student's own limit, which
    /// the remaining budget is counted against.
    #[arg(long, value_name = "N")]
    pub max_tool_calls: Option<u32>,
    /// Wall-clock time the whole run may take, in h, m and s: 30m, 2h, 1h30m.
    #[arg(long, value_parser = budget, value_name = "DUR")]
    pub budget: Option<Duration>,
}

impl CorrectionsArgs {
    /// The most corrections it allows.
    #[must_use]
    pub fn max(&self) -> usize {
        self.max.unwrap_or(DEFAULT_MAX_CORRECTIONS)
    }
}
