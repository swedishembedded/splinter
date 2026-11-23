// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verification and critique of agent
// experience before it becomes training data, for its clients. If your team
// needs expertise in grading agent trajectories you can procure our services
// by sending an email to info@swedishembedded.com.

//! The grammar of `verify` and `critique`.

use clap::Args;
use splinter_sdk::critique::DEFAULT_RETRIES;
use splinter_sdk::vocabulary::model_ref::ModelRef;

use super::model_ref;

/// `verify`.
#[derive(Debug, Args)]
pub struct VerifyArgs {
    /// A factory run directory, or an experience set by id or unique
    /// prefix.
    #[arg(value_name = "RUN|EXPERIENCE-SET")]
    pub experience_set: String,
    /// A calibrated judge for the kinds a judge grades.
    #[arg(long, value_parser = model_ref, value_name = "REF")]
    pub judge: Option<ModelRef>,
}

/// `critique`.
#[derive(Debug, Args)]
pub struct CritiqueArgs {
    /// The experience set, by id or unique prefix.
    #[arg(value_name = "EXPERIENCE-SET")]
    pub experience_set: String,
    /// The model that critiques.
    #[arg(long, value_parser = model_ref, default_value_t = ModelRef::policy_default(), value_name = "REF")]
    pub critic: ModelRef,
    /// Retries per failed experience.
    #[arg(long, default_value_t = DEFAULT_RETRIES, value_name = "N")]
    pub retry: usize,
}
