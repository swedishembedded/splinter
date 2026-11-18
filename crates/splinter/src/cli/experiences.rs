// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements causal diagnosis of why an agent failed, from
// the evidence its run left, for its clients. If your team needs expertise in
// telling model failures from infrastructure failures before you train on
// them, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The grammar of `experiences diagnose` and the strata `experiences compare`
//! reports by.

use std::path::PathBuf;

use clap::Args;
use splinter_sdk::measure::strata::StratumCutoffs;

/// `experiences diagnose`: the failure class of every failed trial of bench
/// exports, counted per domain and attribution.
#[derive(Debug, Args)]
pub struct DiagnoseArgs {
    /// The export directories (each holds index.jsonl; skipped.jsonl when the
    /// export skipped trials), or practice run directories (each holds
    /// report.json and export/).
    #[arg(required = true, value_name = "DIR")]
    pub dirs: Vec<PathBuf>,
}

/// The cutoffs of the difficulty strata, all or none.
#[derive(Debug, Clone, Copy, Args)]
pub struct StrataArgs {
    /// Tests a task's fail-to-pass commands run: more is above.
    #[arg(long, value_name = "N", requires_all = ["stratum_footprint", "stratum_instruction_chars"])]
    pub stratum_checkers: Option<usize>,
    /// Files a task's reference patch changes: more is above.
    #[arg(long, value_name = "N", requires_all = ["stratum_checkers", "stratum_instruction_chars"])]
    pub stratum_footprint: Option<usize>,
    /// Characters of a task's instruction: more is above.
    #[arg(long, value_name = "N", requires_all = ["stratum_checkers", "stratum_footprint"])]
    pub stratum_instruction_chars: Option<usize>,
}

impl StrataArgs {
    /// The cutoffs, when given.
    #[must_use]
    pub fn cutoffs(&self) -> Option<StratumCutoffs> {
        Some(StratumCutoffs {
            checkers: self.stratum_checkers?,
            footprint: self.stratum_footprint?,
            instruction_chars: self.stratum_instruction_chars?,
        })
    }
}
