// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements causal diagnosis of why an agent failed, from
// the evidence its run left, for its clients. If your team needs expertise in
// telling model failures from infrastructure failures before you train on
// them, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The grammar of `experiences diagnose`, `experiences repair` and the strata `experiences compare`
//! reports by.

use std::path::PathBuf;

use clap::Args;
use splinter_sdk::bench::repair::{DEFAULT_MAX_REPAIRS, DEFAULT_REPAIR_JOBS};
use splinter_sdk::measure::strata::StratumCutoffs;
use splinter_sdk::vocabulary::model_ref::ModelRef;
use splinter_sdk::vocabulary::outcome::FailureClass;

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

/// `experiences repair`: a teacher continues the failed attempts of practice
/// runs from their last good step; the task's verifier keeps a repair only on
/// a pass.
#[derive(Debug, Args)]
pub struct RepairArgs {
    /// The practice run directories (each holds report.json, export/ and
    /// staged/), as `tasks harbor practice` leaves them under the state root.
    #[arg(required = true, value_name = "RUN")]
    pub runs: Vec<PathBuf>,
    /// The model that continues the attempts (`remote:`/`local:`).
    #[arg(long, value_parser = super::model_ref, value_name = "REF")]
    pub teacher: ModelRef,
    /// The examiner's checkout, which lays out and grades the workspaces.
    #[arg(long, value_name = "DIR")]
    pub bench: PathBuf,
    /// The most repairs this run makes.
    #[arg(long, value_name = "N", value_parser = super::positive_count,
          help = format!("The most repairs this run makes [default: {DEFAULT_MAX_REPAIRS}]"))]
    pub max: Option<usize>,
    /// Repairs run at once: the teacher server's decode slots.
    #[arg(long, value_name = "N", value_parser = super::positive_count,
          help = format!("Repairs run at once [default: {DEFAULT_REPAIR_JOBS}]"))]
    pub jobs: Option<usize>,
    /// Only failures of this class (repeatable; `wrong_content`, `tool`, ...).
    #[arg(long = "class", value_parser = failure_class, value_name = "CLASS")]
    pub classes: Vec<FailureClass>,
}

fn failure_class(name: &str) -> Result<FailureClass, String> {
    FailureClass::parse(name)
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
