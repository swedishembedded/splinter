// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements causal diagnosis of why an agent failed, from
// the evidence its run left, for its clients. If your team needs expertise in
// telling model failures from infrastructure failures before you train on
// them, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The grammar of `experiences diagnose`, `experiences capabilities`, `experiences import`, `experiences compare-arms` and the strata `experiences compare`
//! reports by.

use std::path::PathBuf;

use clap::Args;
use splinter_sdk::measure::strata::StratumCutoffs;
use splinter_sdk::seed::Seed;

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

/// `experiences capabilities`: competence per capability of the examiner's
/// taxonomy, per policy, from bench exports or practice runs.
#[derive(Debug, Args)]
pub struct CapabilitiesArgs {
    /// The export directories, or practice run directories (each holds
    /// report.json and export/; its material's practice.json labels its
    /// tasks).
    #[arg(required = true, value_name = "DIR")]
    pub dirs: Vec<PathBuf>,
    /// A practice directory whose practice.json labels tasks the runs'
    /// material does not (staged before the examiner named capabilities);
    /// a task is labelled from it only when its source digest matches.
    /// Repeatable.
    #[arg(long = "labels", value_name = "PRACTICE_DIR")]
    pub labels: Vec<PathBuf>,
    /// The policy the trials of an export without a practice run's report
    /// count under.
    #[arg(long, value_name = "NAME")]
    pub policy: Option<String>,
}

/// `experiences import`: experience sets from another state root, each
/// experience checked against its content address.
#[derive(Debug, Args)]
pub struct ImportArgs {
    /// The state root to read the sets from (only read; import from a copy of
    /// a root another process writes).
    #[arg(long = "from", value_name = "DIR")]
    pub from: PathBuf,
    /// The sets, by id or unique prefix in that root.
    #[arg(required = true, value_name = "SET")]
    pub sets: Vec<String>,
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

/// `experiences compare-arms`: graded trials of several arms on the same
/// tasks, each arm against a reference arm.
#[derive(Debug, Args)]
pub struct CompareArmsArgs {
    /// An arm and the Harbor job directory of its trials (one subdirectory
    /// with a result.json per trial), as NAME=DIR; repeat for each arm.
    #[arg(long = "arm", value_name = "NAME=DIR")]
    pub arms: Vec<String>,
    /// A trials file: one JSON object per line, {"task", "arm", "family"?,
    /// "attempt"?, "pass" or "reward", "status"?} with status graded
    /// (default), timeout or infra_error; repeat for several.
    #[arg(long = "trials", value_name = "FILE")]
    pub trials: Vec<PathBuf>,
    /// The arm every other arm is compared with.
    #[arg(long, value_name = "NAME")]
    pub reference: String,
    /// A JSON object naming each task's family ({"task": "family"}): the
    /// difference is then also resampled by whole families.
    #[arg(long, value_name = "FILE")]
    pub families: Option<PathBuf>,
    /// A k to report pass@k at (unbiased, over the tasks with at least k
    /// graded attempts); repeat for several. pass@1 is always reported.
    #[arg(long = "k", value_name = "K")]
    pub k: Vec<usize>,
    /// Resamples per bootstrap interval.
    #[arg(long, value_name = "N", default_value_t = splinter_sdk::arm_compare::DEFAULT_RESAMPLES)]
    pub resamples: usize,
    /// The share of resampled means an interval holds.
    #[arg(long, value_name = "SHARE", default_value_t = splinter_sdk::arm_compare::DEFAULT_LEVEL)]
    pub level: f64,
    /// The seed the intervals are resampled with (drawn and printed when not
    /// given; the same trials and seed give the same intervals).
    #[arg(long, value_name = "N")]
    pub interval_seed: Option<Seed>,
}
