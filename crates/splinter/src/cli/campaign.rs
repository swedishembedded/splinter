// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that run model learning as one
// resumable, multi-generation campaign with a single promotion gate and
// rollback, for its clients. If your team needs expertise in self-improving
// training loops you can procure our services by sending an email to
// info@swedishembedded.com.

//! The grammar of `campaign`.

use std::path::PathBuf;

use clap::{Subcommand, ValueEnum};

/// `campaign`.
#[derive(Debug, Subcommand)]
pub enum CampaignCommand {
    /// Run a campaign's generations: the champion practises on the
    /// examiner's train-split practice tasks it has not mastered, the graded
    /// trials are ingested, a dataset is curated over every generation's
    /// experience, a candidate is trained from the base as a served brain
    /// job, the examiner's signed verdicts on it are gated by the one
    /// promotion gate, and the champion is promoted or kept. Every stage is
    /// recorded before the next starts; run it again after a kill and it
    /// resumes at the first stage without a record, reusing brain's job by
    /// its resume key. Ctrl-C cancels the stage in progress.
    Run {
        /// The campaign file (TOML): its name, base, generations, stage
        /// settings, budget, `[practice]` and `[examine]`.
        #[arg(value_name = "FILE")]
        spec: PathBuf,
        /// Cards a training run holds, to count its GPU-hours.
        #[arg(long, default_value = "1", value_name = "N")]
        gpus: u32,
    },
    /// What a campaign has done: every generation's candidate and decision,
    /// the champion, its history and what was spent. Nothing runs.
    Status {
        /// The campaign file.
        #[arg(value_name = "FILE")]
        spec: PathBuf,
    },
    /// Move the champion back to the one a generation made (0: the base),
    /// recorded with its reason as one more entry of the champion's history.
    Rollback {
        /// The campaign file.
        #[arg(value_name = "FILE")]
        spec: PathBuf,
        /// The generation whose champion to move back to; 0 is the base.
        #[arg(long, value_name = "GENERATION")]
        to: u32,
        /// Why, as the history records it.
        #[arg(long, value_name = "TEXT")]
        reason: String,
    },
    /// The campaign's problem state: the problems that limit the model,
    /// each with its evidence, and the candidate interventions against each
    /// with a belief, revised by every verdict. Kept as an append-only list
    /// of events beside the campaign's ledger; an event recorded again
    /// records nothing.
    #[command(subcommand)]
    Problems(ProblemsCommand),
    /// Propose the next hypothesis: the pending candidate of an open problem
    /// with the highest belief / (attempts on its problem + 1). When the
    /// candidate declares a value, its arm is this campaign file continuing
    /// from the current champion under the same seed with that one change,
    /// checked to change nothing else.
    Next {
        /// The campaign file.
        #[arg(value_name = "FILE")]
        spec: PathBuf,
        /// Propose an engineering candidate instead (a throughput or serving
        /// change, held to the engineering paths).
        #[arg(long)]
        engineering: bool,
        /// Write the arm's campaign file here.
        #[arg(long, value_name = "ARM")]
        write: Option<PathBuf>,
    },
}

/// `campaign problems`.
#[derive(Debug, Subcommand)]
pub enum ProblemsCommand {
    /// Every problem with its evidence and attempts, every candidate with its
    /// belief, priority, status and history, the parameters, and the
    /// state's digest.
    Show {
        /// The campaign file.
        #[arg(value_name = "FILE")]
        spec: PathBuf,
    },
    /// Seed the problem state with a document (TOML, or JSON by its `.json`
    /// extension), or extend it: new problems, new candidates of a known
    /// problem, new parameters from this point on.
    Add {
        /// The campaign file.
        #[arg(value_name = "FILE")]
        spec: PathBuf,
        /// The problem-state document (schema `splinter.problem-state/1`).
        #[arg(value_name = "DOCUMENT")]
        document: PathBuf,
    },
    /// Record one experiment's verdict on one candidate: from an
    /// `experiences compare-arms` report (`--compare`, `--arm`), from an arm
    /// campaign's promotion gate (`--gate`, `--generation`), or stated with
    /// its reason (`--verdict`, `--reason`). Supported adds to the belief and
    /// adopts the candidate, falsified takes from it, inconclusive and
    /// invalid leave it; each counts one attempt against the problem.
    Update {
        /// The campaign file.
        #[arg(value_name = "FILE")]
        spec: PathBuf,
        /// The candidate the experiment tested.
        #[arg(long, value_name = "ID")]
        candidate: String,
        /// The experiment's id: one verdict per experiment and candidate.
        #[arg(long, value_name = "ID")]
        experiment: String,
        /// A verdict stated by hand; needs `--reason`.
        #[arg(long, value_enum, value_name = "VERDICT", conflicts_with_all = ["compare", "gate"])]
        verdict: Option<VerdictArg>,
        /// An `experiences compare-arms --json` report.
        #[arg(long, value_name = "REPORT", requires = "arm", conflicts_with = "gate")]
        compare: Option<PathBuf>,
        /// The report's arm whose contrast against its reference is read.
        #[arg(long, value_name = "NAME", requires = "compare")]
        arm: Option<String>,
        /// The arm's campaign file whose generation's gate is read; it must
        /// differ from FILE in the candidate's variable only.
        #[arg(long, value_name = "ARM", requires = "generation")]
        gate: Option<PathBuf>,
        /// The arm campaign's generation.
        #[arg(long, value_name = "N", requires = "gate")]
        generation: Option<u32>,
        /// Where the result is recorded, as `<kind>:<id>` (repeatable).
        #[arg(long, value_name = "REF")]
        evidence: Vec<String>,
        /// Why, as the history records it.
        #[arg(long, value_name = "TEXT")]
        reason: Option<String>,
    },
    /// A coordinator's revision, recorded with its reason: a candidate's
    /// belief (`--candidate`, `--belief`) or a problem's status
    /// (`--problem`, `--status`).
    Revise {
        /// The campaign file.
        #[arg(value_name = "FILE")]
        spec: PathBuf,
        /// The candidate whose belief is set.
        #[arg(
            long,
            value_name = "ID",
            requires = "belief",
            conflicts_with = "problem"
        )]
        candidate: Option<String>,
        /// The new belief, in [0, 1].
        #[arg(long, value_name = "B", requires = "candidate")]
        belief: Option<f64>,
        /// The problem whose status is set.
        #[arg(long, value_name = "ID", requires = "status")]
        problem: Option<String>,
        /// The new status.
        #[arg(long, value_enum, value_name = "STATUS", requires = "problem")]
        status: Option<StatusArg>,
        /// Why.
        #[arg(long, value_name = "TEXT")]
        reason: String,
    },
    /// Check that an arm's campaign file differs from FILE in one declared
    /// variable only (`--variable`, or the candidate's), and, for an
    /// engineering change, only in the engineering paths.
    Check {
        /// The campaign file of the champion.
        #[arg(value_name = "FILE")]
        spec: PathBuf,
        /// The arm's campaign file.
        #[arg(value_name = "ARM")]
        arm: PathBuf,
        /// The candidate the arm tests: its variable and phase.
        #[arg(long, value_name = "ID", conflicts_with = "variable")]
        candidate: Option<String>,
        /// The one variable the arm may change, as a dotted path.
        #[arg(long, value_name = "PATH")]
        variable: Option<String>,
        /// Hold the arm to the engineering paths too.
        #[arg(long)]
        engineering: bool,
    },
}

/// A verdict as given on the command line.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum VerdictArg {
    /// The effect is shown and at least the target.
    Supported,
    /// An effect the size of the target is ruled out.
    Falsified,
    /// Measured, but neither.
    Inconclusive,
    /// Not evidence: a crash or a measurement the infrastructure decided.
    Invalid,
}

/// A problem's status as given on the command line.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum StatusArg {
    /// It limits the model.
    Open,
    /// It no longer does.
    Resolved,
}
