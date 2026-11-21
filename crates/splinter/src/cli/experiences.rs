// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements causal diagnosis of why an agent failed, from
// the evidence its run left, for its clients. If your team needs expertise in
// telling model failures from infrastructure failures before you train on
// them, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The grammar of `experiences diagnose`, `experiences capabilities`, `experiences repair`, `experiences rationalise`, `experiences import`, `experiences compare-arms` and the strata `experiences compare`
//! reports by.

use std::path::PathBuf;

use clap::Args;
use splinter_sdk::bench::repair::hinted::{DEFAULT_RATIONALISE_JOBS, DEFAULT_SAMPLES};
use splinter_sdk::bench::repair::{
    InterventionKind, Proposer, DEFAULT_BUDGET, DEFAULT_MAX_REPAIRS, DEFAULT_REPAIR_JOBS,
};
use splinter_sdk::measure::strata::StratumCutoffs;
use splinter_sdk::seed::Seed;
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
    /// The student's own model (`remote:`/`local:`): a retry samples it
    /// again from the last good step, and a decomposition's subgoals are
    /// worked by it. Without it neither runs.
    #[arg(long, value_parser = super::model_ref, value_name = "REF")]
    pub student: Option<ModelRef>,
    /// Who proposes a decomposition's subgoals: `teacher` or `student`.
    #[arg(long, value_name = "WHO", default_value = "teacher")]
    pub proposer: Proposer,
    /// A public reference document, or a directory of them (`.md`, `.txt`,
    /// `.rst`), a knowledge-assisted correction retrieves from (repeatable).
    /// Sealed material (a verifier's `tests`, a `solution`, an `oracle`,
    /// `truth.json`, run data) is refused when named and left out of a
    /// directory. Without it the knowledge-assisted correction does not run.
    #[arg(long = "knowledge", value_name = "PATH")]
    pub knowledge: Vec<PathBuf>,
    /// Only this intervention (repeatable: `retry`, `fork_correction`,
    /// `knowledge_correction`, `decompose`); without it every one the
    /// models and knowledge given allow, in the order each failure class's
    /// chain tries them.
    #[arg(long = "intervention", value_name = "NAME")]
    pub interventions: Vec<InterventionKind>,
    /// Attempts one failure may cost, every intervention of its chain and
    /// every sample counted.
    #[arg(long, value_name = "N", value_parser = super::positive_count,
          help = format!("Attempts one failure may cost, every intervention and sample counted [default: {DEFAULT_BUDGET}]"))]
    pub budget: Option<usize>,
    /// The examiner's checkout, which lays out and grades the workspaces.
    #[arg(long, value_name = "DIR")]
    pub bench: PathBuf,
    /// The most failures this run repairs.
    #[arg(long, value_name = "N", value_parser = super::positive_count,
          help = format!("The most failures this run repairs [default: {DEFAULT_MAX_REPAIRS}]"))]
    pub max: Option<usize>,
    /// Failures repaired at once: the teacher server's decode slots.
    #[arg(long, value_name = "N", value_parser = super::positive_count,
          help = format!("Failures repaired at once [default: {DEFAULT_REPAIR_JOBS}]"))]
    pub jobs: Option<usize>,
    /// The most repairs of one task, the ones a run already graded counted:
    /// spreads a run across task families.
    #[arg(long, value_name = "N", value_parser = super::positive_count)]
    pub per_task: Option<usize>,
    /// The seed that orders the tasks and the attempts of each (drawn and
    /// printed when not given).
    #[arg(long, value_name = "N")]
    pub seed: Option<Seed>,
    /// Samples of each intervention of a failure's chain, each a repair of
    /// its own (default 1).
    #[arg(long, value_name = "K", value_parser = super::positive_count)]
    pub samples: Option<usize>,
    /// Only failures of this class (repeatable; `wrong_content`, `tool`, ...).
    #[arg(long = "class", value_parser = failure_class, value_name = "CLASS")]
    pub classes: Vec<FailureClass>,
}

/// `experiences rationalise`: the hinted retry. The rationaliser works the
/// student's failed train-side tasks of a compiled Harbor dataset with each
/// task's reference solution as a hint, as Harbor trials through the
/// examiner's sandbox, gateway and Sven adapter; an attempt is kept when it
/// is verified and no answer content (the hint's wording, a file line
/// nothing showed yet) is in the model's own text. `--regrade` judges an
/// earlier run's attempts again instead, offline unless `--sufficiency`.
#[derive(Debug, Args)]
pub struct RationaliseArgs {
    /// The compiled dataset (holding bench-compile.json; hints come only
    /// from its train-side tasks).
    #[arg(long, value_name = "DIR")]
    pub dataset: PathBuf,
    /// A student's graded Harbor job, packed (`bench pack`) or not
    /// (repeatable; optional with `--regrade`); every trial must be of a
    /// train-side task of the dataset, and its passes are kept as they are.
    /// Never changed: a packed job is unpacked in a copy.
    #[arg(
        long = "student",
        required_unless_present = "regrade",
        value_name = "JOB"
    )]
    pub students: Vec<PathBuf>,
    /// The model that rationalises: a local model the brain endpoint
    /// (`--brain-endpoint`, `--brain-key-file`) serves. Needed when trials
    /// run (not for an offline `--regrade`).
    #[arg(long, value_parser = super::model_ref, value_name = "REF",
          required_unless_present = "regrade")]
    pub rationaliser: Option<ModelRef>,
    /// The examiner's checkout (`./bench` and its Harbor run from it; `bench
    /// unpack` reads packed jobs).
    #[arg(long, value_name = "DIR")]
    pub bench: PathBuf,
    /// The `harbor` program (default: the checkout's .venv/bin/harbor)
    #[arg(long, value_name = "PATH")]
    pub harbor: Option<PathBuf>,
    /// The sven build the agent runs, given to every sandbox. Needed when
    /// trials run.
    #[arg(long, value_name = "PATH", required_unless_present = "regrade")]
    pub sven_build: Option<PathBuf>,
    /// The run profile the rationaliser's is derived from: its sampling and
    /// sven settings, its main role forwarded to the rationaliser. Needed
    /// when trials run.
    #[arg(long, value_name = "FILE", required_unless_present = "regrade")]
    pub profile: Option<PathBuf>,
    /// The examiner's cache of task images (default:
    /// .bench-data/rootfs-cache in the checkout).
    #[arg(long, value_name = "DIR")]
    pub images: Option<PathBuf>,
    /// Failed tasks to pick, stratified by difficulty rung.
    #[arg(long, value_name = "N", value_parser = super::positive_count,
          required_unless_present = "regrade", conflicts_with = "regrade")]
    pub count: Option<usize>,
    /// The seed of the pick (drawn and printed when not given; a rerun with
    /// the same seed resumes the run).
    #[arg(long, value_name = "N", conflicts_with = "regrade")]
    pub seed: Option<Seed>,
    /// Attempts of each picked task, each its own record.
    #[arg(long, value_name = "K", value_parser = super::positive_count, conflicts_with = "regrade",
          help = format!("Attempts of each picked task [default: {DEFAULT_SAMPLES}]"))]
    pub samples: Option<usize>,
    /// Trials run at once.
    #[arg(long, value_name = "N", value_parser = super::positive_count,
          help = format!("Trials run at once [default: {DEFAULT_RATIONALISE_JOBS}]"))]
    pub jobs: Option<usize>,
    /// A word or phrase whose mention in the model's own text also rejects
    /// an attempt (repeatable; none by default: naming a hint is no leak,
    /// only the answer's content is).
    #[arg(long = "leak-term", value_name = "TEXT")]
    pub leak_terms: Vec<String>,
    /// Also require real observations: the first tool call shows every file
    /// the task changes before writing it.
    #[arg(long)]
    pub real_observations: bool,
    /// Also require sufficiency: a second Harbor trial whose instruction
    /// carries only the attempt's reasoning, no hint, must pass.
    #[arg(long)]
    pub sufficiency: bool,
    /// How the hint is framed for the rationaliser: `worked-answer` (a worked
    /// answer for its eyes only) or `private-scratch` (private scratch
    /// knowledge: work as if you found the approach yourself, observe the
    /// files through the tool first, never mention or allude to a hint).
    #[arg(
        long,
        value_enum,
        default_value = "worked-answer",
        conflicts_with = "regrade"
    )]
    pub hint_framing: HintFramingArg,
    /// An earlier `experiences rationalise` run's directory whose graded
    /// attempts are judged again with this run's checks, each with its own
    /// run's hint framing, instead of drawing and running new ones
    /// (repeatable). The earlier run is only read: its packed jobs are
    /// unpacked in a copy. Without `--sufficiency` nothing runs.
    #[arg(long = "regrade", value_name = "RUN_DIR")]
    pub regrade: Vec<PathBuf>,
}

/// The `--hint-framing` choices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum HintFramingArg {
    WorkedAnswer,
    PrivateScratch,
}

fn failure_class(name: &str) -> Result<FailureClass, String> {
    FailureClass::parse(name)
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
