// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The command line's grammar: one verb per pipeline stage, the thing acted
//! on as a positional argument, one model-reference syntax, and four global
//! flags every command takes. Defaults come from the crates that own them.

use std::path::PathBuf;
use std::time::Duration;

use clap::{ArgGroup, Args, Parser, Subcommand};
use splinter_sdk::budget::parse_budget;
use splinter_sdk::curriculum::frontier::{PassAtK, DEFAULT_K, DEFAULT_SAMPLING};
use splinter_sdk::datasets::{
    parse_advantage, parse_reward_metric, parse_strength, parse_strip, AdvantageKind, StripPolicy,
    ViewName,
};
use splinter_sdk::eval::SuiteChoice;
use splinter_sdk::seed::Seed;
use splinter_sdk::vocabulary::annotation::Strength;
use splinter_sdk::vocabulary::model_ref::ModelRef;
use splinter_sdk::vocabulary::terms::UsagePolicy;

/// A learning agent with its own model. Tell it what to learn - a document,
/// a repository, a command's output - and it generates tasks from it,
/// solves and verifies them, trains a candidate on what passed, and
/// releases it only if it measurably improves without forgetting.
///
/// With no command, a REPL on the current policy: each line is handled
/// exactly like `splinter "<line>"`.
#[derive(Debug, Parser)]
#[command(
    name = "splinter",
    version,
    after_help = "Models are named policy:default (or policy:<alias>), \
                  local:<checkpoint>[+<adapter>] or remote:<provider>/<name>; remote ones need \
                  --allow-remote. Every stage stores \
                  what it makes under the state root by content address, so any stage can be \
                  rerun or inspected alone.\n\n\
                  Start here: `splinter doctor` says whether a run can start; \
                  `splinter init --template knowledge-learning` starts a run that learns from a document; \
                  `splinter status` shows what the policy is now; `splinter ask <QUESTION>` \
                  asks it. A worked example is examples/first-run."
)]
pub struct Cli {
    /// Flags every command takes.
    #[command(flatten)]
    pub global: Global,
    /// A sentence, classified by the policy into one of the commands below;
    /// an unclear or unconfident reading is asked back, never run.
    #[arg(value_name = "SENTENCE")]
    pub sentence: Option<String>,
    /// The command.
    #[command(subcommand)]
    pub command: Option<Command>,
}

/// The commands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Continue the attempts that failed or got stuck of an existing
    /// experience set with a teacher, from the state the student actually
    /// reached, and keep the continuations the task's own hidden checks
    /// verify as an experience set a dataset can be built from.
    Corrections(Box<CorrectionsArgs>),
    /// Answer a question with a model, closed-book unless --open-book.
    Ask(AskArgs),
    /// Show the policy in use, recent runs and what the stores hold; with a
    /// factory run directory, the run's stages and slots.
    Status(RunStatusArgs),
    /// Write a template, a task repository that works as written, into a
    /// directory; writes over no file.
    Init(InitArgs),
    /// List the templates `init` writes.
    Templates,
    /// Check a factory recipe and everything it names, reporting every
    /// problem at once; runs nothing.
    Validate(RecipeArgs),
    /// Show the graph of stages a factory recipe runs and the work each does.
    Plan(RecipeArgs),
    /// Write a factory collection's task directories with the command its
    /// file names (the task repository's own compiler).
    Generate(GenerateArgs),
    /// Run a factory recipe into a run directory, each stage reused when its
    /// product is already committed.
    Run(RunArgs),
    /// Continue a factory run from what it recorded.
    Resume(RunDirArgs),
    /// Ask a running factory run to stop at its next stage or batch.
    Cancel(RunDirArgs),
    /// Show a factory run's products and reports, or open its trials.
    Inspect(InspectArgs),
    /// Start a factory run on a changed recipe, reusing the products of an
    /// earlier run that the change leaves unchanged.
    Fork(ForkArgs),
    /// Print a factory run's report.
    Report(RunDirArgs),
    /// Write a factory run's products as a portable bundle.
    Export(ExportArgs),
    /// File a factory run's trials in the experience database (an optional
    /// index across runs; the run's products stay its record).
    Index(RunDirArgs),
    /// Play one session of a factory run's interactive pass on this
    /// terminal; the run is then continued, so the session reaches the
    /// dataset and what is trained on it.
    Play(PlayArgs),
    /// Check that a run can start here: the state root is writable with room
    /// to spare, the policy's weights fit on a GPU, and every role's model
    /// resolves to weights on disk. Loads no model and uses no network. Exits 1
    /// when a check fails.
    Doctor,
    /// Start the brain servers (one per usable node of a multi-node
    /// allocation, else one here), print where they are and keep them until
    /// interrupted.
    Serve,
    /// Print the `baseline-v1` record of what a registered run runs on: the
    /// three commits, toolchain, binaries, model and tokenizer digests, seeds
    /// and hardware.
    Baseline(BaselineArgs),
    /// Capture and inspect sources.
    #[command(subcommand)]
    Source(SourceCommand),
    /// Compile a learning objective into validated, versioned training
    /// environments: classify how each skill can be verified, generate and
    /// admit environments, red-team their verifiers.
    #[command(subcommand)]
    Env(EnvCommand),
    /// Knowledge objects (rules, thresholds, deadlines, exceptions, facts,
    /// procedure steps) a model proposes from documents, admitted by code.
    #[command(subcommand)]
    Objects(ObjectsCommand),
    /// Check an experiment plan: which families each generation trains on,
    /// which earlier ones it is held to, which are the sealed final exam.
    #[command(subcommand)]
    Experiment(ExperimentCommand),
    /// Learn from the sessions you held with an agent: take them in, extract
    /// what you taught, teach it from many records, train again from the base
    /// and release only if every claim the champion answered still is and a
    /// new one is.
    Absorb(Box<AbsorbArgs>),
    /// Take the generation the policy is in into the weights: a full
    /// checkpoint, trained from the generation's base on its claims (or the
    /// adapter folded in), gated against the release in use, becomes the base
    /// of the next generation, whose claim set starts empty. The claims it
    /// holds must keep being answered at every release after.
    Consolidate(Box<ConsolidateArgs>),
    /// Choose how to improve an agent on a target capability from what each
    /// way was measured to do, net of what it costs: more samples, another
    /// episode protocol, a trained candidate, or no change.
    Improve(ImproveArgs),
    /// Take in recorded agent sessions, secrets removed.
    #[command(subcommand)]
    Session(SessionCommand),
    /// What a person taught in sessions: claims proposed by a model, ruled
    /// on by code, and the ledger of the rulings.
    #[command(subcommand)]
    Claims(ClaimsCommand),
    /// Generate and inspect task sets.
    #[command(subcommand)]
    Tasks(TasksCommand),
    /// Solve every task of a task set, each in the environment it records.
    Solve(SolveArgs),
    /// Check a factory run: every product against its blobs and its view;
    /// or append verdicts to an experience set (re-grades without
    /// re-running).
    Verify(VerifyArgs),
    /// Critique the failed experiences of a set and retry them with it.
    Critique(CritiqueArgs),
    /// Measure a judge against labelled experiences.
    #[command(subcommand)]
    Judge(JudgeCommand),
    /// Inspect experience sets and experiences.
    #[command(subcommand)]
    Experiences(ExperiencesCommand),
    /// Build datasets from experience sets, and export them.
    #[command(subcommand)]
    Dataset(DatasetCommand),
    /// Goal-directed learning: what may enter a selection of material.
    #[command(subcommand)]
    Goal(GoalCommand),
    /// Train a candidate adapter on datasets; it is not released.
    Train(TrainArgs),
    /// Build a rehearsal dataset: the base model's own answers to general
    /// tasks - sums and format requests built by code, and requests the
    /// base writes itself - never an anchor task or a near copy of one; for
    /// `train --rehearsal`, so a fine-tune keeps what the base does.
    Rehearse(RehearseArgs),
    /// Run the release gate on a candidate and release it if every check
    /// passes; or list the releases.
    Release(ReleaseArgs),
    /// Point an alias back at the release its current one was trained
    /// from.
    Rollback {
        /// The alias, e.g. default.
        alias: String,
    },
    /// Evaluate a factory run's candidates under another plan into a new run
    /// (`RUN --plan PLAN --out DIR`); grade a candidate or a model
    /// closed-book on a suite; or freeze and show the anchor suite.
    Eval(EvalArgs),
    /// Put a candidate and the policy it continues to its held-out tasks and
    /// have a calibrated judge compare them; or, with --exam-set, to a frozen
    /// exam of reserved families, with every verdict kept.
    Exam(ExamArgs),
    /// Choose the step of a run kept with its evaluations by what each writes
    /// on the dev suite: right answers, no invented specifics, then voice.
    Select(SelectArgs),
    /// Analyses done after an exam from what its report kept: labels to
    /// calibrate the judge by, and memorisation.
    #[command(subcommand)]
    ExamReport(ExamReportCommand),
    /// Make and show frozen exams: families reserved from sources, tasks
    /// written from their text alone.
    #[command(subcommand)]
    ExamSet(ExamSetCommand),
    /// List, inspect and cancel runs.
    #[command(subcommand)]
    Runs(RunsCommand),
    /// Submit work that runs outside this process, list it, reconcile it
    /// after a restart, retry a failed attempt, show one job or cancel one;
    /// plain `jobs` lists the subcommands.
    Jobs(JobsArgs),
    /// Show, maintain, verify, repair, archive and restore the state: the
    /// experience database and the files it tracks.
    #[command(subcommand)]
    State(StateCommand),
    /// Trace any artifact up to where it came from (down to source bytes)
    /// and down to everything that came from it.
    Lineage(LineageArgs),
    /// Publish experience sets, datasets, candidates and signed aggregate
    /// verdicts as one tagged, immutable revision of a private Hugging Face
    /// dataset repository; sealed and item-level material is refused.
    Publish(PublishArgs),
    /// Print the command reference as Markdown, generated from this grammar;
    /// with --check FILE, fail when a committed copy of it is stale; with
    /// --check-invocations FILE..., fail on a command a manual shows that
    /// this grammar rejects.
    #[command(hide = true)]
    CliReference(CliReferenceArgs),
}

/// A model reference, for clap.
fn model_ref(text: &str) -> Result<ModelRef, String> {
    text.parse()
        .map_err(|e: splinter_sdk::vocabulary::model_ref::RefError| e.to_string())
}

fn budget(text: &str) -> Result<Duration, String> {
    parse_budget(text).map_err(|e| e.to_string())
}

fn view(text: &str) -> Result<ViewName, String> {
    text.parse().map_err(|e: splinter_sdk::Error| e.to_string())
}

fn strip(text: &str) -> Result<StripPolicy, String> {
    parse_strip(text).map_err(|e| e.to_string())
}

fn share(text: &str) -> Result<f64, String> {
    let value: f64 = text
        .parse()
        .map_err(|_| format!("{text:?} is not a number"))?;
    if value > 0.0 && value <= 1.0 {
        Ok(value)
    } else {
        Err(format!("{value} is not a share in (0, 1]"))
    }
}

fn reward_metric(text: &str) -> Result<(String, f64), String> {
    parse_reward_metric(text).map_err(|e| e.to_string())
}

fn advantage(text: &str) -> Result<AdvantageKind, String> {
    parse_advantage(text).map_err(|e| e.to_string())
}

fn strength(text: &str) -> Result<Strength, String> {
    parse_strength(text).map_err(|e| e.to_string())
}

/// A fraction from 0 to 1, both included.
fn fraction(text: &str) -> Result<f64, String> {
    let share: f64 = text.parse().map_err(|e| format!("{text:?}: {e}"))?;
    if (0.0..=1.0).contains(&share) {
        Ok(share)
    } else {
        Err(format!("{text:?} is not between 0 and 1"))
    }
}

fn voice_share(text: &str) -> Result<f64, String> {
    let share: f64 = text.parse().map_err(|e| format!("{text:?}: {e}"))?;
    if (0.0..1.0).contains(&share) {
        Ok(share)
    } else {
        Err(format!("{text:?} is not a share in [0, 1)"))
    }
}

fn suite(text: &str) -> Result<SuiteChoice, String> {
    text.parse().map_err(|e: splinter_sdk::Error| e.to_string())
}

/// pass@k's parameters, for the commands that measure it.
#[derive(Debug, Args)]
pub struct PassAtKArgs {
    /// Attempts per task when measuring pass@k.
    #[arg(long = "k", default_value_t = DEFAULT_K, value_name = "N")]
    pub k: usize,
    /// The policy's sampling temperature for its pass@k attempts.
    #[arg(long, value_name = "T", help = format!(
        "The policy's sampling temperature for its pass@k attempts [default for a local \
         model: {}; a remote model samples as its server does unless this is given]",
        DEFAULT_SAMPLING.temperature
    ))]
    pub temperature: Option<f32>,
    /// The policy's top-k cut for its pass@k attempts.
    #[arg(long, value_name = "N", help = format!(
        "The policy's top-k cut for its pass@k attempts [default for a local model: {}]",
        DEFAULT_SAMPLING.top_k
    ))]
    pub top_k: Option<u32>,
}

impl PassAtKArgs {
    /// The parameters.
    #[must_use]
    pub fn pass_at_k(&self) -> PassAtK {
        PassAtK::new(self.k, self.temperature, self.top_k)
    }
}

/// What retrieval is asked to do, for `ask` and `exam`.
#[derive(Debug, Args)]
pub struct RetrieveArgs {
    /// Show the model the passages of these stored sources that bear on the
    /// question, found by meaning and by exact words, instead of all of one.
    #[arg(long = "retrieve", value_name = "SOURCE-ID", num_args = 1..)]
    pub sources: Vec<String>,
    /// How many passages `--retrieve` shows.
    #[arg(long, default_value_t = 6, value_name = "N", requires = "sources")]
    pub passages: usize,
    /// A model that reads the candidates search found beside the question
    /// and moves the ones that bear on it forward.
    #[arg(long, value_parser = model_ref, value_name = "REF", requires = "sources")]
    pub reranker: Option<ModelRef>,
}

/// `ask`.
#[derive(Debug, Args)]
pub struct AskArgs {
    /// The question.
    pub question: String,
    /// Show the model this stored source's text with the question.
    #[arg(long, value_name = "SOURCE-ID")]
    pub open_book: Option<String>,
    /// Retrieval in place of the whole text of one source.
    #[command(flatten)]
    pub retrieval: RetrieveArgs,
    /// The model asked.
    #[arg(long, value_parser = model_ref, default_value_t = ModelRef::policy_default(), value_name = "REF")]
    pub policy: ModelRef,
}

/// `source ...`.
#[derive(Debug, Subcommand)]
pub enum SourceCommand {
    /// Capture a file (a document), a directory (a repository) or
    /// cmd:COMMAND... (a command run), and store it. URLs are refused.
    Add {
        /// PATH, or cmd: followed by the command and its arguments.
        #[arg(
            required = true,
            num_args = 1..,
            trailing_var_arg = true,
            allow_hyphen_values = true,
            value_name = "PATH|cmd:COMMAND..."
        )]
        target: Vec<String>,
        /// The usage policy the content is under: redistributable,
        /// research_only, noncommercial, restricted_DUA or unknown. Without
        /// it the terms are unstated, which no release can be distributed
        /// under. The policy is part of the source's identity.
        #[arg(long, value_parser = usage_policy, value_name = "POLICY")]
        usage_policy: Option<UsagePolicy>,
    },
    /// List the stored sources.
    List,
    /// Show one stored source.
    Show {
        /// The source's id, or a unique prefix of it.
        id: String,
    },
}

/// `solve`.
#[derive(Debug, Args)]
#[command(group = ArgGroup::new("pass_at_k_given")
    .args(["k", "temperature", "top_k", "teacher"])
    .multiple(true)
    .requires("frontier"))]
pub struct SolveArgs {
    /// The task set, by id or unique prefix.
    #[arg(value_name = "TASKSET-ID")]
    pub task_set: String,
    /// The model that solves.
    #[arg(long, value_parser = model_ref, default_value_t = ModelRef::policy_default(), value_name = "REF")]
    pub solver: ModelRef,
    /// Solve each task k times, grade every attempt, have a teacher solve
    /// open-book each task never solved, and keep the tasks worth training
    /// on: failed at least sometimes, with a verified answer.
    #[arg(long)]
    pub frontier: bool,
    /// With --frontier: the model that solves open-book the tasks the
    /// solver never solves (default: the solver).
    #[arg(long, value_parser = model_ref, value_name = "REF")]
    pub teacher: Option<ModelRef>,
    /// How the frontier is measured.
    #[command(flatten)]
    pub pass_at_k: PassAtKArgs,
    /// Run this sven harness document on each workspace task in place of the
    /// conversational agent loop; the attempts are verified and compared
    /// like any other (`experiences compare`).
    #[arg(long, value_name = "FILE", conflicts_with_all = ["frontier", "teacher"])]
    pub harness: Option<PathBuf>,
    /// Resume an interrupted solve: the experience set its last progress
    /// line named. A task that set already answers is not solved again, and
    /// its experiences are members of the new set.
    #[arg(long, value_name = "EXPERIENCE-SET", conflicts_with = "frontier")]
    pub resume: Option<String>,
    /// Record every episode whole: the solver keeps its conversation instead of
    /// compacting it when it nears the context limit, so a failure can be
    /// continued from its own transcript. A compacted transcript has no
    /// faithful prefix and cannot be corrected, so this is the flag an arm
    /// run for a correction corpus passes.
    #[arg(long)]
    pub no_compaction: bool,
}

/// `judge ...`.
#[derive(Debug, Subcommand)]
pub enum JudgeCommand {
    /// Measure a judge on labelled experiences and keep the calibration
    /// `verify --judge` gates its verdicts by.
    Calibrate {
        /// JSON Lines: {"experience": "<id>", "label": "pass"|"fail"}.
        #[arg(value_name = "LABELLED-FILE")]
        labelled: PathBuf,
        /// The judge.
        #[arg(long, required = true, value_parser = model_ref, value_name = "REF")]
        judge: ModelRef,
    },
    /// Measure a judge on the controls a task set's own references give -
    /// each task's reference as the right answer, another family's as the
    /// wrong one - and show its precision, whether it would be trusted and
    /// the controls it got wrong. A calibration that rests on enough controls
    /// is kept, and a run reuses it.
    Measure {
        /// The task set, by id or unique prefix.
        #[arg(value_name = "TASK-SET")]
        tasks: String,
        /// The judge.
        #[arg(long, required = true, value_parser = model_ref, value_name = "REF")]
        judge: ModelRef,
        /// Measure it as a judge of fit - is a passage a natural reply to its
        /// message - and not of whether an answer gives the reference.
        #[arg(long)]
        fit: bool,
    },
}

/// `experiences ...`.
#[derive(Debug, Subcommand)]
pub enum ExperiencesCommand {
    /// List the stored experience sets.
    List,
    /// Show an experience set, or one experience with its annotations.
    Show {
        /// An experience set's or an experience's id, or a unique prefix.
        id: String,
        /// Show the relations around it instead: retries, critiques,
        /// preferences.
        #[arg(long)]
        graph: bool,
    },
    /// Run an experience's code calls again in the environment it records
    /// (or every experience of a set) and compare each result with the one
    /// it observed.
    Replay {
        /// An experience set's or an experience's id, or a unique prefix.
        id: String,
    },
    /// Read the trials Harbor left for imported tasks (each a trial directory
    /// with result.json and agent/trajectory.json in ATIF) as experiences
    /// carrying Harbor's own verdict: a reward of exactly 1 passes, any other
    /// fails, no reward or an exception is no verdict.
    IngestHarbor {
        /// The task set the trials' tasks were imported into.
        #[arg(value_name = "TASKSET-ID")]
        task_set: String,
        /// The trial directories.
        #[arg(required = true, value_name = "TRIAL")]
        trials: Vec<PathBuf>,
    },
    /// Admit the trials of exports written by `bench export` as experiences:
    /// every file is checked against the digest in the trial's manifest and
    /// the manifest against the export's index, a tampered or partial trial is
    /// refused with the reason, the benchmark's grade is the verdict and the
    /// ATIF trajectory the record. Ingesting an export again changes nothing;
    /// a trial name with other content is refused. The examiner's score
    /// record must agree with the grade; a trial it does not count has no
    /// verdict, never a fail. A confidential trial, and an export of sealed
    /// material (`sealed-export.json` beside it), are refused. Trials come
    /// out in one experience set per split: `train` trials can be built into
    /// datasets, `eval` trials are examination material no dataset accepts.
    IngestBench {
        /// The export directories (each holds index.jsonl).
        #[arg(required = true, value_name = "EXPORT")]
        exports: Vec<PathBuf>,
        /// The split of trials whose export names none (exports made before
        /// the benchmark recorded splits). Without it such a trial is refused:
        /// a split is never guessed. The register records that the operator
        /// stated it, and a trial that names its split must agree.
        #[arg(long, value_enum, value_name = "SPLIT")]
        assume_set: Option<BenchSplit>,
    },
    /// Why each failed trial of bench exports failed: one failure class per
    /// trial, from its evidence, with the step and excerpt that decided it.
    Diagnose(DiagnoseArgs),
    /// A policy's competence per capability of the examiner's taxonomy, from
    /// bench exports or practice runs: pass rates with 95% intervals,
    /// failure classes, frontier bands and the curriculum's weight, kept in
    /// the store with where every number came from.
    Capabilities(CapabilitiesArgs),
    /// Import experience sets, with their experiences' annotations and trial
    /// register, from another state root.
    Import(ImportArgs),
    /// Ingest a bundle `publish` wrote, at a pinned revision of its Hugging
    /// Face dataset repository: every file checked against the manifest's
    /// digest, everything stored under its own content address.
    IngestHf(IngestHfArgs),
    /// Compare two experience sets of the same tasks: the candidate's against
    /// the baseline's, paired by task, a family counted once. Exposed and
    /// harness-failed attempts are left out and counted.
    Compare {
        /// The candidate's experience set, by id or unique prefix.
        #[arg(value_name = "CANDIDATE-SET")]
        candidate: String,
        /// The baseline's experience set, by id or unique prefix.
        #[arg(value_name = "BASELINE-SET")]
        baseline: String,
        /// The seed the intervals are resampled with (drawn and printed when
        /// not given; the same sets and seed give the same intervals).
        #[arg(long, value_name = "N")]
        interval_seed: Option<Seed>,
        /// Report the paired tasks by difficulty stratum, next to the family
        /// test: a task's stratum is how many of checkers, footprint and
        /// instruction length exceed their cutoffs (a value equal to its
        /// cutoff counts as not above). Report only; no threshold. Needs all
        /// three cutoffs.
        #[command(flatten)]
        strata: StrataArgs,
    },
    /// Compare graded trials of several arms (Harbor jobs or trials files)
    /// with a reference arm: pass@1 and unbiased pass@k, the paired
    /// difference with task and family bootstrap intervals, exact one- and
    /// two-sided sign tests, Holm-adjusted; timeouts and infrastructure
    /// errors counted apart.
    CompareArms(Box<CompareArmsArgs>),
}

/// `dataset ...`.
#[derive(Debug, Subcommand)]
pub enum DatasetCommand {
    /// Project experience sets through a view into a stored dataset.
    Build(Box<BuildArgs>),
    /// Build a policy-gradient dataset from graded conversations with a
    /// counterpart (a client, a student): each line of the input names the
    /// case, the model that played the learner, the reward the case's grade
    /// earned and the conversation record. Only conversations the admission
    /// rules accept are used; the learner's turns are the actions.
    Conversations(ConversationsArgs),
    /// Write labelled decision fixtures (one example per labelled choice
    /// question) from the run records of an ablation benchmark output
    /// directory, split by task family and seed so that no task is in two
    /// splits. The output is a JSON-lines file, not a stored dataset.
    DecisionFixtures(DecisionFixturesArgs),
    /// Have the trainer read a stored dataset before any device is claimed.
    Check(CheckArgs),
    /// Copy a dataset and its manifest into a directory.
    Export {
        /// The dataset, by id or unique prefix.
        #[arg(value_name = "DATASET-ID")]
        id: String,
        /// The directory to copy into.
        #[arg(long, required = true, value_name = "DIR")]
        out: PathBuf,
    },
}

/// `--usage-policy`'s value.
fn usage_policy(text: &str) -> Result<UsagePolicy, String> {
    UsagePolicy::parse(text)
}

/// `status`.
#[derive(Debug, Args)]
pub struct RunStatusArgs {
    /// A factory run directory; none shows the policy.
    #[arg(value_name = "RUN")]
    pub run: Option<PathBuf>,
}

mod absorb;
mod baseline;
mod bench;
mod consolidate;
mod control;
mod corrections;
mod dataset;
mod env;
mod eval;
mod exam;
mod experiences;
mod factory;
mod gate;
mod global;
mod goal;
mod harbor;
mod hub;
mod improve;
mod invocations;
mod jobs;
mod objects;
pub mod reference;
mod review;
mod state;
mod tasks;
mod train;
pub use absorb::{AbsorbArgs, ClaimsCommand, SessionCommand};
pub use baseline::BaselineArgs;
pub use bench::BenchSplit;
pub use consolidate::ConsolidateArgs;
pub use control::ControlKind;
pub use corrections::CorrectionsArgs;
pub use dataset::{BuildArgs, CheckArgs, ConversationsArgs, DecisionFixturesArgs};
pub use env::{EnvCommand, ObjectiveArgs};
pub use eval::{
    EvalArgs, EvalCommand, ExperimentCommand, LineageArgs, ReleaseArgs, ReleaseCommand,
};
pub use exam::{ExamArgs, ExamReportCommand, ExamSetArgs, ExamSetCommand, PowerArgs, SelectArgs};
pub use experiences::{CapabilitiesArgs, CompareArmsArgs, DiagnoseArgs, ImportArgs, StrataArgs};
pub use factory::{
    ExportArgs, ForkArgs, GenerateArgs, InitArgs, InspectArgs, PlayArgs, RecipeArgs, RunArgs,
    RunDirArgs,
};
pub use gate::GateArgs;
pub use global::Global;
pub use goal::GoalCommand;
pub use harbor::{HarborCommand, OracleArg};
pub use hub::{HubArgs, IngestHfArgs, PublishArgs};
pub use improve::ImproveArgs;
pub use jobs::{help as jobs_help, JobsArgs, JobsCommand};
pub use objects::ObjectsCommand;
pub use reference::CliReferenceArgs;
pub use review::{CritiqueArgs, VerifyArgs};
pub use state::{RunsCommand, StateCommand};
pub use tasks::TasksCommand;
use train::positive_count;

/// A sampling temperature a provider accepts: 0 to 2.
pub use train::{OptimiserArgs, RehearseArgs, TrainArgs};

#[cfg(test)]
mod factory_tests;
#[cfg(test)]
mod tests;
