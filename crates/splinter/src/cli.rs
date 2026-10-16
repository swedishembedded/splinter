// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The command line's grammar: one verb per pipeline stage, the thing acted
//! on as a positional argument, one model-reference syntax, and four global
//! flags every command takes. Defaults come from the crates that own them.

use std::path::PathBuf;
use std::time::Duration;

use clap::{ArgAction, ArgGroup, Args, Parser, Subcommand};
use splinter_sdk::canaries::CANARY_COUNT;
use splinter_sdk::critique::DEFAULT_RETRIES;
use splinter_sdk::curriculum::frontier::{PassAtK, DEFAULT_K, DEFAULT_SAMPLING};
use splinter_sdk::datasets::{parse_strength, parse_strip, StripPolicy, ViewName};
use splinter_sdk::eval::SuiteChoice;
use splinter_sdk::learn::parse_budget;
use splinter_sdk::lineage::Direction;
use splinter_sdk::seed::Seed;
use splinter_sdk::variants::DEFAULT_VARIANTS_PER_TASK;
use splinter_sdk::vocabulary::annotation::Strength;
use splinter_sdk::vocabulary::model_ref::{ModelRef, POLICY_DEFAULT};
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
                  `splinter learn <SOURCE> --kinds <KINDS>` learns from a document; \
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

/// Flags every command takes, meaning the same everywhere.
#[derive(Clone, Debug, Default, Args)]
pub struct Global {
    /// The state root (default: $SPLINTER_STATE, else ~/.splinter).
    #[arg(long, global = true, value_name = "DIR")]
    pub state: Option<PathBuf>,
    /// Print machine-readable JSON on stdout instead of text.
    #[arg(long, global = true)]
    pub json: bool,
    /// Report more as work goes (each stage's full summary).
    #[arg(short, long, global = true, action = ArgAction::Count)]
    pub verbose: u8,
    /// Permit models reached over the network (remote:...); without it, and
    /// without SPLINTER_ALLOW_REMOTE=1, every model runs locally.
    #[arg(long, global = true)]
    pub allow_remote: bool,
    /// Limit the policy model's context to this many tokens (default: the
    /// largest its checkpoint supports); for a policy too large for the
    /// card at that size. Another model is limited by its reference,
    /// `local:<checkpoint>@<tokens>`.
    #[arg(long, global = true, value_name = "TOKENS", value_parser = clap::value_parser!(u32).range(1..))]
    pub policy_context_tokens: Option<u32>,
}

/// The commands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Learn from sources: capture them, generate tasks, solve, verify,
    /// critique and retry failures, build an sft-final dataset, train a
    /// candidate on it from the champion, and release it if it passes the
    /// release gate.
    Learn(Box<LearnArgs>),
    /// Learn a repository from its own tests: take a revision of it, make
    /// executable tasks of faults its tests catch, have the policy solve them
    /// as an agent in copies of the tree, keep what hidden checks confirm
    /// (reserving some files as an exam first, and refusing any attempt that
    /// saw the hidden tests), train a candidate from the champion and
    /// release it if it passes the release gate.
    Practice(Box<PracticeArgs>),
    /// Answer a question with a model, closed-book unless --open-book.
    Ask(AskArgs),
    /// Show the policy in use, recent runs and what the stores hold.
    Status,
    /// Check that a run can start here: the state root is writable with room
    /// to spare, the policy's weights fit on a GPU, and every role's model
    /// resolves to weights on disk. Loads no model and uses no network. Exits 1
    /// when a check fails.
    Doctor,
    /// Capture and inspect sources.
    #[command(subcommand)]
    Source(SourceCommand),
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
    /// Append verdicts to an experience set (re-grades without re-running).
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
    /// Grade a candidate or a model closed-book on a suite; or freeze and
    /// show the anchor suite.
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
    /// Show, maintain, verify, repair, archive and restore the state: the
    /// experience database and the files it tracks.
    #[command(subcommand)]
    State(StateCommand),
    /// Trace any artifact up to where it came from (down to source bytes)
    /// and down to everything that came from it.
    Lineage(LineageArgs),
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

fn strength(text: &str) -> Result<Strength, String> {
    parse_strength(text).map_err(|e| e.to_string())
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

/// `tasks ...`.
#[derive(Debug, Subcommand)]
pub enum TasksCommand {
    /// Generate tasks from stored sources into a task set.
    Generate {
        /// The sources, by id or unique prefix.
        #[arg(required = true, value_name = "SOURCE-ID")]
        sources: Vec<String>,
        /// Task kinds, comma-separated.
        #[arg(long, required = true, value_delimiter = ',', value_name = "K,..")]
        kinds: Vec<String>,
        /// The model that writes the tasks.
        #[arg(long, value_parser = model_ref, default_value_t = ModelRef::policy_default(), value_name = "REF")]
        generator: ModelRef,
        /// Who wrote the sources, when they are one person's: the generator is
        /// told, and a question may name the author as its subject.
        #[arg(long, value_name = "NAME")]
        author: Option<String>,
        /// The seed the `denoise` tasks are built with (drawn and printed
        /// when not given and the kinds include denoise).
        #[arg(long, value_name = "N")]
        denoise_seed: Option<Seed>,
    },
    /// Write differently worded questions about each task of a task set,
    /// into a task set of their own: the same facts, to be measured and
    /// never trained on.
    Variants {
        /// The task set, by id or unique prefix.
        #[arg(value_name = "TASKSET-ID")]
        task_set: String,
        /// The model that writes the variants (the one that wrote the
        /// tasks).
        #[arg(long, value_parser = model_ref, default_value_t = ModelRef::policy_default(), value_name = "REF")]
        generator: ModelRef,
        /// The most variants one task is given.
        #[arg(long, default_value_t = DEFAULT_VARIANTS_PER_TASK, value_name = "N")]
        per_task: usize,
    },
    /// Make canaries of the workspace tasks of a task set: each task with its
    /// hidden assertions also required to fail, so that no honest change
    /// passes it. They go into a task set of their own, which `release
    /// --canaries` hands the gate: a candidate that passes one is refused.
    Canaries {
        /// The task set, by id or unique prefix.
        #[arg(value_name = "TASKSET-ID")]
        task_set: String,
        /// The most canaries made.
        #[arg(long, value_name = "N", default_value_t = CANARY_COUNT,
            value_parser = positive_count)]
        count: usize,
    },
    /// List the stored task sets.
    List,
    /// Show a task set's tasks, or one task.
    Show {
        /// A task set's or a task's id, or a unique prefix of it.
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
}

/// `verify`.
#[derive(Debug, Args)]
pub struct VerifyArgs {
    /// The experience set, by id or unique prefix.
    #[arg(value_name = "EXPERIENCE-SET")]
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
}

/// `dataset ...`.
#[derive(Debug, Subcommand)]
pub enum DatasetCommand {
    /// Project experience sets through a view into a stored dataset.
    Build {
        /// The experience sets, by id or unique prefix.
        #[arg(required = true, value_name = "EXPERIENCE-SET")]
        sets: Vec<String>,
        /// sft-final, sft-step, critic, preference, verifier, decision,
        /// retrieval, outcome, denoise, cpt, voice or rehearsal.
        #[arg(long, required = true, value_parser = view, value_name = "VIEW")]
        view: ViewName,
        /// What the student sees: all (only the instruction), keep:K,..
        /// or mix:F.
        #[arg(long, value_parser = strip, value_name = "all|keep:K,..|mix:F")]
        strip: Option<StripPolicy>,
        /// The seed that picks which subjects keep their privileged context
        /// under `--strip mix:F` (drawn and printed when not given; a
        /// dataset built again with the same seed holds the same records).
        #[arg(long, value_name = "N", requires = "strip")]
        mix_seed: Option<Seed>,
        /// The weakest decision counted: executable, formal, consistency
        /// (default) or judged.
        #[arg(long, value_parser = strength, value_name = "STRENGTH")]
        min_strength: Option<Strength>,
        /// The system prompt every conversation opens with, in place of the
        /// default assistant's; the dataset records it.
        #[arg(long, value_name = "TEXT")]
        system_prompt: Option<String>,
        /// Write an objective brain cannot train in the export format.
        #[arg(long)]
        export_only: bool,
        /// Keep at most this many records, evenly spread over the
        /// projection (for voice, over the parts of the sources).
        #[arg(long, value_name = "N")]
        limit: Option<usize>,
        /// For the voice view: who the writer is. Requests are written as
        /// theirs and half the records open with their persona prompt, half
        /// with the line that says who the model is; refused with
        /// --system-prompt.
        #[arg(long, value_name = "NAME")]
        writer: Option<String>,
        /// For the voice view: keep at most this many tokens of the
        /// writer's text, evenly spread over all of it.
        #[arg(long, value_name = "N")]
        token_limit: Option<usize>,
        /// For the voice view: the most of the tokens one family may
        /// supply.
        #[arg(long, value_name = "SHARE")]
        max_family_share: Option<f64>,
        /// For the voice view: the model that writes, for each passage, a
        /// description of what it is about, which its request carries in
        /// place of the passage's opening (a passage with no admitted
        /// description keeps the opening).
        #[arg(long, value_parser = model_ref, value_name = "REF")]
        describe_with: Option<ModelRef>,
    },
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

/// `release`.
#[derive(Debug, Args)]
#[command(args_conflicts_with_subcommands = true)]
pub struct ReleaseArgs {
    /// The candidate, by id or unique prefix.
    #[arg(value_name = "CANDIDATE-ID")]
    pub candidate: Option<String>,
    /// The alias it replaces the champion of.
    #[arg(long, default_value = POLICY_DEFAULT, value_name = "NAME")]
    pub alias: String,
    /// The calibrated model that judges the tasks a judge decides.
    #[arg(long, value_parser = model_ref, value_name = "REF")]
    pub judge: Option<ModelRef>,
    /// Release for distribution and commercial use: refused unless the terms
    /// of everything the candidate was made from allow both. Without it the
    /// release is restricted: it records the terms and stays local.
    #[arg(long)]
    pub unrestricted: bool,
    /// The seed the gate's bootstrap intervals are resampled with (drawn and
    /// printed when not given; the same outcomes and seed give the same
    /// intervals).
    #[arg(long, value_name = "N")]
    pub interval_seed: Option<Seed>,
    /// A task set of canaries (`tasks canaries`) the candidate attempts beside
    /// the exam: a pass on any one refuses it, whatever else it passed.
    #[arg(long, value_name = "TASKSET-ID")]
    pub canaries: Option<String>,
    /// The gate's statistical settings.
    #[command(flatten)]
    pub gate: GateArgs,
    /// `list`.
    #[command(subcommand)]
    pub command: Option<ReleaseCommand>,
}

/// `release ...`.
#[derive(Debug, Subcommand)]
pub enum ReleaseCommand {
    /// List the releases and the aliases pointing at them.
    List,
}

/// `eval`.
#[derive(Debug, Args)]
#[command(args_conflicts_with_subcommands = true)]
pub struct EvalArgs {
    /// A candidate id (or prefix) or a model reference; none shows the
    /// anchor suite.
    #[arg(value_name = "REF")]
    pub model: Option<String>,
    /// held-out, retention, anchor, or an anchor-format file of tasks.
    #[arg(long, default_value = "held-out", value_parser = suite, value_name = "SUITE")]
    pub suite: SuiteChoice,
    /// With --suite anchor: make these files' tasks (repeat the flag for
    /// several) the anchor suite's next version first. JSON Lines:
    /// {"instruction", "reference", "kind"?}.
    #[arg(long, value_name = "FILE")]
    pub freeze: Vec<PathBuf>,
    /// The calibrated model that judges the tasks a judge decides.
    #[arg(long, value_parser = model_ref, value_name = "REF")]
    pub judge: Option<ModelRef>,
    /// `matrix`.
    #[command(subcommand)]
    pub command: Option<EvalCommand>,
}

/// `eval ...`.
#[derive(Debug, Subcommand)]
pub enum EvalCommand {
    /// The continual-learning results matrix of a lineage: what each gate
    /// measured of each release on each earlier release's suite, with the
    /// metrics read from it. A cell no gate measured is shown absent.
    Matrix {
        /// The alias whose releases are the generations.
        #[arg(long, default_value = POLICY_DEFAULT, value_name = "NAME")]
        alias: String,
    },
}

/// `lineage`.
#[derive(Debug, Args)]
pub struct LineageArgs {
    /// Any artifact's id - a source, task, experience, dataset, candidate,
    /// release, answer, ... - or a unique prefix of it.
    #[arg(value_name = "ID")]
    pub id: String,
    /// Only where it came from.
    #[arg(long, conflicts_with_all = ["down", "both"])]
    pub up: bool,
    /// Only what came from it.
    #[arg(long, conflicts_with = "both")]
    pub down: bool,
    /// Both ways (the default).
    #[arg(long)]
    pub both: bool,
    /// The most edges walked from it (default: all the way).
    #[arg(long, value_name = "N")]
    pub depth: Option<usize>,
}

impl LineageArgs {
    /// The direction the flags ask for.
    pub fn direction(&self) -> Direction {
        match (self.up, self.down) {
            (true, _) => Direction::Up,
            (_, true) => Direction::Down,
            _ => Direction::Both,
        }
    }
}

mod absorb;
mod consolidate;
mod exam;
mod gate;
mod learn;
mod practice;
mod state;
mod train;
pub use absorb::{AbsorbArgs, ClaimsCommand, SessionCommand};
pub use consolidate::ConsolidateArgs;
pub use exam::{ExamArgs, ExamReportCommand, ExamSetArgs, ExamSetCommand, PowerArgs, SelectArgs};
pub use gate::GateArgs;
pub use learn::LearnArgs;
pub use practice::PracticeArgs;
pub use state::{RunsCommand, StateCommand};
use train::positive_count;
pub use train::{MonitoringArgs, OptimiserArgs, RehearseArgs, TrainArgs};

#[cfg(test)]
mod tests;
