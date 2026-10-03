// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The command line's grammar: one verb per pipeline stage, the thing acted
//! on as a positional argument, one model-reference syntax, and four global
//! flags every command takes. Defaults come from the crates that own them.

use std::path::PathBuf;
use std::time::Duration;

use clap::{ArgAction, ArgGroup, Args, Parser, Subcommand};
use splinter_sdk::critique::DEFAULT_RETRIES;
use splinter_sdk::curriculum::frontier::{PassAtK, DEFAULT_K, DEFAULT_SAMPLING};
use splinter_sdk::datasets::{parse_strength, parse_strip, Strip, ViewName};
use splinter_sdk::eval::SuiteChoice;
use splinter_sdk::learn::parse_budget;
use splinter_sdk::lineage::Direction;
use splinter_sdk::train::{
    DEFAULT_DPO_BETA, DEFAULT_LORA_RANK, DEFAULT_REPLAY_FRACTION, DEFAULT_STEPS,
};
use splinter_sdk::variants::DEFAULT_VARIANTS_PER_TASK;
use splinter_sdk::vocabulary::annotation::Strength;
use splinter_sdk::vocabulary::model_ref::{ModelRef, POLICY_DEFAULT};

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
                  rerun or inspected alone."
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
    /// The state root (default: $SPLINTER_STATE, else ~/.sven/splinter).
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
}

/// The commands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Learn from sources: capture them, generate tasks, solve, verify,
    /// critique and retry failures, build an sft-final dataset, train a
    /// candidate on it from the champion, and release it if it passes the
    /// release gate.
    Learn(LearnArgs),
    /// Answer a question with a model, closed-book unless --open-book.
    Ask(AskArgs),
    /// Show the policy in use, recent runs and what the stores hold.
    Status,
    /// Capture and inspect sources.
    #[command(subcommand)]
    Source(SourceCommand),
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

fn strip(text: &str) -> Result<Strip, String> {
    parse_strip(text).map_err(|e| e.to_string())
}

fn strength(text: &str) -> Result<Strength, String> {
    parse_strength(text).map_err(|e| e.to_string())
}

fn suite(text: &str) -> Result<SuiteChoice, String> {
    text.parse().map_err(|e: splinter_sdk::Error| e.to_string())
}

/// `learn`.
#[derive(Debug, Args)]
pub struct LearnArgs {
    /// Files, directories, cmd:<command> (quoted as one argument), or stored
    /// source ids.
    #[arg(required = true, value_name = "SOURCE")]
    pub sources: Vec<String>,
    /// What the learner is after; steers the tasks generated.
    #[arg(long, value_name = "TEXT")]
    pub goal: Option<String>,
    /// Task kinds, comma-separated; when not named, a planner model surveys
    /// the sources and chooses them.
    #[arg(long, value_delimiter = ',', value_name = "K,..")]
    pub kinds: Vec<String>,
    /// Wall-clock time the whole run may take, in h, m and s: 30m, 2h, 1h30m.
    #[arg(long, value_parser = budget, value_name = "DUR")]
    pub budget: Option<Duration>,
    /// Report the plan and write nothing.
    #[arg(long)]
    pub dry_run: bool,
    /// Stop at the trained candidate: do not run the release gate.
    #[arg(long)]
    pub no_release: bool,
    /// Solve each task once and train on every task, instead of measuring
    /// pass@k and keeping the tasks worth training on (failed at least
    /// sometimes, with a verified answer).
    #[arg(long, conflicts_with_all = ["k", "temperature", "top_k"])]
    pub no_frontier: bool,
    /// Skip the policy's own attempts: the teacher answers every task
    /// open-book and the policy is trained on its verified answers. For a
    /// policy that cannot answer the tasks closed-book at all.
    #[arg(long, conflicts_with_all = ["no_frontier", "k", "temperature", "top_k"])]
    pub distill: bool,
    /// The model that solves open-book, shown each task's grounding
    /// material, the tasks the policy never solves (default: the policy).
    #[arg(long, value_parser = model_ref, value_name = "REF")]
    pub teacher: Option<ModelRef>,
    /// The model that writes the tasks (default: the policy).
    #[arg(long, value_parser = model_ref, value_name = "REF")]
    pub generator: Option<ModelRef>,
    /// The model that surveys the sources and plans the learning when no
    /// --kinds are named (default: the generator).
    #[arg(long, value_parser = model_ref, value_name = "REF")]
    pub planner: Option<ModelRef>,
    /// Optimizer steps of the training.
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..),
        help = format!("Optimizer steps of the training [default: {DEFAULT_STEPS}]"))]
    pub steps: Option<u32>,
    /// LoRA rank of the adapter.
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..),
        help = format!("LoRA rank of the adapter [default: {DEFAULT_LORA_RANK}]"))]
    pub rank: Option<u32>,
    /// The peak learning rate of the training (brain's default if not given).
    #[arg(long, value_name = "LR")]
    pub lr: Option<f32>,
    /// Hold the frozen base at bf16, half the bytes of fp32: what a 7B
    /// base needs to train on one 24 GiB card.
    #[arg(long)]
    pub bf16_base: bool,
    /// How the frontier is measured.
    #[command(flatten)]
    pub pass_at_k: PassAtKArgs,
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

/// `ask`.
#[derive(Debug, Args)]
pub struct AskArgs {
    /// The question.
    pub question: String,
    /// Show the model this stored source's text with the question.
    #[arg(long, value_name = "SOURCE-ID")]
    pub open_book: Option<String>,
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
        /// retrieval, outcome, denoise or cpt.
        #[arg(long, required = true, value_parser = view, value_name = "VIEW")]
        view: ViewName,
        /// What the student sees: all (only the instruction), keep:K,..
        /// or mix:F.
        #[arg(long, value_parser = strip, value_name = "all|keep:K,..|mix:F")]
        strip: Option<Strip>,
        /// The weakest decision counted: executable, formal, consistency
        /// (default) or judged.
        #[arg(long, value_parser = strength, value_name = "STRENGTH")]
        min_strength: Option<Strength>,
        /// Write an objective brain cannot train in the export format.
        #[arg(long)]
        export_only: bool,
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

/// `train`: the datasets' objective decides how - chat datasets by
/// supervised fine-tuning, preference datasets by DPO.
#[derive(Debug, Args)]
pub struct TrainArgs {
    /// The datasets, by id or unique prefix, all chat or all preference
    /// pairs; the newest records of the last are held out for scoring.
    #[arg(required = true, value_name = "DATASET-ID")]
    pub datasets: Vec<String>,
    /// The base to train, and an adapter on it to continue.
    #[arg(long, value_parser = model_ref, default_value_t = ModelRef::policy_default(), value_name = "REF")]
    pub from: ModelRef,
    /// The fraction of each earlier release's training records replayed,
    /// when training from a policy alias.
    #[arg(long, default_value_t = DEFAULT_REPLAY_FRACTION, value_name = "F")]
    pub replay_fraction: f64,
    /// Optimizer steps.
    #[arg(long, default_value_t = DEFAULT_STEPS, value_name = "N")]
    pub steps: u32,
    /// LoRA rank of a new adapter.
    #[arg(long, default_value_t = DEFAULT_LORA_RANK, value_name = "R")]
    pub rank: u32,
    /// The DPO temperature, for preference datasets only.
    #[arg(long, value_name = "BETA", help = beta_help())]
    pub beta: Option<f32>,
}

/// `--beta`'s help, naming the default it falls back to.
fn beta_help() -> String {
    format!("The DPO temperature, for preference datasets only [default: {DEFAULT_DPO_BETA}]")
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
pub struct EvalArgs {
    /// A candidate id (or prefix) or a model reference; none shows the
    /// anchor suite.
    #[arg(value_name = "REF")]
    pub model: Option<String>,
    /// held-out, retention, anchor, or an anchor-format file of tasks.
    #[arg(long, default_value = "held-out", value_parser = suite, value_name = "SUITE")]
    pub suite: SuiteChoice,
    /// With --suite anchor: make this file's tasks the anchor suite's next
    /// version first. JSON Lines: {"instruction", "reference", "kind"?}.
    #[arg(long, value_name = "FILE")]
    pub freeze: Option<PathBuf>,
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

/// `state ...`.
#[derive(Debug, Subcommand)]
pub enum StateCommand {
    /// What the experience database holds, as files.
    Status,
    /// Merge small files, index what is not indexed and retire finished
    /// writers; nothing stored changes.
    Maintain {
        /// Also delete files nothing reaches that are past their grace
        /// period; a snapshot a dataset pinned is never touched.
        #[arg(long)]
        collect: bool,
    },
    /// Let go of a snapshot a dataset (or another holder) keeps alive, named
    /// as `state status` lists it, so its files can be collected.
    Unpin {
        /// The holder's name.
        holder: String,
    },
    /// Check the database and every file it tracks, and report each one that
    /// is missing or damaged; exits 1 when anything is.
    Verify {
        /// Read every byte instead of only checking sizes.
        #[arg(long)]
        deep: bool,
    },
    /// Recover what verify finds: fill holes from copies (archives or other
    /// state roots), rebuild indexes, and with --accept-loss write off what
    /// no copy has.
    Repair {
        /// A copy to take missing files from: an archive file, or another
        /// state root. May be given more than once.
        #[arg(long = "from", value_name = "PATH")]
        from: Vec<PathBuf>,
        /// Give up on what no copy has: withdraw damaged database files and
        /// record lost artifacts in the ledger, which `state status` lists.
        #[arg(long)]
        accept_loss: bool,
    },
    /// Pack the database and the files it tracks into one archive; the same
    /// state always gives the same bytes.
    Archive {
        /// Where to write the archive (a .tar.zst).
        file: PathBuf,
        /// Leave the artifacts (adapters, datasets) out.
        #[arg(long)]
        no_artifacts: bool,
        /// An earlier archive: carry only what it lacks. Restore the result
        /// together with that archive.
        #[arg(long, value_name = "ARCHIVE")]
        since: Option<PathBuf>,
    },
    /// Unpack an archive (and the archives an incremental one was made
    /// after) into an empty state root, verifying everything first.
    Restore {
        /// The archive to restore, then any it builds on.
        #[arg(required = true)]
        files: Vec<PathBuf>,
    },
}

/// `runs ...`.
#[derive(Debug, Subcommand)]
pub enum RunsCommand {
    /// List the recorded runs.
    List,
    /// Show one run's record: arguments, stages, status, outputs.
    Show {
        /// The run's id.
        id: String,
    },
    /// Ask a run in progress to stop.
    Cancel {
        /// The run's id.
        id: String,
    },
}

#[cfg(test)]
mod tests;
