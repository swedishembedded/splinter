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
use splinter_sdk::exam_set::{DEFAULT_DEV_TASKS_PER_FAMILY, DEFAULT_TASKS_PER_FAMILY};
use splinter_sdk::learn::{parse_budget, DEFAULT_REHEARSAL_SHARE, DEFAULT_VOICE_SHARE};
use splinter_sdk::lineage::Direction;
use splinter_sdk::powered::DEFAULT_RESAMPLES;
use splinter_sdk::reserve::{DEFAULT_DEV_FAMILIES, DEFAULT_EXAM_FAMILIES};
use splinter_sdk::train::{
    DEFAULT_LORA_RANK, DEFAULT_MONITOR_SHARE, DEFAULT_PATIENCE, EVALUATIONS_PER_BUDGET,
    MAX_MONITOR_SHARE, MAX_PASSES,
};
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

fn strip(text: &str) -> Result<Strip, String> {
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

fn monitor_share(text: &str) -> Result<f64, String> {
    let share: f64 = text.parse().map_err(|e| format!("{text:?}: {e}"))?;
    if share > 0.0 && share <= MAX_MONITOR_SHARE {
        Ok(share)
    } else {
        Err(format!(
            "{text:?} is not a share in (0, {MAX_MONITOR_SHARE}]"
        ))
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
    /// The model that judges answers to tasks only a judge can pass (a
    /// conversation: a check that nothing was invented does not make an
    /// answer right). It is another model than the teacher, the generator and
    /// the policy, and is measured before its verdicts count.
    #[arg(long, value_parser = model_ref, value_name = "REF")]
    pub judge: Option<ModelRef>,
    /// The model that surveys the sources and plans the learning when no
    /// --kinds are named (default: the generator).
    #[arg(long, value_parser = model_ref, value_name = "REF")]
    pub planner: Option<ModelRef>,
    /// Give this share of the training records retrieved passages of the
    /// sources in their prompt, four to a record and the passage the task was
    /// written from among them for four in five, so the policy learns to use
    /// context where it holds the answer and to answer without it where it
    /// does not. Needs the embedding model.
    #[arg(long, value_name = "SHARE", value_parser = share)]
    pub with_passages: Option<f64>,
    /// Who the policy is to become, when it is to think like a person: its
    /// training records open with a system prompt saying so, and it is asked
    /// under that prompt afterwards (default: the persona the plan finds in
    /// the goal, if any).
    #[arg(long, value_name = "NAME")]
    pub persona: Option<String>,
    /// The share of the training examples that is the writer's own text,
    /// in [0, 1): records built by code from the sources, a stretch of the
    /// writer's words word for word as the answer, trained beside the
    /// dialogues and held out with the families they print. 0 turns it off.
    #[arg(long, value_name = "SHARE", value_parser = voice_share, help = format!(
        "The share of the training examples that is the writer's own text, in [0, 1): records \
         built by code from the sources, no model involved, trained beside the dialogues and held \
         out with the families they print; 0 turns it off [default for a run with a persona: \
         {DEFAULT_VOICE_SHARE}]"
    ))]
    pub voice: Option<f64>,
    /// The share of the training draws that are the base model's own
    /// answers to general tasks, in [0, 1): a rehearsal set built by the
    /// `rehearse` stage - sums and format requests built by code, and
    /// requests the base writes itself, never an anchor task - answered by
    /// the base with no adapter under the default prompt, mixed into
    /// training at this share and into the monitoring set, so the run does
    /// not move the model off what the base does. 0 turns it off.
    #[arg(long, value_name = "SHARE", value_parser = voice_share, help = format!(
        "The share of the training draws that are the base model's own answers to general tasks, \
         in [0, 1): built by the rehearse stage, never an anchor task, mixed into training and \
         the monitoring set; 0 turns it off [default for a run with a persona: \
         {DEFAULT_REHEARSAL_SHARE}]"
    ))]
    pub rehearsal: Option<f64>,
    /// How many families of the sources to reserve for the exam before
    /// anything is generated or built: their text, in every edition, is in no
    /// source the run reads after that. 0 reserves none.
    #[arg(long, value_name = "N", help = format!(
        "How many families of the sources to reserve for the exam before anything is generated \
         or built: their text, in every edition, is in no source the run reads after that; the \
         run is refused when the sources cannot spare them; 0 reserves none [default for a run \
         with a persona: {DEFAULT_EXAM_FAMILIES}]"
    ))]
    pub exam_families: Option<usize>,
    /// Tasks of each family of the final test.
    #[arg(long, value_name = "N", help = format!(
        "Tasks of each family of the reserved final test [default: {DEFAULT_TASKS_PER_FAMILY}]"))]
    pub exam_tasks_per_family: Option<usize>,
    /// How many more families to reserve for the dev suite.
    #[arg(long, value_name = "N", help = format!(
        "How many more families to reserve for the dev suite a checkpoint is chosen on, as far \
         from training as the final test's and apart from it; 0 reserves none [default with a \
         final test: {DEFAULT_DEV_FAMILIES}]"))]
    pub dev_families: Option<usize>,
    /// Tasks of each dev family.
    #[arg(long, value_name = "N", help = format!(
        "Tasks of each family of the dev suite [default: {DEFAULT_DEV_TASKS_PER_FAMILY}]"))]
    pub dev_tasks_per_family: Option<usize>,
    /// Answers per task per arm in the exam.
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..),
        help = format!("Answers per task per arm in the reserved exam: the greedy one and the \
                        rest sampled [default: {DEFAULT_RESAMPLES}]"))]
    pub exam_resamples: Option<u32>,
    /// The step budget of the training: the most steps it may take.
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..),
        help = format!("The step budget of the training: the most steps it may take [default: \
                        {MAX_PASSES} passes over the examples, within bounds]"))]
    pub steps: Option<u32>,
    /// LoRA rank of the adapter.
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..),
        help = format!("LoRA rank of the adapter [default: {DEFAULT_LORA_RANK}]"))]
    pub rank: Option<u32>,
    /// The learning rate, alpha and weight decay of the training.
    #[command(flatten)]
    pub optimiser: OptimiserArgs,
    /// Records averaged into one optimizer step (default: from the size of the
    /// dataset, when the steps are not named).
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..))]
    pub records_per_step: Option<u32>,
    /// The seed of the adapter's initialisation and the batch order (brain's
    /// default if not given); two runs that differ only here show how much
    /// of a measured difference is the draw.
    #[arg(long, value_name = "N")]
    pub seed: Option<u64>,
    /// How the training is watched.
    #[command(flatten)]
    pub monitoring: MonitoringArgs,
    /// Hold the frozen base at bf16, half the bytes of fp32: what a 7B
    /// base needs to train on one 24 GiB card.
    #[arg(long)]
    pub bf16_base: bool,
    /// How the frontier is measured.
    #[command(flatten)]
    pub pass_at_k: PassAtKArgs,
}

/// How a supervised training run is watched as it trains, for the commands
/// that train: a share of its training families is set aside and scored
/// every few steps, the adapter of the evaluation with the lowest loss on
/// them is the candidate's, and the run stops once that loss has gone a
/// patience of evaluations without improving.
#[derive(Debug, Default, Args)]
pub struct MonitoringArgs {
    /// Steps between evaluations of the monitoring records; 0 monitors
    /// nothing and the candidate carries its last step.
    #[arg(long, value_name = "N", help = format!(
        "Steps between evaluations of the monitoring records; 0 monitors nothing and the \
         candidate carries its last step [default: {EVALUATIONS_PER_BUDGET} evaluations over the \
         step budget]"
    ))]
    pub eval_every: Option<u32>,
    /// Evaluations without improvement before the training stops; 0 runs
    /// the whole budget (the best evaluation is carried either way).
    #[arg(long, value_name = "N", help = format!(
        "Evaluations without improvement before the training stops; 0 runs the whole budget \
         (the best evaluation is carried either way) [default: {DEFAULT_PATIENCE}]"
    ))]
    pub patience: Option<u32>,
    /// The share of the training families set aside as the monitoring
    /// records, in (0, 1/2]: whole families, never the held-out ones the
    /// gate and the exam decide on.
    #[arg(long, value_name = "SHARE", value_parser = monitor_share, help = format!(
        "The share of the training families set aside as the monitoring records, in (0, \
         {MAX_MONITOR_SHARE}]: whole families, never the held-out ones the gate and the exam \
         decide on [default: {DEFAULT_MONITOR_SHARE}]"
    ))]
    pub monitor_share: Option<f64>,
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
        strip: Option<Strip>,
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
    /// With --suite anchor: make these files' tasks (repeat the flag for
    /// several) the anchor suite's next version first. JSON Lines:
    /// {"instruction", "reference", "kind"?}.
    #[arg(long, value_name = "FILE")]
    pub freeze: Vec<PathBuf>,
    /// The calibrated model that judges the tasks a judge decides.
    #[arg(long, value_parser = model_ref, value_name = "REF")]
    pub judge: Option<ModelRef>,
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

mod exam;
mod state;
mod train;
pub use exam::{ExamArgs, ExamReportCommand, ExamSetArgs, ExamSetCommand, PowerArgs};
pub use state::{RunsCommand, StateCommand};
pub use train::{OptimiserArgs, RehearseArgs, TrainArgs};

#[cfg(test)]
mod tests;
