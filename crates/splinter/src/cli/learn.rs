// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The grammar of `learn`.

use std::path::PathBuf;
use std::time::Duration;

use clap::Args;
use splinter_sdk::acquire::{
    DEFAULT_MASTERY, DEFAULT_MAX_ROUNDS, DEFAULT_MAX_ROUNDS_PER_SKILL, DEFAULT_MIN_PROBES,
    DEFAULT_RETENTION_DROP,
};
use splinter_sdk::exam_set::{DEFAULT_DEV_TASKS_PER_FAMILY, DEFAULT_TASKS_PER_FAMILY};
use splinter_sdk::learn::{DEFAULT_REHEARSAL_SHARE, DEFAULT_VOICE_SHARE};
use splinter_sdk::powered::DEFAULT_RESAMPLES;
use splinter_sdk::reserve::{DEFAULT_DEV_FAMILIES, DEFAULT_EXAM_FAMILIES};
use splinter_sdk::seed::Seed;
use splinter_sdk::train::{DEFAULT_LORA_RANK, DEFAULT_REPLAY_FRACTION, MAX_PASSES};
use splinter_sdk::vocabulary::model_ref::ModelRef;

use super::{
    budget, model_ref, share, voice_share, GateArgs, MonitoringArgs, OptimiserArgs, PassAtKArgs,
};

/// The flags of an acquisition (`learn --source`).
#[derive(Debug, Args)]
pub struct AcquireArgs {
    /// The pass rate on a skill's probes that counts as mastered.
    #[arg(long, value_name = "SHARE", value_parser = share, requires = "source",
        help = format!("The pass rate on a skill's probes that counts as mastered \
        [default: {DEFAULT_MASTERY}]"))]
    pub mastery: Option<f64>,
    /// Probes per skill in each half (dev and final).
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(usize),
        requires = "source")]
    pub probes: Option<usize>,
    /// The graded probes a skill's competence must rest on.
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(usize),
        requires = "source", help = format!("The graded probes a skill's competence must rest on \
        [default: {DEFAULT_MIN_PROBES}]"))]
    pub min_probes: Option<usize>,
    /// Training rounds in all.
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(usize),
        requires = "source", help = format!("Training rounds in all [default: {DEFAULT_MAX_ROUNDS}]"))]
    pub max_rounds: Option<usize>,
    /// Training rounds one skill may take.
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(usize),
        requires = "source", help = format!("Training rounds one skill may take \
        [default: {DEFAULT_MAX_ROUNDS_PER_SKILL}]"))]
    pub max_rounds_per_skill: Option<usize>,
    /// How far a skill measured earlier may fall in a round.
    #[arg(long, value_name = "SHARE", value_parser = voice_share, requires = "source",
        help = format!("How far a skill measured earlier may fall in a round before its \
        candidate is rejected [default: {DEFAULT_RETENTION_DROP}]"))]
    pub retention_drop: Option<f64>,
    /// Train every weight in each round instead of a LoRA adapter.
    #[arg(long, requires = "source")]
    pub full_fine_tune: bool,
    /// brain's training precision of a full fine-tune.
    #[arg(
        long,
        value_name = "SPEC",
        requires = "full_fine_tune",
        help = "brain's training precision of a full fine-tune, e.g. state=int8,sr=on \
        [default: brain's: fp32 everywhere]"
    )]
    pub train_precision: Option<String>,
    /// Keep the checkpoints of rejected and superseded full-fine-tune rounds.
    #[arg(long, requires = "full_fine_tune")]
    pub keep_candidates: bool,
    /// The fraction of each accepted round's records replayed into the next
    /// full-fine-tune round.
    #[arg(long, value_name = "F", value_parser = share, requires = "full_fine_tune",
        help = format!("The fraction of each accepted round's trained-on records replayed \
        into the next full-fine-tune round [default: {DEFAULT_REPLAY_FRACTION}]"))]
    pub replay_fraction: Option<f64>,
    /// Let the incumbent write the next round's training tasks.
    #[arg(long, requires = "source")]
    pub self_generate: bool,
    /// Read the skill graph from the sources' structure, one skill per
    /// section, instead of asking a planner model.
    #[arg(long, requires = "source")]
    pub structural_graph: bool,
    /// Teach a task the policy never solves with the task's own reference
    /// answer (the output the runtime computed, a reference program),
    /// graded like any answer, instead of asking the teacher model.
    #[arg(long, requires = "source", conflicts_with = "teacher")]
    pub reference_teacher: bool,
    /// Take the graph and probes of an earlier acquisition run.
    #[arg(
        long,
        value_name = "RUN-ID",
        requires = "source",
        help = "Take the skill graph and probes of this earlier learn --source run (its id is in \
        `splinter runs list`) instead of planning and generating them; refused unless that run \
        used the same sources. The baseline, rounds and assessment are still this run's"
    )]
    pub probes_from: Option<String>,
}

/// The flywheel harnesses, run as `sven harness run` against a model server
/// that is already running.
#[derive(Debug, Args)]
pub struct TaskHarnessArgs {
    /// The invent harness (a directory): agents read each window of the
    /// sources and write task directories - instruction, environment,
    /// reference solution, hidden tests, the task's own solving process -
    /// in place of one call to the generator model. Each admitted task is a
    /// workspace task graded by its own tests, and is solved by its own
    /// process.
    #[arg(
        long = "task-harness",
        value_name = "PATH",
        requires_all = ["sven", "solve_harness", "task_harness_endpoint", "task_harness_model"]
    )]
    pub invent: Option<PathBuf>,
    /// The generic solve harness (a directory holding `harness.yaml`): it
    /// solves a task that brings no process of its own and is the template
    /// the invent harness adapts a task's process from.
    #[arg(long, value_name = "PATH", requires = "invent")]
    pub solve_harness: Option<PathBuf>,
    /// The validate harness (a directory): each passing attempt is judged
    /// under every focus (outcome, process, teaching) and kept only when
    /// every decision is keep. Without it passing attempts are not validated.
    #[arg(long, value_name = "PATH", requires = "invent")]
    pub validate_harness: Option<PathBuf>,
    /// The `sven` executable that runs the harnesses.
    #[arg(long, value_name = "PATH", requires = "invent")]
    pub sven: Option<PathBuf>,
    /// Base URL of the already running OpenAI-compatible server the
    /// harnesses' agents use.
    #[arg(
        long = "task-harness-endpoint",
        value_name = "URL",
        requires = "invent"
    )]
    pub endpoint: Option<String>,
    /// The model the harnesses' agents ask the server for, as
    /// `provider/name`; a brain server is addressed as `brain/<name>`.
    #[arg(long = "task-harness-model", value_name = "SPEC", requires = "invent")]
    pub model: Option<String>,
    /// The file holding the server's API key: a JSON object of provider
    /// name to key (`{"brain": "..."}`), or the key on one line.
    #[arg(
        long = "task-harness-key-file",
        value_name = "FILE",
        requires = "invent"
    )]
    pub key_file: Option<PathBuf>,
    /// The most angles, and so new tasks, one invent run works on.
    #[arg(
        long = "task-harness-angles",
        value_name = "N",
        default_value_t = 3,
        requires = "invent"
    )]
    pub angles: usize,
    /// How long one harness run may take, in seconds. A run of the invent
    /// harness takes tens of minutes.
    #[arg(
        long = "task-harness-timeout",
        value_name = "SECS",
        default_value_t = 7200,
        requires = "invent"
    )]
    pub timeout_secs: u64,
}

/// `learn`.
#[derive(Debug, Args)]
pub struct LearnArgs {
    /// Files, directories, cmd:<command> (quoted as one argument), or stored
    /// source ids; with --source, the words of the objective instead.
    #[arg(required_unless_present = "source", value_name = "SOURCE")]
    pub sources: Vec<String>,
    /// Learn an objective from documentation: `learn "Become an exceptional
    /// Lua 5.1 programmer" --source https://www.lua.org/manual/5.1/manual.html`.
    /// The positional words are the objective, and each --source (a URL, a
    /// file or a directory; repeat for more) is read into a skill graph with
    /// probes, a measured baseline, training rounds on the skill expected to
    /// benefit most and an assessment on probes that decided nothing.
    #[arg(
        long = "source",
        value_name = "URL|PATH",
        conflicts_with_all = ["goal", "dry_run", "distill", "no_frontier", "adapter", "base", "persona"]
    )]
    pub source: Vec<String>,
    /// How an acquisition (--source) is measured and stopped.
    #[command(flatten)]
    pub acquire: AcquireArgs,
    /// What the learner is after; steers the tasks generated.
    #[arg(long, value_name = "TEXT")]
    pub goal: Option<String>,
    /// The flywheel harnesses that invent the tasks.
    #[command(flatten)]
    pub task_harness: TaskHarnessArgs,
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
    /// Make this share of all the dialogue records abstentions: the writer,
    /// shown passages that do not hold the answer or asked what lies beyond
    /// the writings, says so in their own voice (two thirds retrieval misses,
    /// a third questions after their time). Taken from the records given
    /// passages, so it is at most --with-passages.
    #[arg(long, value_name = "SHARE", value_parser = share, requires = "with_passages")]
    pub abstain: Option<f64>,
    /// The checkpoint to train: a name in the model store or a path. Used
    /// with --adapter; the configured base when not given.
    #[arg(long, value_name = "CHECKPOINT", requires = "adapter")]
    pub base: Option<String>,
    /// The name to keep the trained adapter under in the model store, where
    /// `local:<base>+<name>` finds it. A name that exists is continued, not
    /// replaced. The run publishes the adapter instead of putting it through
    /// the release gate.
    #[arg(long, value_name = "NAME")]
    pub adapter: Option<String>,
    /// Who the policy is to become, when it is to think like a person: its
    /// training records open with a system prompt saying so, and it is asked
    /// under that prompt afterwards (default: the persona the plan finds in
    /// the goal, if any).
    #[arg(long, value_name = "NAME")]
    pub persona: Option<String>,
    /// The share of the training tokens that is the writer's own text,
    /// in [0, 1): records built by code from the sources, a chunk of the
    /// writer's words word for word as the answer, trained beside the
    /// dialogues and held out with the families they print. 0 turns it off.
    #[arg(long, value_name = "SHARE", value_parser = voice_share, help = format!(
        "The share of the training tokens that is the writer's own text, in [0, 1): records \
         built by code from the sources, no model involved, trained beside the dialogues and held \
         out with the families they print; 0 turns it off [default for a run with a persona: \
         {DEFAULT_VOICE_SHARE}]"
    ))]
    pub voice: Option<f64>,
    /// Ask the writer's passages by what the generator says they are about
    /// instead of by their heading and opening.
    #[arg(long)]
    pub describe_voice: bool,
    /// Keep the adapter of every evaluation of the training and choose the
    /// step by what each writes on the reserved dev suite - right answers,
    /// no invented specifics, then voice - instead of by the monitoring loss.
    #[arg(long)]
    pub select_on_dev: bool,
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
    /// The seed the rehearsal's form tasks are built with.
    #[arg(
        long,
        value_name = "N",
        help = "The seed the rehearsal's form tasks are built with \
        [default: drawn and printed]"
    )]
    pub rehearsal_seed: Option<Seed>,
    /// The seed the `denoise` tasks are built with.
    #[arg(
        long,
        value_name = "N",
        help = "The seed the denoise tasks are built with, when the \
        kinds include them [default: drawn and printed]"
    )]
    pub denoise_seed: Option<Seed>,
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
    /// The seed the exam families are reserved with.
    #[arg(
        long,
        value_name = "N",
        help = "The seed the exam's families are reserved with: the \
        same seed over the same sources reserves the same families, and exams are comparable \
        across runs only when they share one [default: drawn and printed]"
    )]
    pub exam_seed: Option<Seed>,
    /// The seed the exam's bootstrap intervals are resampled with.
    #[arg(
        long,
        value_name = "N",
        help = "The seed the exam's bootstrap intervals are resampled \
        with [default: drawn and printed]"
    )]
    pub interval_seed: Option<Seed>,
    /// The release gate's statistical settings.
    #[command(flatten)]
    pub gate: GateArgs,
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
    /// The seed of the adapter's initialisation and the batch order (drawn,
    /// printed and recorded if not given); two runs that differ only here
    /// show how much of a measured difference is the draw.
    #[arg(long, value_name = "N")]
    pub seed: Option<u64>,
    /// How the training is watched.
    #[command(flatten)]
    pub monitoring: MonitoringArgs,
    /// How the frontier is measured.
    #[command(flatten)]
    pub pass_at_k: PassAtKArgs,
}
