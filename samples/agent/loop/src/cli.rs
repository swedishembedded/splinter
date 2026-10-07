// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements self-improving coding agents whose every
// step is auditable, for its clients. If your team needs expertise in agent
// evaluation or locally operated coding agents, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The command line: run a task, look at a run, cancel it, resume it, apply
//! its result.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use splinter_sdk::config::Config;
use splinter_sdk::store::StateRoot;
use splinter_sdk::vocabulary::digest::Digest;
use splinter_sdk::{runs, Splinter};

use crate::acceptance::Check;
use crate::contract::{defaults, Contract, Limits, CONTRACT_SCHEMA, DEFAULT_MODEL};
use crate::outcome::{Outcome, Status};
use crate::repo::{git, head_rev, is_clean};
use crate::run::{execute, system_prompt, Request};
use crate::store::{read_json, LoopHome, CHECKPOINT_FILE, CONTRACT_FILE, OUTCOME_FILE, PATCH_FILE};

/// The loop.
#[derive(Parser, Debug)]
#[command(
    name = "agent-loop",
    version,
    about = "A supervised coding-agent loop on one repository"
)]
pub struct Cli {
    /// Home directory whose `.sven/` holds the loop's and Splinter's state
    /// (default: `$HOME`).
    #[arg(long, global = true)]
    home: Option<PathBuf>,
    /// Brain's model store, where `local:<org>/<model>` names a checkpoint
    /// (default: `$BRAIN_MODELS_DIR`, else `<home>/.local/share/brain/models`).
    #[arg(long, global = true)]
    models_dir: Option<PathBuf>,
    /// Let a model served by brain reason before it answers (default: it is
    /// asked for no reasoning block). Written in the run's `model_selected`
    /// event.
    #[arg(long, global = true)]
    thinking: bool,
    /// Sampling temperature (0 to 2) asked of a model reached through sven's
    /// providers, a served model included (default: the provider's, which for
    /// a served model can be greedy decoding and can loop). Written in the
    /// run's `model_selected` event.
    #[arg(long, global = true)]
    temperature: Option<f32>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Work a task in a repository until a candidate is accepted or a limit is reached.
    Run(Box<RunArgs>),
    /// Show a run: its state, outcome and where its evidence is.
    Status { run: String },
    /// Ask a running run (in any process) to stop.
    Cancel { run: String },
    /// Continue a run from its last checkpoint.
    Resume { run: String },
    /// Apply an accepted run's patch to the repository it was started on.
    Apply { run: String },
    /// List runs.
    Runs,
    /// Train a candidate adapter on a dataset (it is not used until judged).
    Train(TrainArgs),
    /// Model versions: list, judge a candidate, roll back.
    Models {
        #[command(subcommand)]
        command: ModelsCommand,
    },
    /// Write training records (brain's chat format) from accepted, unaided
    /// runs on a local model, and a manifest saying what each offered run
    /// became.
    Dataset(DatasetArgs),
}

/// The arguments of `train`.
#[derive(Args, Debug)]
struct TrainArgs {
    /// The dataset `dataset` wrote.
    #[arg(long)]
    dataset: PathBuf,
    /// The base model, `local:<org>/<model>` (no adapter).
    #[arg(long, default_value = DEFAULT_MODEL)]
    base: String,
    /// Optimiser steps.
    #[arg(long, default_value_t = 60)]
    steps: u32,
    /// LoRA rank.
    #[arg(long, default_value_t = 16)]
    rank: u32,
    /// LoRA alpha.
    #[arg(long, default_value_t = 32.0)]
    alpha: f32,
    /// Peak learning rate (default: brain's).
    #[arg(long)]
    lr: Option<f32>,
    /// Records averaged into one optimiser step.
    #[arg(long, default_value_t = 2)]
    records_per_step: u32,
    /// Seed.
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Hold the frozen base at bf16.
    #[arg(long)]
    bf16: bool,
}

#[derive(Subcommand, Debug)]
enum ModelsCommand {
    /// Every version and where it stands.
    List {
        /// The base model of the registry.
        #[arg(long, default_value = DEFAULT_MODEL)]
        base: String,
    },
    /// The model reference to run with now: the base, or the base with the promoted adapter.
    Current {
        /// The base model of the registry.
        #[arg(long, default_value = DEFAULT_MODEL)]
        base: String,
    },
    /// Judge a candidate against the model in use on the same tasks, by the declared rule.
    Judge {
        /// The candidate's version id.
        #[arg(long)]
        candidate: String,
        /// Directory of `run --json` outputs of the model in use, one file per task.
        #[arg(long)]
        baseline_results: PathBuf,
        /// Directory of the candidate's outputs on the same tasks (same file names).
        #[arg(long)]
        candidate_results: PathBuf,
        /// The base model of the registry.
        #[arg(long, default_value = DEFAULT_MODEL)]
        base: String,
    },
    /// Return to the version that was in use before the present one.
    Rollback {
        /// The base model of the registry.
        #[arg(long, default_value = DEFAULT_MODEL)]
        base: String,
    },
}

/// The arguments of `dataset`.
#[derive(Args, Debug)]
struct DatasetArgs {
    /// A run to offer; repeatable.
    #[arg(long = "run")]
    runs: Vec<String>,
    /// A directory of `agent-loop run --json` outputs; every run named in one is offered.
    #[arg(long)]
    outcomes: Option<PathBuf>,
    /// Where to write the dataset (the manifest is written beside it).
    #[arg(long)]
    out: PathBuf,
}

/// The arguments of `run`.
#[derive(Args, Debug)]
struct RunArgs {
    /// The repository the task is about (a git working tree, left untouched).
    #[arg(long)]
    workspace: PathBuf,
    /// The task.
    #[arg(long, conflicts_with = "task_file")]
    task: Option<String>,
    /// A file holding the task.
    #[arg(long)]
    task_file: Option<PathBuf>,
    /// A hidden acceptance check, `NAME=COMMAND`, run in the candidate
    /// checkout; the worker is told its name and nothing else. Repeatable.
    #[arg(long = "accept", value_name = "NAME=COMMAND")]
    accept: Vec<String>,
    /// An acceptance check the worker is shown, `NAME=COMMAND`; repeatable.
    #[arg(long = "accept-visible", value_name = "NAME=COMMAND")]
    accept_visible: Vec<String>,
    /// A shell command run once in the fresh checkout before the first attempt; repeatable.
    #[arg(long = "setup", value_name = "COMMAND")]
    setup: Vec<String>,
    /// A file or directory the worker must not change; repeatable.
    #[arg(long = "protect", value_name = "PATH")]
    protect: Vec<String>,
    /// The model: `local:<checkpoint>[+<adapter>][@<tokens>]` or `remote:<provider>/<name>`.
    #[arg(long, default_value = DEFAULT_MODEL)]
    model: String,
    /// Allow models reached over an API. Without it a remote model is refused.
    #[arg(long)]
    allow_api_models: bool,
    /// Seconds one attempt may take.
    #[arg(long, default_value_t = defaults::ATTEMPT_SECS)]
    attempt_secs: u64,
    /// Seconds the whole run may take.
    #[arg(long, default_value_t = defaults::TOTAL_SECS)]
    total_secs: u64,
    /// Output tokens per attempt.
    #[arg(long, default_value_t = defaults::MAX_OUTPUT_TOKENS)]
    max_output_tokens: u64,
    /// Tool calls per attempt.
    #[arg(long, default_value_t = defaults::MAX_TOOL_CALLS)]
    max_tool_calls: u32,
    /// Times one call may return the same answer before the attempt is
    /// stopped as going nowhere (0: no bound).
    #[arg(long, default_value_t = defaults::MAX_REPEATED_RESULTS)]
    max_repeats: u32,
    /// Attempts, the first included.
    #[arg(long, default_value_t = defaults::MAX_ATTEMPTS)]
    max_attempts: u32,
    /// Retries of a model call that failed in transit, per attempt.
    #[arg(long, default_value_t = defaults::PROVIDER_RETRIES)]
    provider_retries: u32,
    /// Further rounds per attempt, in the same conversation, after the worker stops with the checks still red.
    #[arg(long, default_value_t = defaults::FOLLOW_UPS)]
    follow_ups: u32,
    /// The most a remote model may cost, in US dollars as its provider reports it.
    #[arg(long)]
    max_cost_usd: Option<f64>,
    /// Seconds an acceptance check may take.
    #[arg(long, default_value_t = defaults::CHECK_SECS)]
    check_secs: u64,
    /// A hint from the supervisor. A run given any is recorded as assisted.
    #[arg(long = "hint")]
    hints: Vec<String>,
    /// Print the outcome as JSON.
    #[arg(long)]
    json: bool,
}

/// The configuration the loop builds Splinter with: its state under the
/// home directory, models from brain's store, nothing read from the
/// environment by the libraries.
fn config(cli: &Cli) -> Result<Config> {
    let home = home_of(cli)?;
    let model_store = cli
        .models_dir
        .clone()
        .or_else(|| std::env::var_os("BRAIN_MODELS_DIR").map(PathBuf::from))
        .unwrap_or_else(|| home.join(".local/share/brain/models"));
    Ok(Config {
        state_root: StateRoot::under_home(&home),
        policy_base: model_store.join("Qwen/Qwen3-0.6B"),
        model_store,
        policy_context_tokens: None,
        openrouter_api_key: std::env::var("AGENT_OPENROUTER_KEY").ok(),
        brain_api_key: std::env::var("BRAIN_API_KEY").ok(),
        allow_remote: false,
        command_env: std::collections::BTreeMap::new(),
        working_dir: home,
        brain_binary: None,
        front_door_model: None,
        assistant_model: None,
        judge_model: None,
        bf16_base: false,
        default_budget: None,
        remote_concurrency: 1,
        min_calibration_controls: 1,
        thinking: cli.thinking,
        served_temperature: cli.temperature.filter(|t| (0.0..=2.0).contains(t)),
    })
}

/// The home directory in use.
fn home_of(cli: &Cli) -> Result<PathBuf> {
    cli.home
        .clone()
        .or_else(|| std::env::var_os("HOME").map(PathBuf::from))
        .context("no home directory: pass --home or set HOME")
}

/// Parses the command line and runs it.
pub fn main() -> Result<ExitCode> {
    let cli = Cli::parse();
    let home = LoopHome::under(&home_of(&cli)?);
    match &cli.command {
        Command::Run(args) => {
            let contract = contract_of(args, &home)?;
            let splinter = Splinter::builder(config(&cli)?)
                .allow_remote(args.allow_api_models)
                .build()?;
            let outcome = execute(&splinter, &home, Request::New(Box::new(contract)))?;
            report(&outcome, args.json)
        }
        Command::Resume { run } => {
            let contract: Contract = read_json(&home.run_dir(run).join(CONTRACT_FILE))?;
            let splinter = Splinter::builder(config(&cli)?)
                .allow_remote(contract.allow_api_models)
                .build()?;
            let outcome = execute(&splinter, &home, Request::Resume(run.clone()))?;
            report(&outcome, true)
        }
        Command::Status { run } => status(&cli, &home, run),
        Command::Cancel { run } => {
            let splinter = Splinter::builder(config(&cli)?).build()?;
            let asked = runs::cancel(
                &splinter.context(),
                &resolve_splinter_run(&cli, &home, run)?,
            )?;
            println!("{}", serde_json::to_string_pretty(&asked)?);
            Ok(ExitCode::SUCCESS)
        }
        Command::Apply { run } => apply(&home, run),
        Command::Dataset(args) => dataset(&home, args),
        Command::Train(args) => train_candidate(&cli, &home, args),
        Command::Models { command } => models(&home, command),
        Command::Runs => {
            let splinter = Splinter::builder(config(&cli)?).build()?;
            let list = runs::list(&splinter.context())?;
            println!("{}", serde_json::to_string_pretty(&list)?);
            Ok(ExitCode::SUCCESS)
        }
    }
}

/// The contract a `run` command asks for, with the baseline read from the
/// repository now.
fn contract_of(args: &RunArgs, home: &LoopHome) -> Result<Contract> {
    let repository = args
        .workspace
        .canonicalize()
        .with_context(|| format!("the workspace {} does not exist", args.workspace.display()))?;
    let task = match (&args.task, &args.task_file) {
        (Some(task), None) => task.clone(),
        (None, Some(file)) => {
            std::fs::read_to_string(file).with_context(|| format!("reading {}", file.display()))?
        }
        _ => bail!("give the task with --task or --task-file"),
    };
    if !is_clean(&repository)? {
        bail!(
            "{} has uncommitted changes; the loop works from a committed baseline and leaves your changes alone",
            repository.display()
        );
    }
    let accept = args
        .accept_visible
        .iter()
        .map(|a| Check::parse(a, args.check_secs, true))
        .chain(
            args.accept
                .iter()
                .map(|a| Check::parse(a, args.check_secs, false)),
        )
        .collect::<Result<Vec<_>>>()?;
    if accept.is_empty() {
        bail!("give at least one acceptance check with --accept or --accept-visible NAME=COMMAND: without one nothing decides success");
    }
    let (system, source) = system_prompt(home)?;
    let definitions_digest = crate::definitions::effective(&repository).digest;
    Ok(Contract {
        schema: CONTRACT_SCHEMA,
        task,
        baseline_rev: head_rev(&repository)?,
        repository,
        setup: args.setup.clone(),
        acceptance: accept,
        protected: args.protect.clone(),
        limits: Limits {
            attempt_secs: args.attempt_secs,
            total_secs: args.total_secs,
            max_output_tokens: args.max_output_tokens,
            max_tool_calls: args.max_tool_calls,
            max_repeated_results: args.max_repeats,
            max_attempts: args.max_attempts,
            provider_retries: args.provider_retries,
            follow_ups: args.follow_ups,
            max_cost_usd: args.max_cost_usd,
        },
        model: args.model.clone(),
        allow_api_models: args.allow_api_models,
        system_prompt_digest: Digest::sha256_of(system.as_bytes()).to_string(),
        system_prompt_source: source,
        definitions_digest,
        hints: args.hints.clone(),
    })
}

/// Prints the outcome and returns the exit code that says how it ended:
/// 0 accepted, 1 not accepted, 3 cancelled.
fn report(outcome: &Outcome, json: bool) -> Result<ExitCode> {
    if json {
        println!("{}", serde_json::to_string_pretty(outcome)?);
    } else {
        println!(
            "run {}: {:?} ({})",
            outcome.run, outcome.status, outcome.termination
        );
        println!(
            "assistance: {}; model: {}",
            outcome.assistance, outcome.model_identity
        );
        for file in &outcome.changed_files {
            println!("  {} {}", file.status, file.path);
        }
        for item in &outcome.unresolved {
            println!("  unresolved: {item}");
        }
        println!("evidence: {}", outcome.artifacts.run_dir.display());
    }
    Ok(match outcome.status {
        Status::Accepted => ExitCode::SUCCESS,
        Status::Cancelled => ExitCode::from(3),
        _ => ExitCode::from(1),
    })
}

/// The Splinter run recorded for a loop run: the newest whose arguments or
/// id name it. A loop run keeps its own id; `cancel` needs the run Splinter
/// recorded it under.
fn resolve_splinter_run(cli: &Cli, home: &LoopHome, run: &str) -> Result<String> {
    let first = read_first_event(&home.run_dir(run))?;
    let recorded = first["data"]["splinter_run"]
        .as_str()
        .unwrap_or(run)
        .to_string();
    let _ = cli;
    Ok(recorded)
}

fn read_first_event(dir: &Path) -> Result<serde_json::Value> {
    let events = crate::trace::read_events(dir)?;
    events
        .into_iter()
        .next()
        .context("the run has no events yet")
}

fn status(cli: &Cli, home: &LoopHome, run: &str) -> Result<ExitCode> {
    let dir = home.run_dir(run);
    let splinter = Splinter::builder(config(cli)?).build()?;
    let recorded = resolve_splinter_run(cli, home, run)
        .ok()
        .and_then(|id| runs::show(&splinter.context(), &id).ok());
    let view = serde_json::json!({
        "run": run,
        "recorded": recorded.map(|r| serde_json::json!({"status": r.status, "started_at": r.started_at, "updated_at": r.updated_at})),
        "checkpoint": read_json::<serde_json::Value>(&dir.join(CHECKPOINT_FILE)).ok(),
        "outcome": read_json::<serde_json::Value>(&dir.join(OUTCOME_FILE)).ok(),
        "events": dir.join(crate::trace::EVENTS_FILE),
    });
    println!("{}", serde_json::to_string_pretty(&view)?);
    Ok(ExitCode::SUCCESS)
}

/// Applies the patch of an accepted run to the repository the run started
/// on, which must be clean and still at the baseline.
fn apply(home: &LoopHome, run: &str) -> Result<ExitCode> {
    let dir = home.run_dir(run);
    let outcome: Outcome = read_json(&dir.join(OUTCOME_FILE))?;
    if outcome.status != Status::Accepted {
        bail!(
            "run {run} was not accepted ({:?}); nothing is applied",
            outcome.status
        );
    }
    let contract: Contract = read_json(&dir.join(CONTRACT_FILE))?;
    if !is_clean(&contract.repository)? || head_rev(&contract.repository)? != contract.baseline_rev
    {
        bail!(
            "{} is not clean at the baseline {}; apply the patch {} by hand",
            contract.repository.display(),
            contract.baseline_rev,
            dir.join(PATCH_FILE).display()
        );
    }
    let patch = dir.join(PATCH_FILE);
    git(
        &contract.repository,
        &["apply", "--index", &patch.to_string_lossy()],
    )?;
    println!(
        "applied {} to {}",
        patch.display(),
        contract.repository.display()
    );
    Ok(ExitCode::SUCCESS)
}

/// Writes the dataset of the runs offered and reports what became of each.
fn dataset(home: &LoopHome, args: &DatasetArgs) -> Result<ExitCode> {
    let mut runs = args.runs.clone();
    if let Some(dir) = &args.outcomes {
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .with_context(|| format!("reading {}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .collect();
        files.sort();
        for file in files {
            // Output of a run that failed before it began has no run id.
            if let Ok(outcome) = read_json::<serde_json::Value>(&file) {
                if let Some(run) = outcome["run"].as_str() {
                    runs.push(run.to_string());
                }
            }
        }
    }
    let manifest = crate::dataset::export(home, &runs, &args.out)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(ExitCode::SUCCESS)
}

/// The directory of the checkpoint a `local:<org>/<model>` reference names.
fn checkpoint_dir(cli: &Cli, reference: &str) -> Result<PathBuf> {
    let name = reference
        .strip_prefix("local:")
        .with_context(|| format!("{reference:?} is not a local: reference"))?;
    let name = name.split(['@', '+']).next().unwrap_or(name);
    let store = config(cli)?.model_store;
    Ok(if Path::new(name).is_absolute() {
        PathBuf::from(name)
    } else {
        store.join(name)
    })
}

/// Trains a candidate adapter and prints what it was and how it did on the
/// records held out of its training.
fn train_candidate(cli: &Cli, home: &LoopHome, args: &TrainArgs) -> Result<ExitCode> {
    let settings = crate::training::Settings {
        steps: args.steps,
        rank: args.rank,
        alpha: args.alpha,
        learning_rate: args.lr,
        records_per_step: args.records_per_step,
        seed: args.seed,
        bf16_base: args.bf16,
    };
    let candidate = crate::training::train(
        home,
        &args.dataset,
        &checkpoint_dir(cli, &args.base)?,
        &args.base,
        &settings,
    )?;
    println!("{}", serde_json::to_string_pretty(&candidate)?);
    Ok(ExitCode::SUCCESS)
}

fn models(home: &LoopHome, command: &ModelsCommand) -> Result<ExitCode> {
    use crate::models::{current_ref, judge, load, pair_results, rollback};
    match command {
        ModelsCommand::List { base } => {
            println!("{}", serde_json::to_string_pretty(&load(home, base)?)?)
        }
        ModelsCommand::Current { base } => println!("{}", current_ref(&load(home, base)?)?),
        ModelsCommand::Judge {
            candidate,
            baseline_results,
            candidate_results,
            base,
        } => {
            let pairs = pair_results(baseline_results, candidate_results)?;
            let decision = judge(home, base, candidate, &pairs)?;
            println!("{}", serde_json::to_string_pretty(&decision)?);
            return Ok(if decision.promoted {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            });
        }
        ModelsCommand::Rollback { base } => {
            let now = rollback(home, base)?;
            println!("{}", now.unwrap_or_else(|| base.clone()));
        }
    }
    Ok(ExitCode::SUCCESS)
}
