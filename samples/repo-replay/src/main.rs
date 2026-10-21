// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! `repo-replay`: replay a git history with an agent and keep what verifies.
//!
//! ```text
//! audit     the sample checks itself: no model, no GPU, no network.
//! run       replay a repository's commits with a served model; one stored
//!           experience per commit attempt (positives and negatives).
//! report    the bias report of a state directory: size, style, reasons; with
//!           --versus, the grounded run beside its plan-only control.
//! dataset   the supervised dataset of the passed attempts, under the serving prompt.
//! ```

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context as _, Result};
use clap::{Args, Parser, Subcommand};
use splinter_sdk::agent::solve::Model;
use splinter_sdk::model::selection::{served_model_sampled, ServedSampling};

use splinter_repo_replay::audit::{audit_and_print, default_config, open_context};
use splinter_repo_replay::budget::Budget;
use splinter_repo_replay::compare::Comparison;
use splinter_repo_replay::control::Control;
use splinter_repo_replay::dataset::{
    attempts_set_filtered, build_dataset, labels, CorrectionRange, DatasetSettings,
};
use splinter_repo_replay::gate::GateConfig;
use splinter_repo_replay::harness::{
    Decoding, HarnessConfig, DEFAULT_DOCUMENT, DEFAULT_PLAN_ONLY_DOCUMENT,
};
use splinter_repo_replay::ledger;
use splinter_repo_replay::pipeline::{run, Models};
use splinter_repo_replay::plan::{changed_lines, validate, PlanCheck};
use splinter_repo_replay::plan_prompt::PlanStyle;
use splinter_repo_replay::regate::{regate, RegateConfig};
use splinter_repo_replay::select::{
    SelectConfig, DEFAULT_CONTEXT_SHARE_PERCENT, DEFAULT_CONTEXT_TOKENS,
};
use splinter_repo_replay::serving_prompt::SERVING_PROMPT;
use splinter_repo_replay::splits::write_splits;

#[derive(Parser, Debug)]
#[command(
    name = "repo-replay",
    version,
    about = "A git history replayed by an agent into verified SFT data"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Check the sample with scripted models: no model, no GPU, no network.
    Audit {
        /// Run in this directory and leave its content behind (default: a temporary one).
        #[arg(long)]
        work: Option<PathBuf>,
    },
    /// Replay a repository's history.
    Run(Box<RunArgs>),
    /// Print the bias report of a state directory.
    Report {
        /// The state directory of a run.
        #[arg(long)]
        state: PathBuf,
        /// The state directory of a plan-only control run: print its pass rates beside this state's.
        #[arg(long)]
        versus: Option<PathBuf>,
        /// Print JSON instead of a table.
        #[arg(long)]
        json: bool,
    },
    /// The plan gate of the harness document: reads a plan on standard input, exits 0 when it
    /// may be used, else prints why and exits 1.
    CheckPlan(CheckPlanArgs),
    /// Judge the passed attempts of a state directory again with the current gate (no model).
    Regate(RegateArgs),
    /// Build the dataset of the passed attempts of a state directory.
    Dataset(DatasetArgs),
}

#[derive(Args, Debug)]
struct CheckPlanArgs {
    /// The plan style: terse or detailed.
    #[arg(long)]
    style: String,
    /// The commit the plan is for; a plan must not carry it.
    #[arg(long)]
    commit: String,
    /// The commit's diff.
    #[arg(long)]
    diff_file: PathBuf,
    /// A diff line of this many words may not be repeated in the plan.
    #[arg(long)]
    min_line_words: usize,
}

#[derive(Args, Debug)]
struct RegateArgs {
    /// The state directory of a run (work on a copy of it to try the rule first).
    #[arg(long)]
    state: PathBuf,
    /// The repository the state was made from.
    repository: PathBuf,
    /// Record the new verdicts: a ledger line and a failing verdict per attempt that now fails
    /// (default: report only, nothing is written).
    #[arg(long)]
    write: bool,
    /// Most changed lines of a replayable commit (as in the run).
    #[arg(long, default_value_t = SelectConfig::default().max_changed_lines)]
    max_changed_lines: usize,
    /// Most changed files of a replayable commit (as in the run).
    #[arg(long, default_value_t = SelectConfig::default().max_files)]
    max_files: usize,
    /// The served model's context in tokens (as in the run; sets the diff size bound).
    #[arg(long, default_value_t = DEFAULT_CONTEXT_TOKENS)]
    context_tokens: usize,
    /// The share of the context, in percent, a commit's diff may take (as in the run).
    #[arg(long, default_value_t = DEFAULT_CONTEXT_SHARE_PERCENT)]
    max_diff_context_percent: usize,
    /// Edit calls a changed file needs.
    #[arg(long, default_value_t = GateConfig::default().min_edits_per_file)]
    min_edits_per_file: usize,
    /// Fewest tool calls an attempt may make.
    #[arg(long, default_value_t = GateConfig::default().min_tool_calls)]
    min_tool_calls: usize,
    /// Tool calls every attempt gets, before the size of the commit adds to it.
    #[arg(long, default_value_t = Budget::default().base)]
    calls_base: usize,
    /// Tool calls each changed file adds.
    #[arg(long, default_value_t = Budget::default().per_file)]
    calls_per_file: usize,
    /// Tool calls each hunk adds.
    #[arg(long, default_value_t = Budget::default().per_hunk)]
    calls_per_hunk: usize,
    /// The most tool calls any attempt gets.
    #[arg(long, default_value_t = Budget::default().cap)]
    calls_cap: usize,
}

#[derive(Args, Debug)]
struct RunArgs {
    /// The git repository to replay (read through a bare clone, never written).
    repository: PathBuf,
    /// Where the run keeps its clone, scratch trees, ledger and store.
    #[arg(long)]
    state: PathBuf,
    /// Replay the commits after this revision.
    #[arg(long)]
    from: Option<String>,
    /// Replay the history up to this revision (default HEAD).
    #[arg(long)]
    to: Option<String>,
    /// Replay at most this many commits: the most recent replayable ones.
    #[arg(long)]
    max_commits: Option<usize>,
    /// The served model's endpoint (OpenAI-compatible, ending in /v1).
    #[arg(long)]
    base_url: String,
    /// A JSON file `{"openai": "<key>"}` holding the server's key.
    #[arg(long)]
    api_key_file: PathBuf,
    /// The served model's name.
    #[arg(long)]
    model: String,
    /// The seed every request carries (required: no default is fixed).
    #[arg(long)]
    seed: u64,
    /// Sampling temperature (default 0.7, the model card's recommendation for the 27B).
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,
    /// Nucleus sampling (default 0.8).
    #[arg(long, default_value_t = 0.8)]
    top_p: f32,
    /// Sample among this many likeliest tokens (default 20).
    #[arg(long, default_value_t = 20)]
    top_k: u32,
    /// Let the model reason before it answers (default: it does not).
    #[arg(long)]
    thinking: bool,
    /// With --thinking: the most thinking tokens per response.
    #[arg(long)]
    thinking_budget: Option<u32>,
    /// What the agent is given: `grounded` (the plan and the masked change) or `plan-only`
    /// (the control: the plan and the repository only; use a state directory of its own).
    #[arg(long, default_value = "grounded")]
    control: String,
    /// Plan styles to attempt per commit, comma separated: terse, detailed.
    #[arg(long, value_delimiter = ',', default_value = "terse,detailed")]
    styles: Vec<String>,
    /// Attempts run at once.
    #[arg(long, default_value_t = 1)]
    jobs: usize,
    /// Seconds one attempt may take: a backstop far above any queueing, since the
    /// tool-call budget (by commit size) is what bounds an attempt.
    #[arg(long, default_value_t = 86_400)]
    attempt_secs: u64,
    /// Seconds one plan may take.
    #[arg(long, default_value_t = 900)]
    plan_secs: u64,
    /// Output tokens one attempt may generate.
    #[arg(long)]
    max_output_tokens: Option<u64>,
    /// Seconds a request may wait for the server's first byte and token (a
    /// shared server queues requests; default: sven's limit).
    #[arg(long)]
    first_response_secs: Option<u64>,
    /// Times an attempt that stopped announcing a call it did not make is told to go on.
    #[arg(long, default_value_t = 2)]
    follow_ups: usize,
    /// Plans written per attempt before giving up.
    #[arg(long, default_value_t = 2)]
    plan_tries: usize,
    /// A diff line of this many words may not be repeated in a plan.
    #[arg(long, default_value_t = 6)]
    plan_verbatim_words: usize,
    /// Most changed lines of a replayable commit.
    #[arg(long, default_value_t = SelectConfig::default().max_changed_lines)]
    max_changed_lines: usize,
    /// Most changed files of a replayable commit.
    #[arg(long, default_value_t = SelectConfig::default().max_files)]
    max_files: usize,
    /// The served model's context in tokens (default: what the server reports,
    /// else 32768).
    #[arg(long)]
    context_tokens: Option<usize>,
    /// The share of the context, in percent, a commit's diff may take; larger
    /// commits are skipped as `diff_over_context_share`.
    #[arg(long, default_value_t = DEFAULT_CONTEXT_SHARE_PERCENT)]
    max_diff_context_percent: usize,
    /// Edit calls a changed file needs.
    #[arg(long, default_value_t = GateConfig::default().min_edits_per_file)]
    min_edits_per_file: usize,
    /// Fewest tool calls an attempt may make.
    #[arg(long, default_value_t = GateConfig::default().min_tool_calls)]
    min_tool_calls: usize,
    /// Tool calls every attempt gets, before the size of the commit adds to it.
    #[arg(long, default_value_t = Budget::default().base)]
    calls_base: usize,
    /// Tool calls each changed file adds.
    #[arg(long, default_value_t = Budget::default().per_file)]
    calls_per_file: usize,
    /// Tool calls each hunk adds.
    #[arg(long, default_value_t = Budget::default().per_hunk)]
    calls_per_hunk: usize,
    /// Tool calls each correction round adds.
    #[arg(long, default_value_t = Budget::default().per_correction)]
    calls_per_correction: usize,
    /// Correction messages a wrong attempt may be sent by the supervisor (the
    /// same model, in a user's voice; default 0: none). Records of corrected
    /// attempts train only on what the agent did after the last correction.
    #[arg(long, default_value_t = 0)]
    corrections: usize,
    /// The most tool calls any attempt gets.
    #[arg(long, default_value_t = Budget::default().cap)]
    calls_cap: usize,
    /// A project check run in the attempt tree after the edits, through `sh -c`;
    /// its outcome is recorded, never gated. It executes the history's code on
    /// this machine. Repeatable.
    #[arg(long = "check")]
    checks: Vec<String>,
    /// Who drives the plan and the work: `rust` (the reference) or `harness` (the sven harness
    /// document, see --harness-file). Settings the document cannot express yet (plan tries or
    /// follow-ups other than 2, corrections, a thinking budget) keep that step on the Rust path.
    #[arg(long, default_value = "rust")]
    driver: String,
    /// The harness document for --driver harness (default: the one the sample ships).
    #[arg(long)]
    harness_file: Option<PathBuf>,
    /// The harness document of the plan-only control (default: the one the sample ships).
    #[arg(long)]
    harness_plan_only_file: Option<PathBuf>,
    /// Passes after the first that retry attempts lost to the server or the disk.
    #[arg(long, default_value_t = 3)]
    error_retries: usize,
    /// Redo the failed attempts whose reasons all lie in this gate class (repeatable):
    /// for a rule that was wrong and has been fixed.
    #[arg(long = "redo-class")]
    redo_classes: Vec<String>,
    /// Seconds to wait before a retry pass.
    #[arg(long, default_value_t = 120)]
    retry_wait_secs: u64,
    /// Seconds a project check may run.
    #[arg(long, default_value_t = 60)]
    check_secs: u64,
}

#[derive(Args, Debug)]
struct DatasetArgs {
    /// The state directory of a run.
    #[arg(long)]
    state: PathBuf,
    /// A file holding the serving prompt (default: splinter's coding-agent prompt).
    #[arg(long)]
    serving_prompt_file: Option<PathBuf>,
    /// Records longer than this many tokens are left out and counted.
    #[arg(long)]
    max_block: Option<usize>,
    /// Write temporal train, eval and test datasets, `splits.json` and one
    /// readable trace per attempt under this directory instead of one dataset.
    #[arg(long)]
    splits: Option<PathBuf>,
    /// Admit only attempts that needed at least this many corrections (1: the
    /// `corrected` record type only).
    #[arg(long, default_value_t = 0)]
    min_corrections: usize,
    /// Admit only attempts that needed at most this many corrections (0: the
    /// `first_try` record type only).
    #[arg(long)]
    max_corrections: Option<usize>,
    /// With --splits: the oldest this percent of the commits are train.
    #[arg(long, default_value_t = 80)]
    train_percent: usize,
    /// With --splits: the next this percent of the commits are eval, the rest test.
    #[arg(long, default_value_t = 10)]
    eval_percent: usize,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Audit { work } => audit_and_print(work.as_deref()),
        Command::Run(args) => run_command(&args),
        Command::Report {
            state,
            versus: Some(control),
            ..
        } => {
            print!(
                "{}",
                Comparison::of(&ledger::read(&state)?, &ledger::read(&control)?)
            );
            Ok(())
        }
        Command::Report { state, json, .. } => {
            let bias = ledger::bias(&ledger::read(&state)?);
            if json {
                println!("{}", serde_json::to_string_pretty(&bias)?);
            } else {
                print!("{bias}");
            }
            Ok(())
        }
        Command::CheckPlan(args) => check_plan_command(&args),
        Command::Regate(args) => regate_command(&args),
        Command::Dataset(args) => dataset_command(&args),
    }
}

fn run_command(args: &RunArgs) -> Result<()> {
    let styles = args
        .styles
        .iter()
        .map(|s| PlanStyle::parse(s).with_context(|| format!("unknown plan style {s:?}")))
        .collect::<Result<Vec<_>>>()?;
    let control = Control::parse(&args.control)
        .with_context(|| format!("unknown control {:?}: grounded or plan-only", args.control))?;
    let keys: serde_json::Value = serde_json::from_slice(&std::fs::read(&args.api_key_file)?)
        .with_context(|| format!("reading {}", args.api_key_file.display()))?;
    let key = keys
        .get("openai")
        .and_then(|k| k.as_str())
        .context("the key file has no \"openai\" entry")?;
    let loaded = served_model_sampled(
        &args.base_url,
        key,
        &args.model,
        &ServedSampling {
            temperature: Some(args.temperature),
            top_p: Some(args.top_p),
            top_k: Some(args.top_k),
            thinking: args.thinking,
            thinking_budget: args.thinking_budget,
            seed: Some(args.seed),
        },
    )?;
    let provider = loaded.provider();
    let models = Models {
        planner: Model::new(
            std::sync::Arc::clone(&provider),
            format!("openai/{}", args.model),
        ),
        supervisor: Model::new(
            std::sync::Arc::clone(&provider),
            format!("openai/{}", args.model),
        ),
        agent: provider,
        identity: format!("openai/{}", args.model),
    };
    let mut config = default_config(&args.repository, &args.state);
    config.from.clone_from(&args.from);
    config.to.clone_from(&args.to);
    config.max_commits = args.max_commits;
    config.styles = styles;
    config.control = control;
    config.corrections = args.corrections;
    config.jobs = args.jobs;
    config.deadline = Duration::from_secs(args.attempt_secs);
    config.plan_deadline = Duration::from_secs(args.plan_secs);
    config.max_output_tokens = args.max_output_tokens;
    config.follow_ups = args.follow_ups;
    config.first_response = args.first_response_secs.map(Duration::from_secs);
    config.plan_tries = args.plan_tries;
    config.min_line_tokens = args.plan_verbatim_words;
    config.harness = match args.driver.as_str() {
        "rust" => None,
        "harness" => Some(HarnessConfig {
            document: match &args.harness_file {
                Some(file) => std::fs::read_to_string(file)
                    .with_context(|| format!("reading {}", file.display()))?,
                None => DEFAULT_DOCUMENT.to_string(),
            },
            plan_only_document: match &args.harness_plan_only_file {
                Some(file) => std::fs::read_to_string(file)
                    .with_context(|| format!("reading {}", file.display()))?,
                None => DEFAULT_PLAN_ONLY_DOCUMENT.to_string(),
            },
            check_bin: std::env::current_exe().context("finding this executable")?,
            decoding: Decoding {
                temperature: f64::from(args.temperature),
                top_p: f64::from(args.top_p),
                top_k: u64::from(args.top_k),
                seed: args.seed,
                thinking: args.thinking,
            },
            thinking_budget: args.thinking_budget,
        }),
        other => anyhow::bail!("unknown driver {other:?}: rust or harness"),
    };
    let ctx = open_context(&args.state)?;
    let served_context = match args.context_tokens {
        Some(tokens) => tokens,
        None => ctx
            .runtime()
            .block_on(models.agent.probe_context_window())
            .map_or(DEFAULT_CONTEXT_TOKENS, |t| t as usize),
    };
    config.select = SelectConfig {
        max_changed_lines: args.max_changed_lines,
        max_files: args.max_files,
        ..SelectConfig::default()
    }
    .with_context_share(served_context, args.max_diff_context_percent);
    config.gate = GateConfig {
        min_edits_per_file: args.min_edits_per_file,
        min_tool_calls: args.min_tool_calls,
        max_tool_calls: GateConfig::default().max_tool_calls,
    };
    config.budget = Budget {
        base: args.calls_base,
        per_file: args.calls_per_file,
        per_hunk: args.calls_per_hunk,
        cap: args.calls_cap,
        per_correction: args.calls_per_correction,
    };
    config.record = serde_json::json!({
        "model": args.model, "seed": args.seed, "temperature": args.temperature, "top_p": args.top_p,
        "top_k": args.top_k, "thinking": args.thinking, "thinking_budget": args.thinking_budget,
        "base_url": args.base_url,
    });
    config.checks.clone_from(&args.checks);
    config.check_timeout = Duration::from_secs(args.check_secs);
    config.error_retries = args.error_retries;
    config.redo_classes.clone_from(&args.redo_classes);
    config.retry_wait = Duration::from_secs(args.retry_wait_secs);

    let report = ctx.runtime().block_on(run(&ctx, &models, &config))?;
    let passed = report
        .entries
        .iter()
        .filter(|e| e.outcome == ledger::Outcome::Passed)
        .count();
    println!(
        "replayable commits: {}; ledger lines written: {}; passed: {passed}",
        report.replayable,
        report.entries.len()
    );
    print!("{}", ledger::bias(&ledger::read(&args.state)?));
    Ok(())
}

fn check_plan_command(args: &CheckPlanArgs) -> Result<()> {
    use std::io::Read;
    let style = PlanStyle::parse(&args.style)
        .with_context(|| format!("unknown plan style {:?}", args.style))?;
    let diff = std::fs::read_to_string(&args.diff_file)
        .with_context(|| format!("reading {}", args.diff_file.display()))?;
    let mut plan = String::new();
    std::io::stdin().read_to_string(&mut plan)?;
    let lines = changed_lines(&diff);
    let check = PlanCheck {
        diff_lines: &lines,
        commit: &args.commit,
        min_line_tokens: args.min_line_words,
    };
    if let Err(rejection) = validate(&plan, style, &check) {
        // The refusal is what the planner is told; no newline, no decoration.
        print!("{rejection}");
        std::process::exit(1);
    }
    Ok(())
}

fn regate_command(args: &RegateArgs) -> Result<()> {
    let ctx = open_context(&args.state)?;
    let report = regate(
        &ctx,
        &RegateConfig {
            source: args.repository.clone(),
            state: args.state.clone(),
            select: SelectConfig {
                max_changed_lines: args.max_changed_lines,
                max_files: args.max_files,
                ..SelectConfig::default()
            }
            .with_context_share(args.context_tokens, args.max_diff_context_percent),
            gate: GateConfig {
                min_edits_per_file: args.min_edits_per_file,
                min_tool_calls: args.min_tool_calls,
                max_tool_calls: GateConfig::default().max_tool_calls,
            },
            budget: Budget {
                base: args.calls_base,
                per_file: args.calls_per_file,
                per_hunk: args.calls_per_hunk,
                cap: args.calls_cap,
                per_correction: Budget::default().per_correction,
            },
            write: args.write,
        },
    )?;
    println!(
        "passed attempts judged again: {}; still pass: {}; now fail: {}; corrected (not judged): {}; unreadable: {}",
        report.examined,
        report.kept(),
        report.failed.len(),
        report.not_judged_corrected,
        report.unreadable.len()
    );
    for (class, n) in report.by_class() {
        println!("  {class}: {n}");
    }
    for (commit, why) in &report.unreadable {
        println!("  unreadable {commit}: {why}");
    }
    println!(
        "{}",
        if args.write {
            "verdicts recorded in the ledger and the store"
        } else {
            "report only: nothing was written (--write records the verdicts)"
        }
    );
    Ok(())
}

fn dataset_command(args: &DatasetArgs) -> Result<()> {
    let serving_prompt = match &args.serving_prompt_file {
        Some(file) => std::fs::read_to_string(file)
            .with_context(|| format!("reading {}", file.display()))?
            .trim_end()
            .to_string(),
        None => SERVING_PROMPT.to_string(),
    };
    let ctx = open_context(&args.state)?;
    let range = CorrectionRange {
        min: args.min_corrections,
        max: args.max_corrections.unwrap_or(usize::MAX),
    };
    if let Some(out) = &args.splits {
        let settings = DatasetSettings {
            serving_prompt,
            max_block: args.max_block,
        };
        let summary = write_splits(
            &ctx,
            &args.state,
            out,
            &settings,
            args.train_percent,
            args.eval_percent,
            range,
        )?;
        println!("{}", serde_json::to_string_pretty(&summary)?);
        return Ok(());
    }
    let set = attempts_set_filtered(&ctx, &args.state, "all", &|e| range.admits(e))?;
    let built = build_dataset(
        &ctx,
        &[set],
        &DatasetSettings {
            serving_prompt,
            max_block: args.max_block,
        },
    )?;
    println!(
        "dataset {} ({} records) at {}",
        built.dataset,
        built.records,
        built.path.display()
    );
    for (reason, count) in &built.excluded {
        println!("  excluded {reason:?}: {count}");
    }
    let labels = labels(&args.state, range)?;
    for kind in ["first_try", "corrected"] {
        let of_kind = labels.iter().filter(|l| l.record_type == kind).count();
        println!("  passed attempts of type {kind}: {of_kind}");
    }
    let sidecar = built.path.with_extension("labels.json");
    std::fs::write(&sidecar, serde_json::to_vec_pretty(&labels)?)
        .with_context(|| format!("writing {}", sidecar.display()))?;
    println!(
        "  labels (record type, corrections, control per experience): {}",
        sidecar.display()
    );
    Ok(())
}
