// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that put an informative on-policy
// reinforcement-learning signal through a language model against verifiers it
// cannot influence, for its clients. If your team needs expertise in
// continuous policy-gradient campaigns you can procure our services by
// sending an email to info@swedishembedded.com.

//! Runs the campaign controller end to end on a real base model against the
//! Lua output-prediction family, in process, and prints what each arm
//! measured: the informative-group fraction, the KL to the reference, the
//! clip fraction and the dev gate.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use splinter_sdk::campaign::brain::{
    BrainTrainer, ChatCollector, ChatValidator, ModelSettings, TrainerSettings,
};
use splinter_sdk::campaign::{
    CampaignConfig, CampaignController, LearningStrategy, Parts, Stopwatch, WallClock,
};
use splinter_sdk::model::policy_gradient::PolicyGradientOptions;
use splinter_sdk::model::precision::BasePrecision;
use splinter_sdk::model::stats::BrainSignificance;
use splinter_sdk::store::StateRoot;
use splinter_sdk::vocabulary::clock::SystemClock;

use campaign_lua::{default_interpreter, LuaPredict, Sizes};

#[derive(Parser)]
#[command(
    name = "campaign-lua",
    about = "Run the campaign controller on a base model against Lua output prediction"
)]
struct Cli {
    #[command(flatten)]
    device: splinter_sdk::device::DeviceFlags,
    /// The base checkpoint directory.
    #[arg(long)]
    model: PathBuf,
    /// The state root and the adapters of the campaign.
    #[arg(long)]
    out: PathBuf,
    /// The campaign's name; running it again continues it.
    #[arg(long, default_value = "lua-predict")]
    name: String,
    /// The Lua interpreter.
    #[arg(long)]
    lua: Option<PathBuf>,
    /// Arms to run.
    #[arg(long, default_value_t = 4)]
    arms: u32,
    /// Training programs.
    #[arg(long, default_value_t = 160)]
    train_tasks: usize,
    /// Dev programs the gate is decided on.
    #[arg(long, default_value_t = 200)]
    dev_tasks: usize,
    /// Retention programs (levels 1 and 2).
    #[arg(long, default_value_t = 40)]
    retention_tasks: usize,
    /// Attempts per task of the calibration, 8 to 16.
    #[arg(long, default_value_t = 8)]
    calibration_k: usize,
    /// Attempts per task of a training group.
    #[arg(long, default_value_t = 8)]
    attempts: usize,
    /// Informative groups per batch.
    #[arg(long, default_value_t = 24)]
    batch_groups: usize,
    /// Optimiser steps per arm.
    #[arg(long, default_value_t = 16)]
    steps: u32,
    /// Learning rate.
    #[arg(long, default_value_t = 1e-5)]
    lr: f32,
    /// Adapter rank.
    #[arg(long, default_value_t = 16)]
    rank: u32,
    /// Records averaged into one optimiser step.
    #[arg(long, default_value_t = 8)]
    records_per_step: u32,
    /// The lead (share of dev programs won) an arm is meant to produce.
    #[arg(long, default_value_t = 0.1)]
    target_effect: f64,
    /// Expected share of dev programs a candidate and its champion differ on.
    #[arg(long, default_value_t = 0.2)]
    discordance: f64,
    /// Requests in flight at once.
    #[arg(long, default_value_t = 16)]
    concurrency: usize,
    /// Tokens a completion may take.
    #[arg(long, default_value_t = 24)]
    max_tokens: u32,
    /// Cards held, for the GPU-hours recorded.
    #[arg(long, default_value_t = 1)]
    gpus: u32,
    /// The seed of the program generator and of every draw; drawn and printed
    /// when absent.
    #[arg(long)]
    seed: Option<u64>,
    /// Strategies in order (`policy-gradient`, `positive-only`, `repair`,
    /// `distill`); a plateau moves to the next.
    #[arg(long, value_delimiter = ',', default_value = "policy-gradient")]
    ladder: Vec<String>,
    /// What to do with a group that all failed (`skip`, `cliff-distill`).
    #[arg(long, default_value = "skip")]
    all_fail: String,
    /// Print this many raw completions of the base model and exit: what the
    /// policy writes and how it is graded, before any campaign.
    #[arg(long)]
    probe: Option<usize>,
    /// Largest mean KL to the reference a step may show before the campaign
    /// halts.
    #[arg(long, default_value_t = 0.5)]
    max_kl: f64,
    /// Fewest informative groups, as a share of those collected, before a
    /// step is reported starved.
    #[arg(long, default_value_t = 0.25)]
    min_informative: f64,
    /// Precision of the frozen base during training: `bf16` or `fp32`.
    #[arg(long, default_value = "bf16")]
    base_precision: String,
    /// Fraction of the batch's records replayed from earlier arms.
    #[arg(long, default_value_t = 0.25)]
    replay: f64,
}

fn strategy(name: &str) -> Result<LearningStrategy> {
    Ok(match name {
        "policy-gradient" => LearningStrategy::PolicyGradient,
        "positive-only" => LearningStrategy::PositiveOnly,
        "repair" => LearningStrategy::Repair,
        "distill" => LearningStrategy::Distill,
        other => anyhow::bail!("{other:?} is not a strategy"),
    })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    cli.device.apply().map_err(|e| anyhow::anyhow!("{e}"))?;
    let seed = splinter_sdk::seed::resolve("seed", cli.seed);
    std::fs::create_dir_all(&cli.out).with_context(|| format!("creating {}", cli.out.display()))?;
    let scratch = cli.out.join("lua-programs");
    std::fs::create_dir_all(&scratch)?;
    let lua = cli.lua.clone().unwrap_or_else(default_interpreter);
    let family = LuaPredict::generate(
        &lua,
        &scratch,
        Sizes {
            pool: cli.train_tasks,
            dev: cli.dev_tasks,
            retention: cli.retention_tasks,
        },
        seed,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut model = ModelSettings::new(&cli.model);
    model.max_tokens = cli.max_tokens;
    model.concurrency = cli.concurrency;
    let mut collector =
        ChatCollector::new(&family, model.clone()).map_err(|e| anyhow::anyhow!("{e}"))?;
    if let Some(count) = cli.probe {
        use splinter_sdk::campaign::brain::TaskFamily as _;
        use splinter_sdk::campaign::{CollectRequest, PolicyRef, RolloutCollector};
        let tasks: Vec<String> = family.pool().into_iter().take(count).collect();
        let groups = collector
            .collect(&CollectRequest {
                tasks,
                attempts: 2,
                policy: PolicyRef::default(),
                seed,
            })
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        for group in groups {
            let program = family.program(&group.task);
            for rollout in group.rollouts {
                println!(
                    "{}\texpected {:?}\tpassed {}\tcompletion {:?}",
                    group.task,
                    program.map(|p| p.expected.as_str()),
                    rollout.passed,
                    rollout.completion
                );
            }
        }
        return Ok(());
    }
    let mut validator = ChatValidator::new(&family, model).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut trainer = BrainTrainer::new(TrainerSettings {
        base: cli.model.clone(),
        steps: cli.steps,
        rank: cli.rank,
        alpha: (cli.rank * 2) as f32,
        learning_rate: cli.lr,
        records_per_step: cli.records_per_step,
        max_block: Some(1024),
        options: PolicyGradientOptions {
            kl_beta: Some(0.0),
            reference: Some(splinter_sdk::model::policy_gradient::Reference::StartOfRun),
            behaviour: Some(splinter_sdk::model::policy_gradient::Behaviour::StartOfRun),
            temperature: Some(1.0),
            ..PolicyGradientOptions::default()
        },
        base_precision: match cli.base_precision.as_str() {
            "bf16" => BasePrecision::Bf16,
            "fp32" => BasePrecision::Fp32,
            other => anyhow::bail!("{other:?} is not a base precision: bf16 or fp32"),
        },
    });

    let mut config = CampaignConfig::new(
        &cli.name,
        "dynamic sampling on the pass-rate frontier puts an informative signal through the policy and raises dev accuracy on Lua output prediction",
    );
    config.calibration_k = cli.calibration_k;
    config.attempts_per_group = cli.attempts;
    config.batch.target_groups = cli.batch_groups;
    config.batch.max_request = cli.batch_groups * 4;
    config.target_effect = cli.target_effect;
    config.expected_discordance = cli.discordance;
    config.gpus = cli.gpus;
    config.replay.per_batch_record = cli.replay;
    config.bounds.max_kl = Some(cli.max_kl);
    config.bounds.min_informative_fraction = Some(cli.min_informative);
    config.ladder = cli
        .ladder
        .iter()
        .map(|s| strategy(s))
        .collect::<Result<_>>()?;
    config.all_fail = cli
        .all_fail
        .parse()
        .map_err(|e: String| anyhow::anyhow!(e))?;
    config.preflight = vec![splinter_sdk::preflight::Requirement {
        path: cli.out.clone(),
        min_free_bytes: 5 << 30,
        min_free_inodes: 20_000,
    }];

    let root = StateRoot::new(cli.out.join("state"));
    let clock = SystemClock;
    let watch = WallClock::start();
    let parts = Parts {
        collector: &mut collector,
        trainer: &mut trainer,
        validator: &mut validator,
        handler: None,
        significance: &BrainSignificance,
        clock: &clock,
        stopwatch: &watch,
        environments: None,
        progress: Some(&|line| eprintln!("[{}s] {line}", watch.seconds() as u64)),
    };
    let mut controller =
        CampaignController::open(&root, config, seed, parts, cli.out.join("adapters"))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    let report = controller
        .run(cli.arms)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    let ledger = controller.ledger().map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("{}", serde_json::to_string_pretty(&ledger)?);
    Ok(())
}
