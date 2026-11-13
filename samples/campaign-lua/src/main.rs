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
use splinter_sdk::campaign::brain::{BrainTrainer, ChatCollector, ChatValidator};
use splinter_sdk::campaign::{CampaignController, Parts, Stopwatch, WallClock};
use splinter_sdk::model::stats::BrainSignificance;
use splinter_sdk::store::StateRoot;
use splinter_sdk::vocabulary::clock::SystemClock;

use campaign_lua::recipe::Recipe;
use campaign_lua::{default_interpreter, LuaPredict, Sizes};

#[derive(Parser)]
#[command(
    name = "campaign-lua",
    about = "Run the campaign controller on a base model against Lua output prediction"
)]
struct Cli {
    #[command(flatten)]
    device: splinter_sdk::device::DeviceFlags,
    #[command(flatten)]
    recipe: Recipe,
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
    /// The seed of the program generator and of every draw; drawn and printed
    /// when absent.
    #[arg(long)]
    seed: Option<u64>,
    /// Print this many raw completions of the base model and exit: what the
    /// policy writes and how it is graded, before any campaign.
    #[arg(long)]
    probe: Option<usize>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    cli.device.apply().map_err(|e| anyhow::anyhow!("{e}"))?;
    let brain = cli.device.brain().map_err(|e| anyhow::anyhow!("{e}"))?;
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

    let model = cli.recipe.model_settings(&cli.model);
    let mut collector = ChatCollector::new(brain.endpoints().clone(), &family, model.clone())
        .map_err(|e| anyhow::anyhow!("{e}"))?;
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
    let mut validator = ChatValidator::new(brain.endpoints().clone(), &family, model)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut trainer = BrainTrainer::new(cli.recipe.trainer_settings(&cli.model)?);
    let config = cli.recipe.campaign_config(
        &cli.name,
        "dynamic sampling on the pass-rate frontier puts an informative signal through the policy and raises dev accuracy on Lua output prediction",
        &cli.out,
    )?;

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
    // The ledger is printed whether or not the run ended in an error: the
    // arms it finished are the evidence, and a refused or failed arm is one of
    // its entries.
    let outcome = controller.run(cli.arms);
    let ledger = controller.ledger().map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("{}", serde_json::to_string_pretty(&ledger)?);
    let report = outcome.map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
