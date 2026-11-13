// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that put an informative on-policy
// reinforcement-learning signal through a language model against verifiers it
// cannot influence, for its clients. If your team needs expertise in
// recursive self-improvement programs with an independent examiner you can
// procure our services by sending an email to info@swedishembedded.com.

//! One lineage of RSI2: generations of the campaign controller, each a single
//! arm from the previous champion, on tasks the current policy proposed from
//! the Lua 5.1 manual beside the seeded generator's. After every generation
//! the champion is handed to the examiner. Running the command again continues
//! the lineage: the campaign state, the proposals and the hand-offs are all
//! on disk.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use serde_json::json;
use splinter_sdk::campaign::served::{ServedCollector, ServedTrainer, ServedValidator};
use splinter_sdk::campaign::{CampaignController, Parts, PolicyRef, Stopwatch, WallClock};
use splinter_sdk::model::stats::BrainSignificance;
use splinter_sdk::store::StateRoot;
use splinter_sdk::vocabulary::clock::SystemClock;

use campaign_lua::family::Rsi2Family;
use campaign_lua::handoff::{self, Build, Candidate};
use campaign_lua::host::{HostSettings, ServerHost};
use campaign_lua::proposer::{as_proposals, Proposer};
use campaign_lua::recipe::Recipe;
use campaign_lua::{LuaPredict, Sizes};

#[derive(Parser)]
#[command(
    name = "rsi2",
    about = "Run generations of one RSI2 lineage and hand each champion to the examiner"
)]
struct Cli {
    #[command(flatten)]
    recipe: Recipe,
    /// The base checkpoint directory.
    #[arg(long)]
    model: PathBuf,
    /// The base model's name, as the examiner knows it.
    #[arg(long)]
    base_name: String,
    /// Brain's model store.
    #[arg(long)]
    models_dir: PathBuf,
    /// The `brain` binary that serves the lineage's current policy.
    #[arg(long)]
    brain: PathBuf,
    /// The card the server runs on, in brain's `--device` grammar.
    #[arg(long)]
    server_device: String,
    /// The server's KV-cache tier.
    #[arg(long, default_value = "f32")]
    kv_cache: String,
    /// Sequences the server decodes together.
    #[arg(long, default_value_t = 32)]
    slots: u32,
    /// The lineage's directory: campaign state, adapters, proposals.
    #[arg(long)]
    out: PathBuf,
    /// The lineage's name.
    #[arg(long)]
    lineage: String,
    /// The hand-off directory the examiner reads candidates from.
    #[arg(long)]
    handoff: PathBuf,
    /// Where a generation's champion is kept for the examiner, outside any
    /// collection.
    #[arg(long)]
    kept_adapters: PathBuf,
    /// A page of the Lua 5.1 manual the policy proposes tasks from (repeat).
    #[arg(long = "manual", required = true)]
    manuals: Vec<PathBuf>,
    /// The Lua 5.1 interpreter.
    #[arg(long)]
    lua: PathBuf,
    /// The controller's seed; drawn and printed when absent.
    #[arg(long)]
    seed: Option<u64>,
    /// The seed of the generator's programs, the same for every lineage;
    /// the controller's seed when absent.
    #[arg(long)]
    generator_seed: Option<u64>,
    /// Generations in the lineage.
    #[arg(long, default_value_t = 3)]
    generations: u32,
    /// Generations to run in this process.
    #[arg(long, default_value_t = 1)]
    run_generations: u32,
    /// Generator programs trained on.
    #[arg(long)]
    train_tasks: usize,
    /// Generator programs the dev gate is decided on.
    #[arg(long)]
    dev_tasks: usize,
    /// Generator programs of levels 1 and 2 kept for retention.
    #[arg(long)]
    retention_tasks: usize,
    /// Proposals of the base held out for the dev gate.
    #[arg(long)]
    proposed_dev: usize,
    /// Proposals each generation's policy is asked for to train on.
    #[arg(long)]
    proposals_per_generation: usize,
    /// The most requests a round of proposing makes.
    #[arg(long, default_value_t = 40)]
    proposal_calls: usize,
    /// Requests of a round of proposing asked at once.
    #[arg(long, default_value_t = 8)]
    proposal_parallel: usize,
    /// Splinter's commit, as the launch record states it.
    #[arg(long)]
    splinter_commit: String,
    /// Sven's commit.
    #[arg(long)]
    sven_commit: String,
    /// Brain's commit.
    #[arg(long)]
    brain_commit: String,
}

fn fail<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

struct Lineage<'a> {
    cli: &'a Cli,
    family: &'a Rsi2Family,
    build: Build,
    base_digest: String,
}

impl Lineage<'_> {
    fn hand_over(
        &self,
        generation: u32,
        champion: &PolicyRef,
        advanced: bool,
        verdict: Option<String>,
    ) -> Result<()> {
        let (adapter, digest) = match &champion.adapter {
            Some(path) => {
                let (kept, digest) = handoff::keep_adapter(
                    &self.cli.kept_adapters,
                    &self.cli.lineage,
                    generation,
                    path,
                )?;
                (Some(kept), Some(digest))
            }
            None => (None, None),
        };
        let candidate = Candidate {
            series: self.cli.lineage.clone(),
            step: generation,
            base: self.cli.base_name.clone(),
            base_digest: self.base_digest.clone(),
            serving: handoff::serving(&self.cli.model, adapter.as_deref()),
            adapter,
            adapter_digest: digest,
            advanced,
            arm_verdict: verdict,
            build: self.build.clone(),
        };
        let written = handoff::write(&self.cli.handoff, &candidate, &self.family.records())?;
        eprintln!(
            "generation {generation}: candidate {}",
            if written {
                "handed over"
            } else {
                "already handed over"
            }
        );
        Ok(())
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let seed = splinter_sdk::seed::resolve("seed", cli.seed);
    let generator_seed = cli.generator_seed.unwrap_or(seed);
    eprintln!(
        "lineage {}: controller seed {seed}, generator seed {generator_seed}",
        cli.lineage
    );
    for dir in [&cli.out, &cli.handoff, &cli.kept_adapters] {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let scratch = cli.out.join("lua-programs");
    std::fs::create_dir_all(&scratch)?;
    let generator = LuaPredict::generate(
        &cli.lua,
        &scratch,
        Sizes {
            pool: cli.train_tasks,
            dev: cli.dev_tasks,
            retention: cli.retention_tasks,
        },
        generator_seed,
    )
    .map_err(fail)?;
    let family = Rsi2Family::open(generator, &cli.out.join("proposals.jsonl")).map_err(fail)?;
    let lua_dir = cli
        .lua
        .parent()
        .map_or_else(|| Path::new(".").to_path_buf(), Path::to_path_buf);
    let host = ServerHost::new(HostSettings {
        brain: cli.brain.clone(),
        models: cli.models_dir.clone(),
        base: cli.model.clone(),
        work: cli.out.join("server-work"),
        state: cli.out.join("proposer-state"),
        store: cli.out.join("artifacts"),
        adapters: cli.out.join("served-adapters"),
        log: cli.out.join("brain-serve.log"),
        device: cli.server_device.clone(),
        kv_cache: cli.kv_cache.clone(),
        slots: cli.slots,
        lua_dir,
    });
    let proposer = Proposer::new(&cli.model, &cli.manuals);

    // Generation 1's policy is the base: it proposes the dev tasks (held out)
    // and the first training tasks before the campaign opens, since the dev
    // set is fixed when it does.
    if !family.has_generation(1) {
        let want = cli.proposed_dev + cli.proposals_per_generation;
        let ctx = host.context(&PolicyRef::default())?;
        let made = proposer.propose(
            &ctx,
            None,
            &family,
            want,
            cli.proposal_calls,
            cli.proposal_parallel,
        )?;
        eprintln!("the base proposed: {}", serde_json::to_string(&made)?);
        anyhow::ensure!(
            made.kept.len() >= cli.proposed_dev,
            "the base proposed {} new programs, fewer than the {} held out for the dev gate",
            made.kept.len(),
            cli.proposed_dev
        );
        family
            .add(as_proposals(made.kept, 1, cli.proposed_dev))
            .map_err(fail)?;
    }

    let lineage = Lineage {
        cli: &cli,
        family: &family,
        build: Build {
            splinter_commit: cli.splinter_commit.clone(),
            sven_commit: cli.sven_commit.clone(),
            brain_commit: cli.brain_commit.clone(),
            binary_digest: splinter_sdk::vocabulary::build::running_binary_digest()
                .map(|d| d.to_string()),
        },
        base_digest: handoff::base_digest(&cli.model)?,
    };

    let model = cli.recipe.model_settings(&cli.model);
    let mut collector = ServedCollector::new(&family, &host, model.clone()).map_err(fail)?;
    let mut validator = ServedValidator::new(&family, &host, model).map_err(fail)?;
    let mut trainer =
        ServedTrainer::new(cli.recipe.trainer_settings(&cli.model)?, host.endpoints()?)
            .map_err(fail)?;
    let config = cli.recipe.campaign_config(
        &cli.lineage,
        "on-policy policy gradient on the policy's own proposed programs and the generator's raises Lua output prediction under the neutral protocol",
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
            .map_err(fail)?;

    for _ in 0..cli.run_generations {
        let generation = controller.state().arms + 1;
        if generation > cli.generations {
            break;
        }
        let champion = controller.state().policy.clone();
        if !family.has_generation(generation) {
            let ctx = host.context(&champion)?;
            let made = proposer.propose(
                &ctx,
                champion.adapter.as_deref(),
                &family,
                cli.proposals_per_generation,
                cli.proposal_calls,
                cli.proposal_parallel,
            )?;
            eprintln!(
                "the policy of generation {generation} proposed: {}",
                serde_json::to_string(&made)?
            );
            family
                .add(as_proposals(made.kept, generation, 0))
                .map_err(fail)?;
        }
        let outcome = controller.run(1);
        let ledger = controller.ledger().map_err(fail)?;
        std::fs::write(
            cli.out.join(format!("ledger-generation-{generation}.json")),
            serde_json::to_vec_pretty(&ledger)?,
        )?;
        let report = outcome.map_err(fail)?;
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({"generation": generation, "report": report}))?
        );
        let arm = report.arms.first().context("the arm reported nothing")?;
        let after = controller.state().policy.clone();
        lineage.hand_over(
            generation,
            &after,
            after.version > champion.version,
            Some(format!("{:?}", arm.verdict)),
        )?;
    }
    host.stop();
    Ok(())
}
