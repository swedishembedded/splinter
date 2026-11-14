// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that let a model teach itself from
// an interpreter's answers to its own proposals, gated on held-out evidence,
// for its clients. If your team needs expertise in self-taught model
// lineages judged by an independent examiner you can procure our services by
// sending an email to info@swedishembedded.com.

//! One lineage: generations of self-taught rounds on Lua output prediction
//! (see `campaign_lua::lineage`), the champion of each handed to an examiner
//! when a hand-off directory is given. Running the command again continues
//! the lineage from what its journals hold.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use serde_json::json;
use splinter_sdk::campaign::brain::ModelSettings;
use splinter_sdk::campaign::served::ServedValidator;
use splinter_sdk::campaign::PolicyRef;
use splinter_sdk::measure::campaign::evidence::EvidenceGate;
use splinter_sdk::measure::campaign::gate::DevGate;

use campaign_lua::family::ProposalFamily;
use campaign_lua::frontier::FrontierArgs;
use campaign_lua::handoff::{self, Build, Candidate};
use campaign_lua::lineage::{
    champion_of, trained_up_to, Acceptance, Files, Lineage, Plan, TrainFrom,
};
use campaign_lua::plan::EvidencePlan;
use campaign_lua::proposer::{Budget, Proposer};
use campaign_lua::sft::Recipe;
use campaign_lua::{LuaPredict, Sizes};
use splinter_sdk::served_host::{HostSettings, ServedHost};

/// The gate a round's candidate must pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum GateKind {
    /// Advance on a net gain over the whole dev set (the registered runs).
    Greedy,
    /// Advance on a noninferiority e-process over dev items drawn per round,
    /// with the false evolution rate held to `--gate-q` across all rounds.
    Evidence,
}

/// The evidence gate's settings; each is required with `--gate evidence` and
/// refused without it.
#[derive(clap::Args)]
struct EvidenceArgs {
    /// The gate a round's candidate must pass.
    #[arg(long, value_enum, default_value_t = GateKind::Greedy)]
    gate: GateKind,
    /// The tolerated loss, on the paired difference scale [-1, 1]: a
    /// candidate worse by this much is a false evolution.
    #[arg(long)]
    gate_eps: Option<f64>,
    /// The false evolution rate budget over every planned round.
    #[arg(long)]
    gate_q: Option<f64>,
    /// The overlap correction: the optimism allowed on a dev item an earlier
    /// round used.
    #[arg(long)]
    gate_beta: Option<f64>,
    /// Dev items a round draws from the dev set, fewer than it holds.
    #[arg(long)]
    dev_draw: Option<usize>,
    /// Dev items graded at a time; a round stops at the first batch whose
    /// evidence crosses. Default: `--slots`.
    #[arg(long)]
    dev_batch: Option<usize>,
}

impl EvidenceArgs {
    fn acceptance(&self, slots: u32) -> Result<Acceptance> {
        let given = [
            self.gate_eps.is_some(),
            self.gate_q.is_some(),
            self.gate_beta.is_some(),
            self.dev_draw.is_some(),
            self.dev_batch.is_some(),
        ];
        match self.gate {
            GateKind::Greedy => {
                anyhow::ensure!(
                    !given.contains(&true),
                    "--gate-eps, --gate-q, --gate-beta, --dev-draw and --dev-batch need --gate evidence"
                );
                Ok(Acceptance::Greedy)
            }
            GateKind::Evidence => {
                let (Some(eps), Some(q), Some(beta), Some(draw)) =
                    (self.gate_eps, self.gate_q, self.gate_beta, self.dev_draw)
                else {
                    anyhow::bail!(
                        "--gate evidence needs --gate-eps, --gate-q, --gate-beta and --dev-draw, fixed before the lineage runs"
                    );
                };
                Ok(Acceptance::Evidence(EvidencePlan {
                    gate: EvidenceGate::new(eps, q, beta).map_err(|e| anyhow::anyhow!("{e}"))?,
                    draw,
                    batch: self.dev_batch.unwrap_or(slots as usize),
                }))
            }
        }
    }
}

#[derive(Parser)]
#[command(
    name = "lineage",
    about = "Run generations of one self-taught lineage and hand each champion to an examiner"
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
    /// Sequences the server decodes together, and requests kept in flight.
    #[arg(long, default_value_t = 32)]
    slots: u32,
    /// The lineage's directory: journals, adapters, proposals.
    #[arg(long)]
    out: PathBuf,
    /// The lineage's name.
    #[arg(long)]
    lineage: String,
    /// The hand-off directory the examiner reads candidates from; absent, no
    /// candidate is handed over (a pilot).
    #[arg(long)]
    handoff: Option<PathBuf>,
    /// Where a generation's champion is kept for the examiner (with
    /// `--handoff`).
    #[arg(long)]
    kept_adapters: Option<PathBuf>,
    /// A page of the Lua 5.1 manual the policy proposes tasks from (repeat).
    #[arg(long = "manual", required = true)]
    manuals: Vec<PathBuf>,
    /// The Lua 5.1 interpreter.
    #[arg(long)]
    lua: PathBuf,
    /// The seed training seeds are derived from; drawn and printed when
    /// absent.
    #[arg(long)]
    seed: Option<u64>,
    /// The seed of the generator's dev and retention programs, the same for
    /// every lineage; the lineage seed when absent.
    #[arg(long)]
    generator_seed: Option<u64>,
    /// Generations in the lineage.
    #[arg(long, default_value_t = 3)]
    generations: u32,
    /// Rounds of each generation; the last accepted candidate is its
    /// champion.
    #[arg(long)]
    rounds: u32,
    /// Generations to run in this process.
    #[arg(long, default_value_t = 1)]
    run_generations: u32,
    /// Generator programs the dev gate is decided on (beside the proposals).
    #[arg(long)]
    dev_generator: usize,
    /// Generator programs of levels 1 and 2 kept for retention.
    #[arg(long)]
    keep_generator: usize,
    /// Proposals of the base held out for the dev gate.
    #[arg(long)]
    dev_proposals: usize,
    /// Proposals of the base held out for retention.
    #[arg(long)]
    keep_proposals: usize,
    /// Programs each round's policy is asked to propose to train on.
    #[arg(long)]
    proposals_per_round: usize,
    /// The most requests a round of proposing makes.
    #[arg(long)]
    proposal_calls: usize,
    /// The most requests the base makes proposing the held-out dev and
    /// retention programs.
    #[arg(long)]
    held_out_calls: usize,
    /// Requests of a round of proposing asked at once.
    #[arg(long, default_value_t = 8)]
    proposal_parallel: usize,
    /// Tokens a prediction may take.
    #[arg(long, default_value_t = 256)]
    max_tokens: u32,
    /// The lead (share of dev programs won) a round is meant to produce.
    #[arg(long)]
    target_effect: f64,
    /// Expected share of dev programs a candidate and its champion differ on.
    #[arg(long)]
    discordance: f64,
    /// Retention loss, as a share of the retention set, a candidate may show.
    #[arg(long)]
    retention_margin: f64,
    /// What each generation's candidate is trained from: the previous
    /// champion's adapter, continued, or a fresh adapter on the base trained
    /// on every accepted generation's records so far.
    #[arg(long, value_enum, default_value_t = TrainFrom::Previous)]
    train_from: TrainFrom,
    /// With `--train-from base-cumulative`: passes over the union a
    /// candidate's training makes; the steps are the larger of `--steps` and
    /// this many passes' worth, rounded up.
    #[arg(long)]
    cumulative_epochs: Option<f32>,
    #[command(flatten)]
    frontier: FrontierArgs,
    #[command(flatten)]
    evidence: EvidenceArgs,
    /// Where the Splinter state of the proposing handle lives (task records,
    /// a file each): disposable, since the proposals themselves are kept in
    /// the lineage's directory; a memory-backed directory spares the disk's
    /// file quota. Default: under `--out`.
    #[arg(long)]
    proposer_state: Option<PathBuf>,
    /// Files the disks must have free before a round.
    #[arg(long, default_value_t = 20_000)]
    min_free_inodes: u64,
    /// Splinter's commit, as the launch record states it.
    #[arg(long)]
    splinter_commit: Option<String>,
    /// Sven's commit.
    #[arg(long)]
    sven_commit: Option<String>,
    /// Brain's commit.
    #[arg(long)]
    brain_commit: Option<String>,
}

fn hand_over(
    cli: &Cli,
    lineage: &Lineage<'_>,
    handoff_dir: &Path,
    generation: u32,
    base_digest: &str,
) -> Result<()> {
    let rounds = lineage.rounds()?;
    let champion = champion_of(&rounds);
    let kept_dir = cli
        .kept_adapters
        .as_deref()
        .context("--handoff needs --kept-adapters")?;
    let (adapter, digest) = match &champion.adapter {
        Some(path) => {
            let (kept, digest) = handoff::keep_adapter(kept_dir, &cli.lineage, generation, path)?;
            (Some(kept), Some(digest))
        }
        None => (None, None),
    };
    let advanced = rounds
        .iter()
        .any(|r| r.generation == generation && r.advanced);
    let verdicts: Vec<String> = rounds
        .iter()
        .filter(|r| r.generation == generation)
        .map(|r| {
            r.decision.as_ref().map_or_else(
                || r.note.clone(),
                |d| format!("{:?}: {}", d.action, d.reason),
            )
        })
        .collect();
    let candidate = Candidate {
        series: cli.lineage.clone(),
        step: generation,
        base: cli.base_name.clone(),
        base_digest: base_digest.to_string(),
        serving: handoff::serving(&cli.model, adapter.as_deref()),
        adapter,
        adapter_digest: digest,
        advanced,
        arm_verdict: Some(verdicts.join(" | ")),
        build: Build {
            splinter_commit: cli.splinter_commit.clone().unwrap_or_default(),
            sven_commit: cli.sven_commit.clone().unwrap_or_default(),
            brain_commit: cli.brain_commit.clone().unwrap_or_default(),
            binary_digest: splinter_sdk::vocabulary::build::running_binary_digest()
                .map(|d| d.to_string()),
        },
    };
    let tasks = lineage.family.records(&trained_up_to(&rounds, generation));
    let written = handoff::write(handoff_dir, &candidate, &tasks)?;
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

fn main() -> Result<()> {
    let cli = Cli::parse();
    cli.recipe.validate()?;
    let seed = splinter_sdk::seed::resolve("seed", cli.seed);
    let generator_seed = cli.generator_seed.unwrap_or(seed);
    eprintln!(
        "lineage {}: seed {seed}, generator seed {generator_seed}",
        cli.lineage
    );
    for dir in [
        Some(&cli.out),
        cli.handoff.as_ref(),
        cli.kept_adapters.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    splinter_sdk::preflight::check(&[splinter_sdk::preflight::Requirement {
        path: cli.out.clone(),
        min_free_bytes: 5 << 30,
        min_free_inodes: cli.min_free_inodes,
    }])
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let scratch = cli.out.join("lua-programs");
    std::fs::create_dir_all(&scratch)?;
    let generator = LuaPredict::generate(
        &cli.lua,
        &scratch,
        Sizes {
            pool: 0,
            dev: cli.dev_generator,
            retention: cli.keep_generator,
        },
        generator_seed,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let frontier = cli.frontier.settings()?;
    let files = Files::at(&cli.out);
    let family =
        ProposalFamily::open(generator, &files.proposals()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let lua_dir = cli
        .lua
        .parent()
        .map_or_else(|| Path::new(".").to_path_buf(), Path::to_path_buf);
    let host = ServedHost::new(HostSettings {
        brain: cli.brain.clone(),
        models: cli.models_dir.clone(),
        base: cli.model.clone(),
        work: cli.out.join("server-work"),
        state: cli
            .proposer_state
            .clone()
            .unwrap_or_else(|| cli.out.join("proposer-state")),
        store: cli.out.join("artifacts"),
        adapters: cli.out.join("served-adapters"),
        log: cli.out.join("brain-serve.log"),
        device: cli.server_device.clone(),
        kv_cache: Some(cli.kv_cache.clone()),
        slots: cli.slots,
        runtime_dirs: vec![lua_dir],
        predict_temperature: frontier.map_or(1.0, |f| f.temperature),
    });
    let proposer = Proposer::new(&cli.model, &cli.manuals);
    let mut model = ModelSettings::new(&cli.model);
    model.max_tokens = cli.max_tokens;
    model.concurrency = cli.slots as usize;
    let mut validator =
        ServedValidator::new(&family, &host, model).map_err(|e| anyhow::anyhow!("{e}"))?;
    let plan = Plan {
        generations: cli.generations,
        rounds: cli.rounds,
        proposals_per_round: cli.proposals_per_round,
        budget: (cli.proposal_calls, cli.proposal_parallel),
        target_effect: cli.target_effect,
        discordance: cli.discordance,
        retention_margin: cli.retention_margin,
        train_from: cli.train_from,
        frontier,
        epochs: cli.cumulative_epochs,
        acceptance: cli.evidence.acceptance(cli.slots)?,
    };
    plan.validate(&cli.recipe)?;
    let base_digest = handoff::base_digest(&cli.model)?;
    let mut lineage = Lineage {
        files,
        family: &family,
        host: &host,
        validator: &mut validator,
        proposer: &proposer,
        base: &cli.model,
        recipe: &cli.recipe,
        plan: &plan,
        gate: DevGate {
            max_retention_loss_rate: cli.retention_margin,
            ..DevGate::default()
        },
        seed,
    };
    let (dev_units, retention_units) = (
        cli.dev_generator + cli.dev_proposals,
        cli.keep_generator + cli.keep_proposals,
    );
    let planned = match plan.acceptance {
        Acceptance::Greedy => {
            let (planned, needed) = lineage.check_power(dev_units, retention_units)?;
            json!({"plan": planned, "retention_units_needed": needed})
        }
        Acceptance::Evidence(evidence) => {
            json!({"evidence": lineage.check_evidence(&evidence, dev_units, retention_units)?})
        }
    };
    eprintln!("{planned}");
    lineage.open_held_out(
        cli.dev_proposals,
        cli.keep_proposals,
        Budget {
            want: 0,
            max_calls: cli.held_out_calls,
            parallel: cli.proposal_parallel,
        },
    )?;
    // The base's own answers: the first champion's dev result. The evidence
    // gate grades only what its rounds draw.
    if plan.acceptance == Acceptance::Greedy {
        lineage.dev_of(&PolicyRef::default())?;
    }
    let done = lineage.rounds()?;
    let first = (1..=cli.generations)
        .find(|g| {
            (1..=cli.rounds).any(|r| !done.iter().any(|d| d.generation == *g && d.round == r))
        })
        .unwrap_or(cli.generations + 1);
    for generation in first..first + cli.run_generations {
        if generation > cli.generations {
            break;
        }
        for round in 1..=cli.rounds {
            let record = lineage.run_round(generation, round)?;
            println!("{}", serde_json::to_string(&record)?);
        }
        if let Some(dir) = &cli.handoff {
            hand_over(&cli, &lineage, dir, generation, &base_digest)?;
        }
    }
    host.stop();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(gate: GateKind) -> EvidenceArgs {
        EvidenceArgs {
            gate,
            gate_eps: None,
            gate_q: None,
            gate_beta: None,
            dev_draw: None,
            dev_batch: None,
        }
    }

    #[test]
    fn the_greedy_gate_is_the_default_and_refuses_evidence_settings() {
        assert_eq!(
            args(GateKind::Greedy).acceptance(32).unwrap(),
            Acceptance::Greedy
        );
        let mut stray = args(GateKind::Greedy);
        stray.gate_q = Some(0.3);
        assert!(stray.acceptance(32).is_err(), "a setting no gate reads");
    }

    #[test]
    fn the_evidence_gate_needs_every_setting_and_batches_by_the_slots() {
        let mut evidence = args(GateKind::Evidence);
        assert!(evidence.acceptance(32).is_err());
        evidence.gate_eps = Some(0.02);
        evidence.gate_q = Some(0.3);
        evidence.gate_beta = Some(0.01);
        evidence.dev_draw = Some(150);
        let Acceptance::Evidence(plan) = evidence.acceptance(32).unwrap() else {
            panic!("an evidence plan")
        };
        assert_eq!((plan.draw, plan.batch), (150, 32));
        assert!((plan.gate.q - 0.3).abs() < 1e-12);
        evidence.gate_beta = Some(0.9);
        assert!(
            evidence.acceptance(32).is_err(),
            "beta above 1 - (1 - eps) / 2"
        );
    }
}
