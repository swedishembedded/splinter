// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measured validation of continual learning
// from a user's own agent sessions for its clients. If your team needs
// expertise in proving that a model absorbed what its user taught it without
// forgetting what it knew, you can procure our services by sending an email
// to info@swedishembedded.com.

//! The command line of the cycle: `cycle plan`, `cycle run`, `cycle report`.

use std::path::PathBuf;

use anyhow::Context as _;
use clap::{Args, Subcommand};
use splinter_sdk::vocabulary::clock::{Clock, SystemClock};

use super::driver::{self, Repo, Request};
use super::plan::{generate, CanaryNights, CycleRole, Plan, PlanRequest};
use super::prereg::AbsorbSettings;
use super::world::SdkWorld;
use crate::facts::Manifest;
use crate::session::Style;

/// The anchor suite a cycle is held to, relative to the repository root.
const DEFAULT_ANCHOR: &str = "samples/jefferson/anchor.jsonl";

/// The multi-night cycle.
#[derive(Subcommand)]
pub enum Cycle {
    /// Write the plan of a cycle: which sessions are recorded on which night.
    Plan(PlanArgs),
    /// Run the nights of a cycle back to back, then report. A rerun resumes.
    Run(Box<RunArgs>),
    /// Build the report of a cycle from what its directory holds.
    Report(ReportArgs),
}

/// `cycle plan`.
#[derive(Args)]
pub struct PlanArgs {
    /// The fact pool's directory (its manifest).
    #[arg(long)]
    manifest: PathBuf,
    /// Which facts the cycle teaches.
    #[arg(long, value_enum)]
    role: CycleRole,
    /// The nights, null, sham and canary nights included.
    #[arg(long)]
    nights: usize,
    /// Facts taught per night in a development cycle.
    #[arg(long, default_value_t = 5)]
    per_night: usize,
    /// Noise sessions on every night.
    #[arg(long, default_value_t = 2)]
    noise: usize,
    /// The styles of the users, given to the facts of a day in turn.
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_value = "plain,realistic"
    )]
    styles: Vec<Style>,
    /// A night of noise sessions only.
    #[arg(long)]
    null_night: Option<usize>,
    /// A night on which the next day's facts are asked about and never corrected.
    #[arg(long)]
    sham_night: Option<usize>,
    /// The canary fact: its false statement is asserted on `--canary-night`.
    #[arg(long, requires_all = ["canary_night", "canary_back"])]
    canary: Option<String>,
    /// The night the false statement is asserted.
    #[arg(long)]
    canary_night: Option<usize>,
    /// The night the true statement is asserted again.
    #[arg(long)]
    canary_back: Option<usize>,
    /// Where to write the plan.
    #[arg(long)]
    out: PathBuf,
}

/// `cycle run`.
#[derive(Args)]
pub struct RunArgs {
    /// The fact pool's directory: manifest, sealed probes, screening.
    #[arg(long)]
    manifest: PathBuf,
    /// The plan file (`cycle plan`).
    #[arg(long)]
    plan: PathBuf,
    /// The nights to run.
    #[arg(long)]
    days: usize,
    /// The cycle's directory.
    #[arg(long)]
    out: PathBuf,
    /// The arm of the protocol: `main`, or an ablation (`no-rehearsal`,
    /// `continue-from-release`, `no-replay`; the last two on development facts only).
    #[arg(long, default_value = "main")]
    arm: String,
    /// The anchor suite.
    #[arg(long, default_value = DEFAULT_ANCHOR)]
    anchor: PathBuf,
    /// The model store, when not the configuration's.
    #[arg(long)]
    models: Option<PathBuf>,
    /// The repository the pipeline's commit is read from.
    #[arg(long, default_value = ".")]
    repo: PathBuf,
    #[command(flatten)]
    absorb: AbsorbArgs,
}

/// The flags of `splinter absorb` every night runs with; the defaults are the
/// registered ones.
#[derive(Args)]
pub struct AbsorbArgs {
    /// The policy: the base and its context limit.
    #[arg(long, default_value = AbsorbSettings::DEFAULT_POLICY)]
    policy: String,
    /// The model that plays the user.
    #[arg(long, default_value = AbsorbSettings::DEFAULT_SIMULATOR)]
    simulator: String,
    /// The judge role, which also grades the evaluations.
    #[arg(long, default_value = AbsorbSettings::DEFAULT_JUDGE)]
    judge: String,
    /// The generator role (the policy by default).
    #[arg(long)]
    generator: Option<String>,
    /// The teacher role (the policy by default).
    #[arg(long)]
    teacher: Option<String>,
    /// Differently worded questions written per claim.
    #[arg(long)]
    paraphrases: Option<usize>,
    /// The share of the training draws that are the base's own answers.
    #[arg(long)]
    rehearsal_share: Option<f64>,
    /// The most passes over the night's records.
    #[arg(long)]
    epochs: Option<u32>,
    /// A step budget in place of the epochs.
    #[arg(long)]
    steps: Option<u32>,
    /// LoRA rank.
    #[arg(long)]
    rank: Option<u32>,
    /// Hold the base at bf16 while training.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    bf16_base: bool,
    /// Extraction passes per session.
    #[arg(long)]
    passes: Option<u32>,
    /// The judge checks that the person's words assert each claim.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    entail: bool,
}

/// `cycle report`.
#[derive(Args)]
pub struct ReportArgs {
    /// The cycle's directory.
    #[arg(long)]
    dir: PathBuf,
    /// The fact pool's directory.
    #[arg(long)]
    manifest: PathBuf,
}

impl AbsorbArgs {
    fn settings(&self) -> AbsorbSettings {
        let default = AbsorbSettings::default();
        AbsorbSettings {
            policy: self.policy.clone(),
            simulator: self.simulator.clone(),
            judge: self.judge.clone(),
            generator: self.generator.clone(),
            teacher: self.teacher.clone(),
            paraphrases: self.paraphrases.unwrap_or(default.paraphrases),
            rehearsal_share: self.rehearsal_share.unwrap_or(default.rehearsal_share),
            epochs: self.epochs.unwrap_or(default.epochs),
            steps: self.steps,
            rank: self.rank.unwrap_or(default.rank),
            bf16_base: self.bf16_base,
            extraction_passes: self.passes.unwrap_or(default.extraction_passes),
            entail: self.entail,
            ..default
        }
    }
}

/// The settings of `arm` over `base`.
///
/// # Errors
/// The arm is unknown, or an ablation is asked of the test facts.
pub fn arm_settings(
    arm: &str,
    role: CycleRole,
    base: AbsorbSettings,
) -> anyhow::Result<AbsorbSettings> {
    let dev_only = || {
        anyhow::ensure!(
            role == CycleRole::Dev,
            "the arm {arm} is a development ablation: it is not run on the test facts"
        );
        Ok(())
    };
    Ok(match arm {
        "main" => base,
        "no-rehearsal" => AbsorbSettings {
            rehearsal_share: 0.0,
            ..base
        },
        "continue-from-release" => {
            dev_only()?;
            AbsorbSettings {
                continue_from_release: true,
                ..base
            }
        }
        "no-replay" => {
            dev_only()?;
            AbsorbSettings {
                continue_from_release: true,
                replay_fraction: 0.0,
                ..base
            }
        }
        other => anyhow::bail!(
            "no arm {other:?}: main, no-rehearsal, continue-from-release or no-replay"
        ),
    })
}

fn git(repo: &std::path::Path, args: &[&str]) -> anyhow::Result<String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .output()
        .with_context(|| format!("running git {args:?}"))?;
    anyhow::ensure!(
        output.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// The commit of `repo` and whether its tree is clean (local git only).
fn repo_state(repo: &std::path::Path) -> anyhow::Result<Repo> {
    Ok(Repo {
        commit: git(repo, &["rev-parse", "HEAD"])?,
        dirty: !git(repo, &["status", "--porcelain", "--untracked-files=no"])?.is_empty(),
    })
}

/// Runs a `cycle` command.
///
/// # Errors
/// Whatever the command fails with.
pub fn run(command: Cycle) -> anyhow::Result<()> {
    match command {
        Cycle::Plan(a) => {
            let manifest = Manifest::read(&a.manifest)?;
            let plan = generate(&PlanRequest {
                manifest: &manifest,
                role: a.role,
                nights: a.nights,
                per_night: a.per_night,
                noise_per_night: a.noise,
                styles: a.styles,
                null_night: a.null_night,
                sham_night: a.sham_night,
                canary: match (a.canary, a.canary_night, a.canary_back) {
                    (Some(fact), Some(night), Some(back)) => {
                        Some(CanaryNights { fact, night, back })
                    }
                    _ => None,
                },
            })?;
            std::fs::write(&a.out, serde_json::to_string_pretty(&plan)? + "\n")
                .with_context(|| format!("writing {}", a.out.display()))?;
            println!("{}: {} nights", a.out.display(), plan.nights.len());
        }
        Cycle::Run(a) => {
            let manifest = Manifest::read(&a.manifest)?;
            let plan: Plan = Plan::read(&a.plan, &manifest)?;
            let settings = arm_settings(&a.arm, plan.role, a.absorb.settings())?;
            let world = SdkWorld::open(
                &a.out.join(crate::runtime::STATE_DIR),
                &a.manifest,
                a.models.clone(),
                settings.clone(),
            )?;
            let report = driver::run(
                &Request {
                    dir: a.out.clone(),
                    manifest_dir: a.manifest,
                    plan: a.plan,
                    anchor: a.anchor,
                    days: a.days,
                    arm: a.arm,
                    settings,
                    repo: repo_state(&a.repo)?,
                    now: SystemClock.utc_now(),
                },
                &world,
            )?;
            println!("{}", report.verdict.text);
        }
        Cycle::Report(a) => {
            let report = driver::make_report(&a.dir, &a.manifest)?;
            println!("{}", report.verdict.text);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_arm_changes_only_what_it_names_and_the_ablations_that_continue_are_for_development_facts()
    {
        let base = AbsorbSettings::default();
        let none = arm_settings("no-rehearsal", CycleRole::Test, base.clone()).expect("an arm");
        assert_eq!(none.rehearsal_share, 0.0);
        assert_eq!(
            AbsorbSettings {
                rehearsal_share: base.rehearsal_share,
                ..none
            },
            base
        );
        assert_eq!(
            arm_settings("main", CycleRole::Test, base.clone()).expect("main"),
            base
        );
        let no_replay = arm_settings("no-replay", CycleRole::Dev, base.clone()).expect("an arm");
        assert!(no_replay.continue_from_release && no_replay.replay_fraction == 0.0);
        let error = format!(
            "{:#}",
            arm_settings("continue-from-release", CycleRole::Test, base.clone())
                .expect_err("refused")
        );
        assert!(error.contains("development ablation"), "{error}");
        assert!(arm_settings("sham", CycleRole::Dev, base).is_err());
    }
}
