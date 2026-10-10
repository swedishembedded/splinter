// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measured validation of continual learning
// from a user's own agent sessions for its clients. If your team needs
// expertise in proving that a model absorbed what its user taught it without
// forgetting what it knew, you can procure our services by sending an email
// to info@swedishembedded.com.

//! The nights of a cycle, back to back: record the day's sessions with the
//! release in effect, absorb them, evaluate the release in effect after the
//! night. Every step of every night leaves a file, and a rerun skips what is
//! there, so a cycle interrupted anywhere resumes where it stopped.
//!
//! ```text
//! DIR/prereg.json  plan.json  items.json     registered before the first session
//! DIR/state/                                  Splinter's state: releases, the claim ledger
//! DIR/nights/night-NN/record.json             sessions recorded (night 0 has none)
//!                     absorb.json             the nightly update's report
//!                     evaluation.json         the night is finished
//! DIR/report.json  report.md
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Context as _;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use super::analysis::FactInfo;
use super::eval::{evaluate, Evaluation, LogProb, Models};
use super::items::{build, Item, Sources};
use super::plan::{NightPlan, Plan};
use super::prereg::{self, analysis_sha256, AbsorbSettings, Entry, Prereg};
use super::report::{self, NightMeta, Report, ReportSources};
use super::verdict::Rule;
use crate::facts::Manifest;
use crate::probes::sha256_hex;
use crate::seal::verified_probes;

/// What recording a night's sessions made.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecordedNight {
    /// Sessions recorded.
    pub sessions: usize,
    /// Seconds it took.
    pub seconds: f64,
}

/// What a night's update found.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AbsorbedNight {
    /// The release the update made, when the gate opened it.
    pub released: Option<String>,
    /// Claims live in the ledger afterwards.
    pub live_claims: Option<u64>,
    /// Why the update stopped before its last stage, if it did.
    pub stopped: Option<String>,
    /// Seconds it took.
    pub seconds: f64,
    /// The whole report of `absorb`.
    pub report: serde_json::Value,
}

/// Everything a night needs from the models and Splinter's state.
pub trait World: Models {
    /// Records the sessions of `night` into `dir` with the release in effect.
    ///
    /// # Errors
    /// A session cannot be recorded.
    fn record(&self, night: &NightPlan, dir: &Path) -> anyhow::Result<RecordedNight>;

    /// Absorbs the sessions recorded in `dir`.
    ///
    /// # Errors
    /// The update fails; a candidate the gate refuses is no failure.
    fn absorb(&self, night: &NightPlan, dir: &Path) -> anyhow::Result<AbsorbedNight>;

    /// The release in effect now; absent before the first.
    ///
    /// # Errors
    /// The state cannot be read.
    fn release_in_effect(&self) -> anyhow::Result<Option<String>>;

    /// The mean log-probability of the fixed text, against the base; absent
    /// where the model layer offers no scoring call.
    ///
    /// # Errors
    /// Scoring was possible and failed.
    fn log_prob(&self) -> anyhow::Result<Option<LogProb>> {
        Ok(None)
    }
}

/// The repository state the run is registered at.
#[derive(Clone, Debug)]
pub struct Repo {
    /// The commit of the pipeline.
    pub commit: String,
    /// Whether the tree has other changes.
    pub dirty: bool,
}

/// What a cycle is asked to do.
pub struct Request {
    /// The cycle's directory.
    pub dir: PathBuf,
    /// The directory of the fact pool: manifest, sealed probes, screening.
    pub manifest_dir: PathBuf,
    /// The plan file.
    pub plan: PathBuf,
    /// The anchor suite.
    pub anchor: PathBuf,
    /// The nights to run.
    pub days: usize,
    /// The arm of the protocol: `main` or an ablation.
    pub arm: String,
    /// `absorb` as every night runs it.
    pub settings: AbsorbSettings,
    /// The pipeline's commit.
    pub repo: Repo,
    /// Now, as the registration records it.
    pub now: String,
}

fn write_json(path: &Path, value: &impl Serialize) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let staging = path.with_extension("tmp");
    std::fs::write(&staging, serde_json::to_string_pretty(value)? + "\n")
        .with_context(|| format!("writing {}", staging.display()))?;
    std::fs::rename(&staging, path).with_context(|| format!("replacing {}", path.display()))
}

fn read_json<T: DeserializeOwned>(path: &Path) -> anyhow::Result<T> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("{} is not valid", path.display()))
}

fn sha256_file(path: &Path) -> anyhow::Result<String> {
    Ok(sha256_hex(
        &std::fs::read(path).with_context(|| format!("reading {}", path.display()))?,
    ))
}

/// A night's directory.
#[must_use]
pub fn night_dir(dir: &Path, night: usize) -> PathBuf {
    dir.join("nights").join(format!("night-{night:02}"))
}

/// The day-0 wrong claim of each fact, from the screening record.
fn wrong_claims(manifest_dir: &Path) -> anyhow::Result<BTreeMap<String, Vec<String>>> {
    #[derive(Deserialize)]
    struct Screening {
        facts: Vec<Screened>,
    }
    #[derive(Deserialize)]
    struct Screened {
        id: String,
        #[serde(default)]
        wrong_claim: Vec<String>,
    }
    let screening: Screening = read_json(&manifest_dir.join("screening.json"))?;
    Ok(screening
        .facts
        .into_iter()
        .map(|f| (f.id, f.wrong_claim))
        .collect())
}

fn fact_infos(manifest: &Manifest) -> Vec<FactInfo> {
    manifest
        .facts
        .iter()
        .filter_map(|f| {
            Some(FactInfo {
                id: f.id.clone(),
                role: f.role?,
                canary_keys: f.canary.as_ref().map_or(Vec::new(), |c| c.keys.clone()),
            })
        })
        .collect()
}

/// The registered inputs, as they stand now.
struct Inputs {
    manifest: Manifest,
    plan: Plan,
    plan_bytes: Vec<u8>,
    items: Vec<Item>,
    items_bytes: Vec<u8>,
    anchor_sha256: String,
}

fn load(request: &Request) -> anyhow::Result<Inputs> {
    let manifest = Manifest::read(&request.manifest_dir)?;
    let probes = verified_probes(&request.manifest_dir, &manifest)?;
    let plan_bytes = std::fs::read(&request.plan)
        .with_context(|| format!("reading {}", request.plan.display()))?;
    let plan = Plan::read(&request.plan, &manifest)?.first(request.days)?;
    let anchor = std::fs::read_to_string(&request.anchor)
        .with_context(|| format!("reading the anchor suite {}", request.anchor.display()))?;
    let items = build(&Sources {
        manifest: &manifest,
        probes: &probes,
        wrong_claims: &wrong_claims(&request.manifest_dir)?,
        anchor: &anchor,
        role: plan.role,
    })?;
    Ok(Inputs {
        items_bytes: serde_json::to_vec_pretty(&items)?,
        plan_bytes,
        anchor_sha256: sha256_hex(anchor.as_bytes()),
        manifest,
        plan,
        items,
    })
}

/// Registers the run, or checks that the registration under `request.dir` is
/// of the inputs now given.
fn register(request: &Request, inputs: &Inputs) -> anyhow::Result<Prereg> {
    let path = request.dir.join(prereg::FILE);
    let current = |run_counter: usize| -> anyhow::Result<Prereg> {
        Prereg {
            schema: prereg::SCHEMA.into(),
            run_counter,
            pipeline_commit: request.repo.commit.clone(),
            dirty: request.repo.dirty,
            role: inputs.plan.role,
            days: request.days,
            arm: request.arm.clone(),
            absorb: request.settings.clone(),
            probes_sha256: inputs
                .manifest
                .probes
                .as_ref()
                .map(|p| p.sha256.clone())
                .context("the probes are not sealed")?,
            manifest_sha256: sha256_file(&request.manifest_dir.join(crate::facts::MANIFEST))?,
            plan_sha256: sha256_hex(&inputs.plan_bytes),
            items_sha256: sha256_hex(&inputs.items_bytes),
            anchor_sha256: inputs.anchor_sha256.clone(),
            analysis_sha256: analysis_sha256(),
            rule: Rule::protocol(),
            registered_at: request.now.clone(),
            hash: String::new(),
        }
        .sealed()
    };
    if path.exists() {
        let registered = Prereg::read(&request.dir)?;
        registered.verify()?;
        let now = current(registered.run_counter)?;
        // Where the run was registered says when, and at which commit it began.
        let same = Prereg {
            registered_at: registered.registered_at.clone(),
            pipeline_commit: registered.pipeline_commit.clone(),
            dirty: registered.dirty,
            ..now
        }
        .sealed()?;
        anyhow::ensure!(
            same.hash == registered.hash,
            "{} was registered with other inputs than these: start a new run in a new directory \
             (a registered run is never changed)",
            request.dir.display()
        );
        return Ok(registered);
    }
    anyhow::ensure!(
        !(inputs.plan.role == super::plan::CycleRole::Test && request.repo.dirty),
        "a run on the test facts starts from a clean tree: commit first"
    );
    std::fs::create_dir_all(&request.dir)?;
    let counter = prereg::next_run(&request.manifest_dir)?;
    prereg::record(
        &request.manifest_dir,
        &Entry::Start {
            run: counter,
            dir: request.dir.display().to_string(),
        },
    )?;
    std::fs::write(request.dir.join("plan.json"), &inputs.plan_bytes)?;
    std::fs::write(request.dir.join("items.json"), &inputs.items_bytes)?;
    let registered = current(counter)?;
    registered.write(&request.dir)?;
    prereg::record(
        &request.manifest_dir,
        &Entry::Registered {
            run: counter,
            hash: registered.hash.clone(),
        },
    )?;
    Ok(registered)
}

fn evaluate_night(
    world: &dyn World,
    items: &[Item],
    dir: &Path,
    night: usize,
    prereg_hash: &str,
) -> anyhow::Result<()> {
    let path = night_dir(dir, night).join("evaluation.json");
    if path.exists() {
        return Ok(());
    }
    let mut evaluation = evaluate(items, world, night, world.release_in_effect()?, prereg_hash)?;
    evaluation.logprob = world.log_prob()?;
    write_json(&path, &evaluation)
}

/// Runs the nights `request` asks for that are not finished, then writes the
/// report.
///
/// # Errors
/// A registration holds other inputs, a model or the update fails (the run
/// resumes from there when run again), or the report is refused.
pub fn run(request: &Request, world: &dyn World) -> anyhow::Result<Report> {
    let inputs = load(request)?;
    let registered = register(request, &inputs)?;
    evaluate_night(world, &inputs.items, &request.dir, 0, &registered.hash)?;
    for night in inputs.plan.nights.iter().take(request.days) {
        let ndir = night_dir(&request.dir, night.night);
        if ndir.join("evaluation.json").exists() {
            continue;
        }
        if !ndir.join("record.json").exists() {
            // A recording that did not finish is made again whole.
            let _ = std::fs::remove_dir_all(ndir.join("sessions"));
            let _ = std::fs::remove_file(ndir.join("subjects.jsonl"));
            let recorded = world
                .record(night, &ndir)
                .with_context(|| format!("night {}: recording the sessions", night.night))?;
            write_json(&ndir.join("record.json"), &recorded)?;
        }
        if !ndir.join("absorb.json").exists() {
            let absorbed = world
                .absorb(night, &ndir)
                .with_context(|| format!("night {}: the nightly update", night.night))?;
            write_json(&ndir.join("absorb.json"), &absorbed)?;
        }
        evaluate_night(
            world,
            &inputs.items,
            &request.dir,
            night.night,
            &registered.hash,
        )
        .with_context(|| format!("night {}: the evaluation", night.night))?;
    }
    let report = make_report(&request.dir, &request.manifest_dir)?;
    if let Some(Entry::Registered { run, .. }) = prereg::read_ledger(&request.manifest_dir)?
        .into_iter()
        .rev()
        .find(|e| matches!(e, Entry::Registered { hash, .. } if *hash == registered.hash))
    {
        prereg::record(
            &request.manifest_dir,
            &Entry::Finished {
                run,
                met: report.verdict.met,
                text: report.verdict.text.clone(),
            },
        )?;
    }
    Ok(report)
}

/// Builds and writes the report of the cycle under `dir`, from what is
/// there.
///
/// # Errors
/// The pre-registration or the files it registered were changed, an
/// evaluation was made under another, or no night was evaluated.
pub fn make_report(dir: &Path, manifest_dir: &Path) -> anyhow::Result<Report> {
    let registered = Prereg::read(dir)?;
    let manifest = Manifest::read(manifest_dir)?;
    for (name, expected) in [
        ("plan.json", &registered.plan_sha256),
        ("items.json", &registered.items_sha256),
    ] {
        let found = sha256_file(&dir.join(name))?;
        anyhow::ensure!(
            found == *expected,
            "refusing a report: {name} hashes to {found}, not the {expected} registered"
        );
    }
    let found = sha256_file(&manifest_dir.join(crate::facts::MANIFEST))?;
    anyhow::ensure!(
        found == registered.manifest_sha256,
        "refusing a report: the manifest hashes to {found}, not the {} registered",
        registered.manifest_sha256
    );
    let plan: Plan = read_json(&dir.join("plan.json"))?;
    let items: Vec<Item> = read_json(&dir.join("items.json"))?;
    let mut evaluations: Vec<Evaluation> = Vec::new();
    let mut meta = Vec::new();
    for night in 0..=registered.days {
        let ndir = night_dir(dir, night);
        let path = ndir.join("evaluation.json");
        if !path.exists() {
            break;
        }
        evaluations.push(read_json(&path)?);
        if night > 0 {
            let recorded: Option<RecordedNight> = ndir
                .join("record.json")
                .exists()
                .then(|| read_json(&ndir.join("record.json")))
                .transpose()?;
            let absorbed: Option<AbsorbedNight> = ndir
                .join("absorb.json")
                .exists()
                .then(|| read_json(&ndir.join("absorb.json")))
                .transpose()?;
            meta.push(NightMeta {
                night,
                sessions: recorded.as_ref().map_or(0, |r| r.sessions),
                released: absorbed.as_ref().map(|a| a.released.is_some()),
                live_claims: absorbed.as_ref().and_then(|a| a.live_claims),
                stopped: absorbed.as_ref().and_then(|a| a.stopped.clone()),
                record_seconds: recorded.map(|r| r.seconds),
                absorb_seconds: absorbed.map(|a| a.seconds),
            });
        }
    }
    let runs = prereg::read_ledger(manifest_dir)?;
    let report = report::build(&ReportSources {
        prereg: &registered,
        plan: &plan,
        items: &items,
        facts: &fact_infos(&manifest),
        evaluations: &evaluations,
        meta: &meta,
        runs: &runs,
    })?;
    write_json(&dir.join("report.json"), &report)?;
    std::fs::write(dir.join("report.md"), report::markdown(&report))?;
    Ok(report)
}
