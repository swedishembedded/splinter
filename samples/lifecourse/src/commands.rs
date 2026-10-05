// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The subcommands: build, freeze, cross-validate, score the locked test,
//! compare.
//!
//! Every run writes its own file under `runs/` (arms can run at the same
//! time), naming the dataset and partition digests it used, so a result
//! cannot be separated from what produced it.

use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use splinter_sdk::data::frozen::Ledger;
use splinter_sdk::data::partition::{partition, units_from_json_lines, Partition, PartitionSpec};
use splinter_sdk::model::timeline::survival::compare::corrected_resampled_t;
use splinter_sdk::model::timeline::{read_jsonl, Subject};
use splinter_sdk::vocabulary::digest::Digest;
use splinter_sdk::vocabulary::terms::{Terms, Use};

use crate::build::{subject, CycleCounts, Design, CODES};
use crate::experiment::{fit, Arm, RunInfo, Training, STEPS};
use crate::metrics::{evaluate, ibs_terms, Horizons, Metrics};
use crate::nhanes::{mortality, Cycle, CYCLES};

const TIMELINES: &str = "timelines.jsonl";
const DESIGN: &str = "design.jsonl";
const BUILD: &str = "build.json";
const PARTITION: &str = "partition.json";
const CRITERIA: &str = "criteria.json";
const LEDGER: &str = "FROZEN.json";

/// What `build` records beside the subjects.
#[derive(Serialize)]
struct BuildReport {
    cycles: BTreeMap<u16, CycleCounts>,
    subjects: usize,
    deaths: usize,
    horizons: Horizons,
    /// `(file, digest)` of every input read.
    sources: Vec<(String, String)>,
    terms: Terms,
}

/// The terms of the NHANES public-use files and the public-use linked
/// mortality file.
pub fn nhanes_terms() -> Terms {
    let mut t = Terms::public_domain("NHANES public-use data (US federal public domain)");
    t.conditions = vec![
        "NCHS data use restrictions: no attempt to identify any participant".into(),
        "the public-use mortality file perturbs follow-up and cause of death for some records"
            .into(),
    ];
    t
}

/// NHANES files and the linkage as timeline-v1 subjects.
pub fn build(nhanes: &Path, mortality_dir: &Path, out: &Path) -> Result<()> {
    std::fs::create_dir_all(out)?;
    let mut lines = String::new();
    let mut design = String::new();
    let mut cycles = BTreeMap::new();
    let mut sources = Vec::new();
    let mut all: Vec<Subject> = Vec::new();
    for (start, suffix) in CYCLES {
        let c = Cycle::load(nhanes, start, suffix)?;
        let mort_file = mortality_dir.join(format!(
            "NHANES_{}_{}_MORT_2019_PUBLIC.dat",
            start,
            start + 1
        ));
        let (mort, mdigest) = mortality(&mort_file)?;
        sources.extend(
            c.sources
                .iter()
                .map(|(f, d)| (format!("{start}/{f}"), d.to_string())),
        );
        sources.push((
            mort_file
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_default(),
            mdigest.to_string(),
        ));
        let diet = crate::diet::days(&c);
        let (stem, var, valid) = crate::concepts::PRESCRIPTIONS;
        let mut rx = c.first_rows(stem, var);
        // "No prescription medicine in the past month" (RXDUSE = 2) is a
        // count of zero, not a missing count.
        for (seqn, used) in c.first_rows(stem, "RXDUSE") {
            if used == 2.0 {
                rx.entry(seqn).or_insert(0.0);
            }
        }
        let mut counts = CycleCounts::default();
        for seqn in c.participants() {
            counts.participants += 1;
            let days = diet.get(&seqn).map(Vec::as_slice).unwrap_or(&[]);
            let extra: Vec<(&str, f64)> = rx
                .get(&seqn)
                .filter(|v| (valid.0..=valid.1).contains(*v))
                .map(|v| ("prescriptions", *v))
                .into_iter()
                .collect();
            match subject(&c, seqn, mort.get(&seqn), days, &extra) {
                Ok((s, d)) => {
                    // The invariant the model's encoder enforces, checked
                    // here too: nothing after the examination is input.
                    if s.observations.iter().any(|o| o.t > s.entry)
                        || s.events
                            .iter()
                            .any(|e| e.t > s.entry && !CODES.contains(&e.code.as_str()))
                    {
                        bail!("{}: an input lies after the examination", s.subject_id);
                    }
                    s.validate().map_err(anyhow::Error::msg)?;
                    counts.subjects += 1;
                    counts.with_diet += usize::from(!days.is_empty());
                    for e in s.events.iter().filter(|e| e.t > s.entry) {
                        *counts.deaths.entry(e.code.clone()).or_default() += 1;
                    }
                    lines.push_str(&serde_json::to_string(&s)?);
                    lines.push('\n');
                    design.push_str(&serde_json::to_string(&d)?);
                    design.push('\n');
                    all.push(s);
                }
                Err(why) => {
                    *counts
                        .excluded
                        .entry(
                            serde_json::to_value(why)?
                                .as_str()
                                .unwrap_or("?")
                                .to_string(),
                        )
                        .or_default() += 1
                }
            }
        }
        println!(
            "{start}: {} participants, {} subjects, deaths {:?}, excluded {:?}",
            counts.participants, counts.subjects, counts.deaths, counts.excluded
        );
        cycles.insert(start, counts);
    }
    let deaths = cycles
        .values()
        .map(|c| c.deaths.values().sum::<usize>())
        .sum();
    let report = BuildReport {
        subjects: all.len(),
        deaths,
        horizons: Horizons::of(&all),
        cycles,
        sources,
        terms: nhanes_terms(),
    };
    std::fs::write(out.join(TIMELINES), lines)?;
    std::fs::write(out.join(DESIGN), design)?;
    std::fs::write(
        out.join(BUILD),
        serde_json::to_string_pretty(&report)? + "\n",
    )?;
    println!(
        "{} subjects, {deaths} deaths; horizons by cycle {:?}",
        report.subjects, report.horizons.0
    );
    Ok(())
}

fn stratum(v: &serde_json::Value) -> String {
    let entry = v["entry"].as_f64().unwrap_or(0.0);
    let died = v["events"]
        .as_array()
        .is_some_and(|ev| ev.iter().any(|e| e["t"].as_f64().unwrap_or(0.0) > entry));
    let band = match entry as u32 {
        0..=39 => "18-39",
        40..=59 => "40-59",
        60..=79 => "60-79",
        _ => "80+",
    };
    format!(
        "{}|{}|{band}",
        v["source"].as_str().unwrap_or("?"),
        if died { "died" } else { "alive" }
    )
}

/// Partition once and pin the data, the partition and the criteria.
pub fn freeze(data: &Path) -> Result<()> {
    let text = std::fs::read_to_string(data.join(TIMELINES)).context("run build first")?;
    let dataset = Digest::of(text.as_bytes());
    let units = units_from_json_lines(&text, stratum).map_err(anyhow::Error::msg)?;
    let p = partition(&units, dataset, &PartitionSpec::default())?;
    let canonical = p.canonical()?;
    let criteria = serde_json::to_vec_pretty(&crate::report::preregistered())?;
    let ledger = Ledger::at(data.join(LEDGER));
    for (name, content) in [
        (TIMELINES, text.as_bytes()),
        (PARTITION, &canonical[..]),
        (CRITERIA, &criteria[..]),
    ] {
        ledger.pin(name, content)?;
    }
    std::fs::write(data.join(PARTITION), &canonical)?;
    std::fs::write(data.join(CRITERIA), &criteria)?;
    println!(
        "locked test {} subjects; {} repeats x {} folds; partition {}",
        p.locked.len(),
        p.repeats.len(),
        p.spec.folds,
        p.digest()?
    );
    Ok(())
}

/// The frozen inputs every experiment reads, each checked against its pin.
pub struct Frozen {
    /// Subjects by id.
    pub subjects: HashMap<String, Subject>,
    /// The partition.
    pub partition: Partition,
    /// Digests of the data and partition.
    pub digests: (String, String),
    /// Horizons per cycle.
    pub horizons: Horizons,
}

/// Load and verify the frozen inputs.
pub fn frozen(data: &Path) -> Result<Frozen> {
    let ledger = Ledger::at(data.join(LEDGER));
    for name in [TIMELINES, PARTITION, CRITERIA] {
        let bytes =
            std::fs::read(data.join(name)).with_context(|| format!("{name}: run freeze first"))?;
        ledger.check(name, &bytes)?;
    }
    let subjects = read_jsonl(data.join(TIMELINES))?;
    let horizons = Horizons::of(&subjects);
    let partition: Partition = serde_json::from_slice(&std::fs::read(data.join(PARTITION))?)?;
    let digests = (
        partition.dataset.to_string(),
        partition.digest()?.to_string(),
    );
    Ok(Frozen {
        subjects: subjects
            .into_iter()
            .map(|s| (s.subject_id.clone(), s))
            .collect(),
        partition,
        digests,
        horizons,
    })
}

/// One run's record.
#[derive(Serialize, Deserialize)]
pub struct Run {
    /// The arm.
    pub arm: Arm,
    /// Seed.
    pub seed: u64,
    /// `(repeat, fold)`, or `None` for the locked test.
    pub fold: Option<(usize, usize)>,
    /// Outcomes shuffled across training subjects.
    pub permuted: bool,
    /// Dataset digest.
    pub dataset: String,
    /// Partition digest.
    pub partition: String,
    /// Training run.
    pub info: RunInfo,
    /// Test subjects.
    pub n_test: usize,
    /// Wall-clock seconds of training and scoring.
    pub seconds: f64,
    /// The metrics.
    pub metrics: Metrics,
    /// Why the locked test was scored again, if it was.
    pub reason: Option<String>,
}

fn write_run(data: &Path, name: &str, run: &Run) -> Result<()> {
    let dir = data.join("runs");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(name);
    let tmp = path.with_extension("tmp");
    std::fs::File::create(&tmp)?.write_all(&serde_json::to_vec_pretty(run)?)?;
    std::fs::rename(tmp, path)?;
    Ok(())
}

/// Every subject's outcome moved to another subject, relative to entry:
/// the inputs stay, the link between inputs and outcome is broken.
fn permute(subjects: &[Subject], seed: u64) -> Vec<Subject> {
    let mut order: Vec<usize> = (0..subjects.len()).collect();
    let mut state = seed ^ 0xD1B5_4A32_D192_ED03;
    for i in (1..order.len()).rev() {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        order.swap(i, (state >> 33) as usize % (i + 1));
    }
    subjects
        .iter()
        .zip(&order)
        .map(|(s, &j)| {
            let donor = &subjects[j];
            let mut v = s.clone();
            v.events.retain(|e| e.t < s.entry);
            v.events
                .extend(donor.events.iter().filter(|e| e.t > donor.entry).map(|e| {
                    splinter_sdk::model::timeline::Event {
                        t: s.entry + (e.t - donor.entry),
                        code: e.code.clone(),
                    }
                }));
            v.at_risk = donor
                .at_risk
                .iter()
                .map(|w| splinter_sdk::model::timeline::AtRisk {
                    code: w.code.clone(),
                    from: s.entry + (w.from - donor.entry),
                    to: s.entry + (w.to - donor.entry),
                })
                .collect();
            v
        })
        .collect()
}

fn run_one(
    f: &Frozen,
    arm: Arm,
    seed: u64,
    train_ids: &[&str],
    test_ids: &[&str],
    permuted: bool,
) -> Result<(
    Run,
    Vec<Subject>,
    Vec<splinter_sdk::model::timeline::Prediction>,
)> {
    let t0 = Instant::now();
    let train: Vec<Subject> = train_ids.iter().map(|id| f.subjects[*id].clone()).collect();
    let train = if permuted {
        permute(&train, seed)
    } else {
        train
    };
    let refs: Vec<&Subject> = train.iter().collect();
    let (model, info) = fit(arm, &refs, &Training { seed, steps: STEPS })?;
    let test: Vec<Subject> = test_ids
        .iter()
        .map(|id| arm.view(&f.subjects[*id]))
        .collect();
    let preds = model.predict(&test)?;
    let metrics = evaluate(&test, &preds, &f.horizons);
    let run = Run {
        arm,
        seed,
        fold: None,
        permuted,
        dataset: f.digests.0.clone(),
        partition: f.digests.1.clone(),
        info,
        n_test: test.len(),
        seconds: t0.elapsed().as_secs_f64(),
        metrics,
        reason: None,
    };
    Ok((run, test, preds))
}

/// Train and score `arm` on the folds asked for (all by default).
pub fn cv(
    data: &Path,
    arm: Arm,
    seed: u64,
    repeats: &[usize],
    folds: &[usize],
    permuted: bool,
) -> Result<()> {
    let f = frozen(data)?;
    let rs: Vec<usize> = if repeats.is_empty() {
        (0..f.partition.repeats.len()).collect()
    } else {
        repeats.to_vec()
    };
    let ks: Vec<usize> = if folds.is_empty() {
        (0..f.partition.spec.folds as usize).collect()
    } else {
        folds.to_vec()
    };
    for &r in &rs {
        for &k in &ks {
            let name = format!(
                "{}{}-s{seed}-r{r}-k{k}.json",
                arm.name(),
                if permuted { "-permuted" } else { "" }
            );
            if data.join("runs").join(&name).exists() {
                println!("{name}: done already");
                continue;
            }
            let (train, test) = f.partition.fold(r, k);
            let (mut run, _, _) = run_one(&f, arm, seed, &train, &test, permuted)?;
            run.fold = Some((r, k));
            let m = &run.metrics;
            println!(
                "{name}: ibs_0_15 {:?} uno_c10 {:?} steps {} in {:.0}s",
                m.ibs_0_15.as_ref().map(|x| x.value),
                m.uno_c.get(&10).map(|x| x.value),
                run.info.steps,
                run.seconds
            );
            write_run(data, &name, &run)?;
        }
    }
    Ok(())
}

/// Per-subject results on the locked test, kept for the design-based intervals.
#[derive(Serialize, Deserialize)]
pub struct LockedDetail {
    /// `(subject_id, integrated Brier term)` on the 15-year set.
    pub ibs_terms: Vec<(String, f64)>,
    /// `(subject_id, predicted all-cause risk by 10 years)`.
    pub risk_10: Vec<(String, f64)>,
}

/// Train on everything but the locked test and score the locked test.
pub fn final_test(data: &Path, arm: Arm, seed: u64, reason: Option<&str>) -> Result<()> {
    let f = frozen(data)?;
    nhanes_terms()
        .permits(Use::Training)
        .map_err(anyhow::Error::msg)?;
    let name = format!("{}-s{seed}-locked.json", arm.name());
    if data.join("runs").join(&name).exists() && reason.is_none() {
        bail!("{name}: the locked test was already scored for this arm and seed; pass --reason to score it again, and the reason is recorded");
    }
    let locked: std::collections::HashSet<&str> =
        f.partition.locked.iter().map(String::as_str).collect();
    let mut train: Vec<&str> = f
        .subjects
        .keys()
        .map(String::as_str)
        .filter(|id| !locked.contains(id))
        .collect();
    train.sort_unstable();
    let test: Vec<&str> = f.partition.locked.iter().map(String::as_str).collect();
    let (mut run, test_subjects, preds) = run_one(&f, arm, seed, &train, &test, false)?;
    run.reason = reason.map(str::to_string);
    let detail = LockedDetail {
        ibs_terms: ibs_terms(&test_subjects, &preds, &f.horizons),
        risk_10: test_subjects
            .iter()
            .zip(&preds)
            .map(|(s, p)| (s.subject_id.clone(), 1.0 - p.survival(10.0)))
            .collect(),
    };
    std::fs::write(
        data.join("runs")
            .join(format!("{}-s{seed}-locked-detail.json", arm.name())),
        serde_json::to_vec(&detail)?,
    )
    .ok();
    write_run(data, &name, &run)?;
    println!("{name}: {}", serde_json::to_string_pretty(&run.metrics)?);
    Ok(())
}

/// Every cross-validation run of `arm` (not permuted), by fold.
pub fn cv_runs(data: &Path, arm: Arm) -> Result<BTreeMap<(usize, usize), Run>> {
    let mut out = BTreeMap::new();
    let Ok(dir) = std::fs::read_dir(data.join("runs")) else {
        return Ok(out);
    };
    for e in dir.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if !name.starts_with(&format!("{}-s", arm.name()))
            || name.contains("locked")
            || !name.ends_with(".json")
        {
            continue;
        }
        let run: Run = serde_json::from_slice(&std::fs::read(e.path())?)?;
        if let (Some(fold), false) = (run.fold, run.permuted) {
            out.insert(fold, run);
        }
    }
    Ok(out)
}

/// A metric two arms are compared on; its name says which way is better.
pub struct Compared {
    /// Name, with the better direction.
    pub name: &'static str,
    /// Reads it from a run's metrics.
    pub get: fn(&Metrics) -> Option<f64>,
}

/// The metrics compared.
pub fn compared_metrics() -> Vec<Compared> {
    vec![
        Compared {
            name: "ibs_0_15 (lower better)",
            get: |m| m.ibs_0_15.as_ref().map(|x| x.value),
        },
        Compared {
            name: "brier_10 (lower better)",
            get: |m| m.brier.get(&10).map(|x| x.value),
        },
        Compared {
            name: "uno_c_5 (higher better)",
            get: |m| m.uno_c.get(&5).map(|x| x.value),
        },
        Compared {
            name: "uno_c_10 (higher better)",
            get: |m| m.uno_c.get(&10).map(|x| x.value),
        },
        Compared {
            name: "calibration_slope_10",
            get: |m| m.calibration_10.as_ref().map(|c| c.slope),
        },
    ]
}

/// Two arms on the folds both ran on.
pub fn compare(data: &Path, a: Arm, b: Arm) -> Result<()> {
    let (ra, rb) = (cv_runs(data, a)?, cv_runs(data, b)?);
    let folds: Vec<&(usize, usize)> = ra.keys().filter(|k| rb.contains_key(k)).collect();
    if folds.len() < 2 {
        bail!(
            "{} folds ran for both arms; at least two are needed",
            folds.len()
        );
    }
    // Each fold tests on about 1/(K-1) of what it trains on.
    let k = folds.iter().map(|f| f.1).max().unwrap_or(1) + 1;
    let test_over_train = 1.0 / (k as f64 - 1.0);
    println!(
        "{} vs {} on {} folds (test/train {:.3})",
        a.name(),
        b.name(),
        folds.len(),
        test_over_train
    );
    println!(
        "{:<28} {:>10} {:>10} {:>10} {:>22} {:>8}",
        "metric",
        a.name(),
        b.name(),
        "diff",
        "95% CI (corrected)",
        "p"
    );
    for Compared { name, get } in compared_metrics() {
        let pairs: Vec<(f64, f64)> = folds
            .iter()
            .filter_map(|f| Some((get(&ra[f].metrics)?, get(&rb[f].metrics)?)))
            .collect();
        let diffs: Vec<f64> = pairs.iter().map(|(x, y)| x - y).collect();
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let (ma, mb) = (
            mean(&pairs.iter().map(|p| p.0).collect::<Vec<_>>()),
            mean(&pairs.iter().map(|p| p.1).collect::<Vec<_>>()),
        );
        match corrected_resampled_t(&diffs, test_over_train) {
            Some(t) => println!(
                "{name:<28} {ma:>10.5} {mb:>10.5} {:>+10.5} [{:>+9.5}, {:>+9.5}] {:>8.4}",
                t.mean, t.ci95.0, t.ci95.1, t.p_two_sided
            ),
            None => println!("{name:<28} {ma:>10.5} {mb:>10.5} (no variance across folds)"),
        }
    }
    Ok(())
}

/// The design records by subject id.
pub fn designs(data: &Path) -> Result<HashMap<String, Design>> {
    let text = std::fs::read_to_string(data.join(DESIGN))?;
    text.lines()
        .filter(|l| !l.is_empty())
        .map(|l| {
            let v: serde_json::Value = serde_json::from_str(l)?;
            let d = Design {
                subject_id: v["subject_id"].as_str().unwrap_or_default().to_string(),
                cycle: v["cycle"].as_u64().unwrap_or(0) as u16,
                stratum: v["stratum"].as_u64().unwrap_or(0),
                psu: v["psu"].as_u64().unwrap_or(0),
            };
            Ok((d.subject_id.clone(), d))
        })
        .collect()
}
