// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Death with diabetes or hypertension listed on the certificate (T2 of the
//! secondary estimands, paper section 4.5).
//!
//! Swedish Embedded AB implements validation of risk models whose outcomes
//! arrive years later, censored and competing, for its clients. If your team
//! needs expertise in telling whether a model predicts a specific outcome or
//! only who dies, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! The outcome of a flag is death WITH it by year `t`, competing with death
//! without it; a death with no multiple-cause data is censored at its time and
//! counted. A certificate mention is not a diagnosis.
//!
//! Flagged events are few, so folds are not scored one by one: the out-of-fold
//! predictions of one repeat are pooled (each subject appears once) and scored
//! as one set, with a percentile interval from resampling clusters (cycle,
//! stratum, PSU). The control that matters is the all-cause incidence used as
//! a ranker: a flag model adds something only if it ranks flagged deaths
//! better than that does. Below the pre-registered count of flagged events a
//! horizon is described and not compared.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use splinter_sdk::model::timeline::survival::auc;
use splinter_sdk::model::timeline::survival::binary::auroc;
use splinter_sdk::model::timeline::survival::brier::brier;
use splinter_sdk::model::timeline::survival::compare::cluster_bootstrap_by;
use splinter_sdk::model::timeline::survival::estimate::{aalen_johansen, censoring};
use splinter_sdk::model::timeline::survival::Obs;
use splinter_sdk::model::timeline::{observed, Subject};

use crate::build::{Design, CODES};
use crate::causes::Cause;
use crate::commands::{designs, frozen, Frozen};
use crate::external::{baseline_dir, fold_stem, parse};
use crate::metrics::all_cause;
use crate::report::cluster;

/// Horizons, in years.
pub const HORIZONS: [u32; 2] = [5, 10];
/// Flagged events a horizon needs before models are ranked (section 4.5).
pub const MIN_EVENTS: usize = 100;
#[cfg(not(test))]
const REPS: usize = 1000;
/// Fewer replicates keep the specifications fast; they test the sign of the interval, not its width.
#[cfg(test)]
const REPS: usize = 100;
const SEED: u64 = 20261008;

/// The two conditions the linked file flags.
pub const FLAGS: [&str; 2] = ["diabetes", "hypertension"];

/// A subject's outcome for `flag`: cause 0 death with the flag, 1 death
/// without it, censored otherwise. A death with no multiple-cause data is
/// censored at its time.
pub fn flag_obs(all: Obs, cause: &Cause, flag: &str) -> Obs {
    let listed = match flag {
        "diabetes" => cause.diabetes_mcod,
        _ => cause.hypertension_mcod,
    };
    match (all.cause, listed, cause.mcod_available) {
        (Some(_), Some(true), true) => Obs {
            cause: Some(0),
            ..all
        },
        (Some(_), Some(false), true) => Obs {
            cause: Some(1),
            ..all
        },
        _ => Obs { cause: None, ..all },
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FlagLine {
    subject_id: String,
    flag_cif: BTreeMap<String, Vec<f64>>,
}

/// Where a model's risk of the flagged outcome comes from.
#[derive(Clone, Debug, PartialEq)]
pub enum Source {
    /// A baseline whose files carry `flag_cif` (`flags.py`).
    Flag(String),
    /// The all-cause incidence of a baseline, used as a ranker.
    Ranker(String),
}

impl Source {
    pub fn parse(s: &str) -> Source {
        match s.strip_prefix("ranker:") {
            Some(b) => Source::Ranker(b.to_string()),
            None => Source::Flag(s.to_string()),
        }
    }
    pub fn name(&self) -> String {
        match self {
            Source::Flag(n) => n.clone(),
            Source::Ranker(n) => format!("ranker:{n}"),
        }
    }
}

/// Yearly curves per subject for `flag` over all folds of `repeat`.
fn curves(
    data: &Path,
    f: &Frozen,
    source: &Source,
    repeat: usize,
    flag: &str,
) -> Result<HashMap<String, Vec<f64>>> {
    let (Source::Flag(n) | Source::Ranker(n)) = source;
    let dir = baseline_dir(data, n)?;
    let mut out = HashMap::new();
    for fold in 0..f.partition.spec.folds as usize {
        let file = dir.join(format!("{}.jsonl", fold_stem(repeat, fold)));
        let text = std::fs::read_to_string(&file).with_context(|| format!("{}", file.display()))?;
        let (_, test) = f.partition.fold(repeat, fold);
        match source {
            Source::Ranker(_) => {
                // `parse` checks the fold's subject set; the all-cause curve is `1 - survival`.
                for (id, p) in test.iter().zip(parse(&text, &test)?) {
                    let curve: Vec<f64> = (1..=15)
                        .map(|t| 1.0 - crate::metrics::Outlook::survival(&p, t as f64))
                        .collect();
                    out.insert((*id).to_string(), curve);
                }
            }
            Source::Flag(_) => {
                for line in text.lines().filter(|l| !l.trim().is_empty()) {
                    let l: FlagLine = serde_json::from_str(line)
                        .with_context(|| format!("{}", file.display()))?;
                    let curve = l
                        .flag_cif
                        .get(flag)
                        .with_context(|| format!("no {flag} curve for {}", l.subject_id))?;
                    if curve.len() != 15 || curve.iter().any(|v| !v.is_finite()) {
                        bail!(
                            "{}: a bad {flag} curve for {}",
                            file.display(),
                            l.subject_id
                        );
                    }
                    out.insert(l.subject_id, curve.clone());
                }
                if test.iter().any(|id| !out.contains_key(*id)) {
                    bail!("{}: does not cover the fold", file.display());
                }
            }
        }
    }
    Ok(out)
}

/// What one model scores at one horizon on the pooled set.
#[derive(Clone, Debug, PartialEq)]
pub struct Eval {
    pub n: usize,
    pub events: usize,
    pub auc: f64,
    pub brier: f64,
    pub null_brier: f64,
    /// Observed (Aalen-Johansen) over mean predicted; meaningful for a flag
    /// model, not for a ranker.
    pub oe: f64,
}

/// Score `risk` (probability of cause 0 by `t`) against `obs`.
pub fn evaluate(obs: &[Obs], risk: &[f64], t: f64) -> Option<Eval> {
    let g = censoring(&all_cause(obs.to_vec()));
    let null = vec![aalen_johansen(obs, 0).at(t); obs.len()];
    let wsum: f64 = obs.iter().map(|o| o.weight).sum();
    let expected = risk.iter().zip(obs).map(|(r, o)| r * o.weight).sum::<f64>() / wsum;
    Some(Eval {
        n: obs.len(),
        events: obs
            .iter()
            .filter(|o| o.cause == Some(0) && o.time <= t)
            .count(),
        auc: auc::at(risk, obs, 0, t, &g)?,
        brier: brier(risk, obs, 0, t, &g)?,
        null_brier: brier(&null, obs, 0, t, &g)?,
        oe: aalen_johansen(obs, 0).at(t) / expected,
    })
}

/// A percentile interval.
type Bounds = (f64, f64);
/// An estimate with its percentile interval.
type Difference = (f64, f64, f64);

/// AUC interval, and the interval of the AUC difference against `other`,
/// resampling clusters.
fn auc_intervals(
    obs: &[Obs],
    risk: &[f64],
    other: Option<&[f64]>,
    clusters: &[u64],
    t: f64,
) -> (Option<Bounds>, Option<Difference>) {
    let pick = |ix: &[usize], r: &[f64]| {
        let o: Vec<Obs> = ix.iter().map(|&i| obs[i]).collect();
        let g = censoring(&all_cause(o.clone()));
        auc::at(&ix.iter().map(|&i| r[i]).collect::<Vec<_>>(), &o, 0, t, &g)
    };
    let own = cluster_bootstrap_by(clusters, REPS, 0.95, SEED, |ix| pick(ix, risk));
    let diff = other.and_then(|o| {
        cluster_bootstrap_by(clusters, REPS, 0.95, SEED, |ix| {
            Some(pick(ix, risk)? - pick(ix, o)?)
        })
    });
    (
        own.map(|i| (i.lo, i.hi)),
        diff.map(|i| (i.estimate, i.lo, i.hi)),
    )
}

fn at_year(curve: &[f64], t: u32) -> f64 {
    curve[t as usize - 1]
}

/// The pooled out-of-fold evaluation of `sources` for one repeat, as markdown,
/// with the first `ranker:` source as the control.
pub fn report(
    data: &Path,
    sources: &[Source],
    repeat: usize,
    with_intervals: bool,
) -> Result<String> {
    let f = frozen(data)?;
    let causes: HashMap<String, Cause> = std::fs::read_to_string(data.join("causes.jsonl"))
        .context("causes.jsonl: rebuild the data with the current `build`")?
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str::<Cause>(l).map(|c| (c.subject_id.clone(), c)))
        .collect::<Result<_, _>>()?;
    let design: HashMap<String, Design> = designs(data)?;
    let mut ids: Vec<&str> = (0..f.partition.spec.folds as usize)
        .flat_map(|k| f.partition.fold(repeat, k).1)
        .collect();
    ids.sort_unstable();
    let mut out = String::new();
    for flag in FLAGS {
        let by_source: Vec<(String, HashMap<String, Vec<f64>>)> = sources
            .iter()
            .map(|s| Ok((s.name(), curves(data, &f, s, repeat, flag)?)))
            .collect::<Result<_>>()?;
        for t in HORIZONS {
            let subs: Vec<&Subject> = ids
                .iter()
                .map(|id| &f.subjects[*id])
                .filter(|s| f.horizons.supports(s, f64::from(t)))
                .collect();
            let all = all_cause(observed(
                &subs.iter().map(|s| (*s).clone()).collect::<Vec<_>>(),
                &CODES,
            ));
            let obs: Vec<Obs> = subs
                .iter()
                .zip(all)
                .map(|(s, a)| flag_obs(a, &causes[&s.subject_id], flag))
                .collect();
            let clusters: Vec<u64> = subs
                .iter()
                .map(|s| design.get(&s.subject_id).map_or(0, cluster))
                .collect();
            let events = obs
                .iter()
                .filter(|o| o.cause == Some(0) && o.time <= f64::from(t))
                .count();
            let rankable = events >= MIN_EVENTS;
            writeln!(
                out,
                "\n### Death with {flag} listed, {t} years, repeat {repeat}: {} subjects, {events} flagged deaths{}\n",
                subs.len(),
                if rankable { "" } else { " (below the count needed to rank models: descriptive only)" }
            )?;
            writeln!(
                out,
                "| model | AUC [95% CI] | Brier | IPA | O/E | AUC minus ranker [95% CI] |"
            )?;
            writeln!(out, "|---|---|---|---|---|---|")?;
            let risks: Vec<Vec<f64>> = by_source
                .iter()
                .map(|(_, m)| subs.iter().map(|s| at_year(&m[&s.subject_id], t)).collect())
                .collect();
            let control = sources.iter().position(|s| matches!(s, Source::Ranker(_)));
            for (i, (name, _)) in by_source.iter().enumerate() {
                let Some(e) = evaluate(&obs, &risks[i], f64::from(t)) else {
                    writeln!(out, "| {name} | not scored | | | | |")?;
                    continue;
                };
                let is_ranker = matches!(sources[i], Source::Ranker(_));
                let other = control.filter(|c| *c != i).map(|c| risks[c].as_slice());
                let (ci, diff) = if with_intervals && rankable {
                    auc_intervals(&obs, &risks[i], other, &clusters, f64::from(t))
                } else {
                    (None, None)
                };
                let interval = |x: Option<(f64, f64)>| {
                    x.map_or(String::new(), |(a, b)| format!(" [{a:.3}, {b:.3}]"))
                };
                let delta = match (&diff, is_ranker, rankable) {
                    (Some((d, a, b)), _, _) => format!("{d:+.4} [{a:+.4}, {b:+.4}]"),
                    (None, true, _) => "control".to_string(),
                    (None, false, false) => "not compared".to_string(),
                    _ => "n/a".to_string(),
                };
                // A ranker's Brier score and calibration are those of a different outcome.
                let (b, ipa, oe) = if is_ranker {
                    ("-".to_string(), "-".to_string(), "-".to_string())
                } else {
                    (
                        format!("{:.5}", e.brier),
                        format!("{:.3}", 1.0 - e.brier / e.null_brier),
                        format!("{:.2}", e.oe),
                    )
                };
                writeln!(
                    out,
                    "| {name} | {:.4}{} | {b} | {ipa} | {oe} | {delta} |",
                    e.auc,
                    interval(ci)
                )?;
            }
        }
        if let Some(c) = sources.iter().position(|s| matches!(s, Source::Ranker(_))) {
            writeln!(out, "\nAmong deaths with multiple-cause data, repeat {repeat}: AUROC of the implied share of the all-cause incidence by 15 years that carries {flag} (a flag model's curve over the first ranker's), against the flag listed or not.\n")?;
            writeln!(out, "| model | deaths | with {flag} | AUROC |")?;
            writeln!(out, "|---|---|---|---|")?;
            let dead: Vec<&&str> = ids
                .iter()
                .filter(|id| {
                    let s = &f.subjects[**id];
                    s.events
                        .iter()
                        .any(|e| e.t > s.entry && CODES.contains(&e.code.as_str()))
                        && causes[**id].mcod_available
                })
                .collect();
            let label: Vec<bool> = dead
                .iter()
                .map(|id| match flag {
                    "diabetes" => causes[**id].diabetes_mcod == Some(true),
                    _ => causes[**id].hypertension_mcod == Some(true),
                })
                .collect();
            let weight: Vec<f64> = dead.iter().map(|id| f.subjects[**id].weight).collect();
            for (i, (name, m)) in by_source.iter().enumerate() {
                if matches!(sources[i], Source::Ranker(_)) {
                    continue;
                }
                let score: Vec<f64> = dead
                    .iter()
                    .map(|id| {
                        let all = by_source[c].1[**id][14].max(1e-12);
                        m[**id][14] / all
                    })
                    .collect();
                let flagged = label.iter().filter(|l| **l).count();
                let cell = if flagged == 0 || flagged == label.len() {
                    "n/a".to_string()
                } else {
                    format!("{:.4}", auroc(&score, &label, &weight))
                };
                writeln!(out, "| {name} | {} | {flagged} | {cell} |", dead.len())?;
            }
        }
    }
    Ok(out)
}

/// Print the report for `repeat`.
pub fn run(
    data: &Path,
    models: &[String],
    rankers: &[String],
    repeat: usize,
    intervals: bool,
) -> Result<()> {
    let mut sources: Vec<Source> = rankers.iter().map(|r| Source::Ranker(r.clone())).collect();
    sources.extend(models.iter().map(|m| Source::parse(m)));
    if !sources.iter().any(|s| matches!(s, Source::Ranker(_))) {
        bail!("name at least one --ranker: the all-cause control is part of the question");
    }
    println!("{}", report(data, &sources, repeat, intervals)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cause(available: bool, d: Option<bool>, h: Option<bool>) -> Cause {
        Cause {
            subject_id: "a".into(),
            ucod: None,
            diabetes_mcod: d,
            hypertension_mcod: h,
            mcod_available: available,
        }
    }

    #[test]
    fn a_death_is_with_or_without_the_flag_and_unknown_is_censored_not_flag_free() {
        let died = Obs {
            time: 4.0,
            cause: Some(0),
            weight: 2.0,
        };
        let c = cause(true, Some(true), Some(false));
        assert_eq!(flag_obs(died, &c, "diabetes").cause, Some(0));
        assert_eq!(flag_obs(died, &c, "hypertension").cause, Some(1));
        let unknown = cause(false, None, None);
        let o = flag_obs(died, &unknown, "diabetes");
        assert_eq!((o.cause, o.time, o.weight), (None, 4.0, 2.0));
        let alive = Obs {
            time: 9.0,
            cause: None,
            weight: 1.0,
        };
        assert_eq!(
            flag_obs(alive, &cause(false, None, None), "diabetes").cause,
            None
        );
    }

    /// Deterministic generator: no random-number dependency in the test.
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Everyone dies at a rate that rises with `x`; a death carries the flag
    /// with probability `z`, independent of `x`. An all-cause ranker knows only
    /// who dies (`x`), so it has some AUC for the flag, and a model that also
    /// knows `z` must do better.
    fn cohort(n: usize) -> (Vec<Obs>, Vec<f64>, Vec<f64>) {
        let mut s = 3u64;
        let (mut obs, mut all_risk, mut flag_risk) = (vec![], vec![], vec![]);
        for _ in 0..n {
            let x = lcg(&mut s);
            let rate = 0.02 + 0.06 * x;
            let t = -lcg(&mut s).ln() / rate;
            let z = lcg(&mut s);
            let carries = lcg(&mut s) < z;
            let censor = 12.0 + 6.0 * lcg(&mut s);
            obs.push(if t <= censor {
                Obs {
                    time: t,
                    cause: Some(if carries { 0 } else { 1 }),
                    weight: 1.0,
                }
            } else {
                Obs::censored(censor)
            });
            all_risk.push(1.0 - (-rate * 10.0).exp());
            flag_risk.push((1.0 - (-rate * 10.0).exp()) * z);
        }
        (obs, all_risk, flag_risk)
    }

    #[test]
    fn a_model_of_the_flag_beats_the_all_cause_ranker_for_the_flagged_outcome() {
        let (obs, all_risk, flag_risk) = cohort(20_000);
        let ranker = evaluate(&obs, &all_risk, 10.0).unwrap();
        let model = evaluate(&obs, &flag_risk, 10.0).unwrap();
        assert!(
            model.auc > ranker.auc + 0.03,
            "{} vs {}",
            model.auc,
            ranker.auc
        );
        assert!(
            ranker.auc > 0.55,
            "death itself carries some signal: {}",
            ranker.auc
        );
        assert!(model.brier < model.null_brier, "{model:?}");
        assert!((model.oe - 1.0).abs() < 0.15, "{}", model.oe);
        assert!(model.events >= MIN_EVENTS);
    }

    #[test]
    fn the_interval_of_the_difference_excludes_zero_when_the_model_is_better() {
        let (obs, all_risk, flag_risk) = cohort(6_000);
        let clusters: Vec<u64> = (0..obs.len() as u64).map(|i| i / 5).collect();
        let (_, diff) = auc_intervals(&obs, &flag_risk, Some(&all_risk), &clusters, 10.0);
        let (d, lo, hi) = diff.unwrap();
        assert!(lo > 0.0 && d > lo && hi > d, "{d} [{lo}, {hi}]");
    }
}
