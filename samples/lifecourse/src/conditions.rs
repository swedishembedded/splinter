// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Prevalent conditions at the examination and undiagnosed disease, from the
//! NHANES questionnaires and laboratory files (T4 and T5 of the secondary
//! estimands; definitions in the paper's Appendix C).
//!
//! Swedish Embedded AB implements solutions for turning survey and clinical
//! records into well-defined labels for its clients. If your team needs
//! expertise in defining a condition from questionnaires and laboratory
//! values without leaking the label into the inputs, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! A label is `Some(true)`, `Some(false)` or `None`. A refused or "don't know"
//! answer is `None`; a condition a cycle did not measure is `None`; positive
//! needs any criterion positive, negative needs the criteria that define the
//! label observed and negative. Nothing is guessed from a missing answer.
//! Thresholds are those of the cited guidelines; the codebooks of each cycle
//! were used to verify variable names, units and codes only.
//!
//! Variables are read through [`Lookup`] so the definitions can be specified
//! without a cycle's files: [`Person`] reads a real cycle.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::causes::Cause;
use crate::nhanes::{Cycle, CYCLES};

/// Where one participant's variables come from.
pub trait Lookup {
    /// The cycle's start year.
    fn cycle(&self) -> u16;
    /// A numeric variable, `None` if missing or not in the cycle's files.
    fn num(&self, var: &str) -> Option<f64>;
}

/// One participant of a real cycle.
pub struct Person<'a> {
    pub cycle: &'a Cycle,
    pub seqn: u64,
}

impl Lookup for Person<'_> {
    fn cycle(&self) -> u16 {
        self.cycle.start
    }
    fn num(&self, var: &str) -> Option<f64> {
        self.cycle.num(var, self.seqn)
    }
}

/// A yes/no question: 1 is yes, 2 is no, anything else (refused, don't know,
/// skipped) is unknown.
fn yes(x: Option<f64>) -> Option<bool> {
    match x? as i64 {
        1 => Some(true),
        2 => Some(false),
        _ => None,
    }
}

fn first(p: &impl Lookup, vars: &[&str]) -> Option<f64> {
    vars.iter().find_map(|v| p.num(v))
}

/// Mean of the readings of `vars` that fall inside `valid`.
fn mean_in(p: &impl Lookup, vars: &[&str], valid: (f64, f64)) -> Option<f64> {
    let v: Vec<f64> = vars
        .iter()
        .filter_map(|n| p.num(n))
        .filter(|x| (valid.0..=valid.1).contains(x))
        .collect();
    (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
}

/// Serum creatinine in mg/dL made comparable with the standardised assay, by
/// the correction each cycle's laboratory documentation gives: 1999-2000
/// `1.013 x + 0.147`, 2005-2006 `0.978 x - 0.016`; 2001-2004 state no
/// correction is necessary, and the 2007 values were adjusted by the laboratory.
pub fn standard_creatinine(cycle: u16, raw: f64) -> f64 {
    match cycle {
        1999 => 1.013 * raw + 0.147,
        2005 => 0.978 * raw - 0.016,
        _ => raw,
    }
}

/// Estimated glomerular filtration rate by the 2021 CKD-EPI creatinine
/// equation without a race term (Inker et al., N Engl J Med 2021).
pub fn egfr_ckd_epi_2021(creatinine_mg_dl: f64, age: f64, female: bool) -> f64 {
    let (kappa, alpha) = if female { (0.7, -0.241) } else { (0.9, -0.302) };
    let ratio = creatinine_mg_dl / kappa;
    142.0
        * ratio.min(1.0).powf(alpha)
        * ratio.max(1.0).powf(-1.200)
        * 0.9938_f64.powf(age)
        * if female { 1.012 } else { 1.0 }
}

/// The values the definitions use, read once.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Raw {
    pub age: Option<f64>,
    pub female: Option<bool>,
    pub pregnant: bool,
    /// DIQ010: 1 yes, 2 no, 3 borderline.
    pub told_diabetes: Option<f64>,
    pub insulin: Option<bool>,
    pub diabetes_pills: Option<bool>,
    pub hba1c: Option<f64>,
    pub glucose: Option<f64>,
    pub fasting_hours: Option<f64>,
    pub sbp: Option<f64>,
    pub dbp: Option<f64>,
    pub told_hypertension: Option<bool>,
    pub bp_medicine: Option<bool>,
    pub told_cholesterol: Option<bool>,
    pub cholesterol_never_checked: bool,
    pub cholesterol_medicine: Option<bool>,
    pub total_cholesterol: Option<f64>,
    pub told_kidney: Option<bool>,
    pub egfr: Option<f64>,
    pub acr: Option<f64>,
    pub hemoglobin: Option<f64>,
    pub told_osteoporosis: Option<bool>,
    pub phq9: Option<f64>,
    pub told_sleep_trouble: Option<bool>,
}

impl Raw {
    pub fn read(p: &impl Lookup) -> Raw {
        let age = p.num("RIDAGEYR");
        let female = p.num("RIAGENDR").map(|g| g == 2.0);
        let creatinine = first(p, &["LBXSCR", "LBDSCR"]).map(|c| standard_creatinine(p.cycle(), c));
        let ucr = p.num("URXUCR").filter(|c| *c > 0.0);
        let ccr_pills = first(p, &["DIQ070", "DID070"]);
        // Either cholesterol drug question: asked of those told to take medicine.
        let chol_med = [yes(p.num("BPQ090D")), yes(p.num("BPQ100D"))]
            .into_iter()
            .flatten()
            .reduce(|a, b| a || b);
        Raw {
            age,
            female,
            pregnant: p.num("RIDEXPRG") == Some(1.0),
            told_diabetes: p.num("DIQ010").filter(|x| [1.0, 2.0, 3.0].contains(x)),
            insulin: yes(p.num("DIQ050")),
            diabetes_pills: yes(ccr_pills),
            hba1c: p.num("LBXGH"),
            glucose: p.num("LBXGLU"),
            fasting_hours: p.num("PHAFSTHR"),
            sbp: mean_in(p, &["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"], (60.0, 280.0)),
            dbp: mean_in(p, &["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"], (20.0, 160.0)),
            told_hypertension: yes(p.num("BPQ020")),
            bp_medicine: yes(p.num("BPQ050A")),
            told_cholesterol: yes(p.num("BPQ080")),
            cholesterol_never_checked: p.num("BPQ060") == Some(2.0),
            cholesterol_medicine: chol_med,
            total_cholesterol: p.num("LBXTC"),
            told_kidney: yes(first(p, &["KIQ020", "KIQ022"])),
            egfr: match (creatinine, age, female) {
                (Some(c), Some(a), Some(f)) if c > 0.0 => Some(egfr_ckd_epi_2021(c, a, f)),
                _ => None,
            },
            acr: match (p.num("URXUMA"), ucr) {
                (Some(a), Some(c)) if a >= 0.0 => Some(100.0 * a / c),
                _ => None,
            },
            hemoglobin: p.num("LBXHGB"),
            told_osteoporosis: yes(p.num("OSQ060")),
            phq9: phq9(p),
            told_sleep_trouble: yes(p.num("SLQ050")),
        }
    }
}

/// The PHQ-9 score when all nine items were answered (0-3 each).
fn phq9(p: &impl Lookup) -> Option<f64> {
    (1..=9)
        .map(|i| {
            p.num(&format!("DPQ0{i}0"))
                .filter(|x| (0.0..=3.0).contains(x))
        })
        .sum()
}

/// Any criterion positive is positive; all defining criteria observed and
/// negative is negative; anything else is unknown.
fn any_of(criteria: &[Option<bool>], defining: &[bool]) -> Option<bool> {
    if criteria.iter().any(|c| *c == Some(true)) {
        Some(true)
    } else if defining.iter().all(|d| *d) {
        Some(false)
    } else {
        None
    }
}

fn at_least(x: Option<f64>, t: f64) -> Option<bool> {
    x.map(|v| v >= t)
}

fn pressure(r: &Raw, sbp: f64, dbp: f64) -> Option<bool> {
    Some(r.sbp? >= sbp || r.dbp? >= dbp)
}

fn kidney_markers(r: &Raw) -> (Option<bool>, bool) {
    let (low_egfr, albuminuria) = (r.egfr.map(|e| e < 60.0), at_least(r.acr, 30.0));
    let observed = r.egfr.is_some() && r.acr.is_some();
    (any_of(&[low_egfr, albuminuria], &[observed]), observed)
}

/// The prevalent conditions of T4.
pub fn prevalent(r: &Raw) -> BTreeMap<&'static str, Option<bool>> {
    let mut out = BTreeMap::new();
    let known_diabetes = [
        r.told_diabetes.map(|d| d == 1.0),
        r.insulin,
        r.diabetes_pills,
        at_least(r.hba1c, 6.5),
    ];
    out.insert(
        "diabetes",
        any_of(
            &known_diabetes,
            &[r.told_diabetes.is_some(), r.hba1c.is_some()],
        ),
    );
    let measured = pressure(r, 140.0, 90.0);
    out.insert(
        "hypertension",
        any_of(
            &[r.told_hypertension, r.bp_medicine, measured],
            &[r.told_hypertension.is_some(), measured.is_some()],
        ),
    );
    let strict = pressure(r, 130.0, 80.0);
    out.insert(
        "hypertension_130_80",
        any_of(
            &[r.told_hypertension, r.bp_medicine, strict],
            &[r.told_hypertension.is_some(), strict.is_some()],
        ),
    );
    out.insert("kidney_markers", kidney_markers(r).0);
    let anaemia = match (r.hemoglobin, r.female, r.pregnant) {
        (_, _, true) | (None, _, _) | (_, None, _) => None,
        (Some(h), Some(female), false) => Some(h < if female { 12.0 } else { 13.0 }),
    };
    out.insert("anaemia", anaemia);
    let checked = r.told_cholesterol.is_some() || r.cholesterol_never_checked;
    out.insert(
        "high_cholesterol",
        any_of(
            &[
                at_least(r.total_cholesterol, 240.0),
                r.told_cholesterol,
                r.cholesterol_medicine,
            ],
            &[r.total_cholesterol.is_some(), checked],
        ),
    );
    out.insert("osteoporosis", r.told_osteoporosis);
    out.insert("depression", at_least(r.phq9, 10.0));
    out.insert("sleep_problem", r.told_sleep_trouble);
    out
}

/// The undiagnosed disease of T5: the measured criterion among those who do
/// not report the diagnosis. `None` is "not in the screened population".
pub fn undiagnosed(r: &Raw) -> BTreeMap<&'static str, Option<bool>> {
    let mut out = BTreeMap::new();
    let on_diabetes_treatment = r.insulin == Some(true) || r.diabetes_pills == Some(true);
    out.insert(
        "undiagnosed_diabetes",
        match (r.told_diabetes, r.hba1c) {
            (Some(t), Some(h)) if t != 1.0 && !on_diabetes_treatment => Some(h >= 6.5),
            _ => None,
        },
    );
    out.insert(
        "undiagnosed_hypertension",
        match (r.told_hypertension, pressure(r, 140.0, 90.0)) {
            (Some(false), Some(m)) if r.bp_medicine != Some(true) => Some(m),
            _ => None,
        },
    );
    out.insert(
        "undiagnosed_kidney_markers",
        match (r.told_kidney, kidney_markers(r)) {
            (Some(false), (positive, true)) => positive,
            _ => None,
        },
    );
    out.insert(
        "undiagnosed_high_cholesterol",
        match r.total_cholesterol {
            Some(tc)
                if r.told_cholesterol != Some(true)
                    && r.cholesterol_medicine != Some(true)
                    && (r.told_cholesterol == Some(false) || r.cholesterol_never_checked) =>
            {
                Some(tc >= 240.0)
            }
            _ => None,
        },
    );
    out
}

/// Diabetes with fasting plasma glucose added (>= 126 mg/dL after at least
/// eight hours); defined only on the fasting subsample, which has its own weights.
pub fn diabetes_fasting(r: &Raw) -> Option<bool> {
    let glucose = r
        .glucose
        .filter(|_| r.fasting_hours.is_some_and(|h| h >= 8.0))?;
    let known = [
        r.told_diabetes.map(|d| d == 1.0),
        r.insulin,
        r.diabetes_pills,
        at_least(r.hba1c, 6.5),
        Some(glucose >= 126.0),
    ];
    any_of(&known, &[r.told_diabetes.is_some(), r.hba1c.is_some()])
}

/// One subject's row in `conditions.jsonl`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Conditions {
    pub subject_id: String,
    pub labels: BTreeMap<String, Option<bool>>,
    /// The pooled fasting-subsample weight, for labels that use glucose.
    pub fasting_weight: Option<f64>,
}

pub fn row(subject_id: String, p: &impl Lookup, fasting_weight: Option<f64>) -> Conditions {
    let r = Raw::read(p);
    let mut labels: BTreeMap<String, Option<bool>> = BTreeMap::new();
    for (k, v) in prevalent(&r).into_iter().chain(undiagnosed(&r)) {
        labels.insert(k.to_string(), v);
    }
    labels.insert("diabetes_fasting".into(), diabetes_fasting(&r));
    Conditions {
        subject_id,
        labels,
        fasting_weight,
    }
}

/// The pooled ten-cycle fasting weight, by the same rule as the examination weight.
pub fn pooled_fasting_weight(c: &Cycle, seqn: u64) -> Option<f64> {
    let w = if c.start <= 2001 {
        c.num("WTSAF4YR", seqn).map(|w| 0.2 * w)
    } else {
        c.num("WTSAF2YR", seqn).map(|w| 0.1 * w)
    };
    w.filter(|w| *w > 0.0)
}

/// Write `conditions.jsonl` for the subjects of the cohort in `data`
/// (those listed in `causes.jsonl`) from the NHANES files, and print how
/// many subjects each label has positive, negative and unknown.
pub fn write(nhanes: &Path, data: &Path) -> Result<()> {
    let cohort: HashSet<String> = std::fs::read_to_string(data.join("causes.jsonl"))
        .context("causes.jsonl: build the data first")?
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str::<Cause>(l).map(|c| c.subject_id))
        .collect::<Result<_, _>>()?;
    let mut text = String::new();
    let mut counts: BTreeMap<String, [usize; 3]> = BTreeMap::new();
    for (start, suffix) in CYCLES {
        let c = Cycle::load(nhanes, start, suffix)?;
        for seqn in c.participants() {
            let id = format!("nhanes-{start}-{seqn}");
            if !cohort.contains(&id) {
                continue;
            }
            let r = row(
                id,
                &Person { cycle: &c, seqn },
                pooled_fasting_weight(&c, seqn),
            );
            for (k, v) in &r.labels {
                let slot = match v {
                    Some(true) => 0,
                    Some(false) => 1,
                    None => 2,
                };
                counts.entry(k.clone()).or_default()[slot] += 1;
            }
            text.push_str(&serde_json::to_string(&r)?);
            text.push('\n');
        }
    }
    std::fs::write(data.join("conditions.jsonl"), text)?;
    println!(
        "{:<30} {:>9} {:>9} {:>9}",
        "label", "positive", "negative", "unknown"
    );
    for (k, [p, n, u]) in counts {
        println!("{k:<30} {p:>9} {n:>9} {u:>9}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct Fake(u16, HashMap<&'static str, f64>);
    impl Lookup for Fake {
        fn cycle(&self) -> u16 {
            self.0
        }
        fn num(&self, var: &str) -> Option<f64> {
            self.1.get(var).copied()
        }
    }
    fn person(cycle: u16, vars: &[(&'static str, f64)]) -> Fake {
        Fake(cycle, vars.iter().copied().collect())
    }
    fn prev(p: &Fake, label: &str) -> Option<bool> {
        prevalent(&Raw::read(p))[label]
    }

    #[test]
    fn diabetes_is_self_report_treatment_or_hba1c_and_the_2005_pill_question_has_another_name() {
        let base = [("DIQ010", 2.0), ("LBXGH", 5.4)];
        assert_eq!(prev(&person(2013, &base), "diabetes"), Some(false));
        assert_eq!(
            prev(
                &person(2013, &[("DIQ010", 2.0), ("LBXGH", 6.5)]),
                "diabetes"
            ),
            Some(true)
        );
        assert_eq!(
            prev(
                &person(2013, &[("DIQ010", 1.0), ("LBXGH", 5.0)]),
                "diabetes"
            ),
            Some(true)
        );
        // 2005 names the pill question DID070, every other cycle DIQ070.
        assert_eq!(
            prev(
                &person(2005, &[("DIQ010", 2.0), ("LBXGH", 5.0), ("DID070", 1.0)]),
                "diabetes"
            ),
            Some(true)
        );
        assert_eq!(
            prev(
                &person(2013, &[("DIQ010", 2.0), ("LBXGH", 5.0), ("DIQ070", 1.0)]),
                "diabetes"
            ),
            Some(true)
        );
        // Borderline is not yes; a refused answer leaves the label unknown unless HbA1c settles it.
        assert_eq!(
            prev(
                &person(2013, &[("DIQ010", 3.0), ("LBXGH", 5.0)]),
                "diabetes"
            ),
            Some(false)
        );
        assert_eq!(
            prev(
                &person(2013, &[("DIQ010", 9.0), ("LBXGH", 5.0)]),
                "diabetes"
            ),
            None
        );
        assert_eq!(
            prev(
                &person(2013, &[("DIQ010", 9.0), ("LBXGH", 7.0)]),
                "diabetes"
            ),
            Some(true)
        );
        assert_eq!(
            prev(&person(2013, &[("DIQ010", 2.0)]), "diabetes"),
            None,
            "no HbA1c, no negative"
        );
    }

    #[test]
    fn a_non_fasting_glucose_is_ignored() {
        let p = person(
            2013,
            &[
                ("DIQ010", 2.0),
                ("LBXGH", 5.0),
                ("LBXGLU", 140.0),
                ("PHAFSTHR", 3.0),
            ],
        );
        assert_eq!(diabetes_fasting(&Raw::read(&p)), None);
        let fasted = person(
            2013,
            &[
                ("DIQ010", 2.0),
                ("LBXGH", 5.0),
                ("LBXGLU", 140.0),
                ("PHAFSTHR", 9.0),
            ],
        );
        assert_eq!(diabetes_fasting(&Raw::read(&fasted)), Some(true));
        assert_eq!(
            prev(&fasted, "diabetes"),
            Some(false),
            "the primary label does not use glucose"
        );
    }

    #[test]
    fn hypertension_uses_the_mean_of_valid_readings_and_a_threshold_sensitivity() {
        let normal = [
            ("BPQ020", 2.0),
            ("BPXSY1", 128.0),
            ("BPXSY2", 132.0),
            ("BPXDI1", 78.0),
            ("BPXDI2", 82.0),
        ];
        assert_eq!(prev(&person(2009, &normal), "hypertension"), Some(false));
        assert_eq!(
            prev(&person(2009, &normal), "hypertension_130_80"),
            Some(true)
        );
        // A diastolic of 0 is "not detectable", not a pressure.
        let p = person(2009, &[("BPQ020", 2.0), ("BPXSY1", 120.0), ("BPXDI1", 0.0)]);
        assert_eq!(Raw::read(&p).dbp, None);
        assert_eq!(prev(&p, "hypertension"), None);
        assert_eq!(
            prev(&person(2009, &[("BPQ020", 1.0)]), "hypertension"),
            Some(true)
        );
    }

    #[test]
    fn creatinine_is_standardised_by_the_cycles_own_correction_before_the_equation() {
        assert!((standard_creatinine(1999, 1.0) - 1.160).abs() < 1e-9);
        assert!((standard_creatinine(2005, 1.0) - 0.962).abs() < 1e-9);
        assert_eq!(standard_creatinine(2013, 1.0), 1.0);
        // Evaluations of the published equation for a 50-year-old man and woman at 1.0 mg/dL.
        assert!((egfr_ckd_epi_2021(1.0, 50.0, false) - 92.0).abs() < 1.0);
        assert!((egfr_ckd_epi_2021(1.0, 50.0, true) - 68.6).abs() < 0.5);
        // 2001 names the creatinine LBDSCR.
        let p = person(
            2001,
            &[
                ("RIDAGEYR", 70.0),
                ("RIAGENDR", 2.0),
                ("LBDSCR", 1.5),
                ("URXUMA", 10.0),
                ("URXUCR", 100.0),
            ],
        );
        let r = Raw::read(&p);
        assert!(r.egfr.is_some_and(|e| e < 60.0), "{:?}", r.egfr);
        assert_eq!(r.acr, Some(10.0));
        assert_eq!(prev(&p, "kidney_markers"), Some(true));
    }

    #[test]
    fn kidney_markers_need_both_measurements_to_be_called_absent() {
        let one = person(
            2013,
            &[("RIDAGEYR", 40.0), ("RIAGENDR", 1.0), ("LBXSCR", 0.9)],
        );
        assert_eq!(prev(&one, "kidney_markers"), None);
        let albuminuria = person(2013, &[("URXUMA", 90.0), ("URXUCR", 100.0)]);
        assert_eq!(
            prev(&albuminuria, "kidney_markers"),
            Some(true),
            "acr 90 mg/g"
        );
    }

    #[test]
    fn anaemia_has_a_threshold_per_sex_and_is_unknown_in_pregnancy() {
        let man = [("RIAGENDR", 1.0), ("LBXHGB", 12.5)];
        assert_eq!(prev(&person(2013, &man), "anaemia"), Some(true));
        let woman = [("RIAGENDR", 2.0), ("LBXHGB", 12.5)];
        assert_eq!(prev(&person(2013, &woman), "anaemia"), Some(false));
        let pregnant = [("RIAGENDR", 2.0), ("LBXHGB", 10.5), ("RIDEXPRG", 1.0)];
        assert_eq!(prev(&person(2013, &pregnant), "anaemia"), None);
    }

    #[test]
    fn a_cholesterol_never_checked_is_not_told_but_an_unanswered_question_is_unknown() {
        let never = person(2013, &[("LBXTC", 190.0), ("BPQ060", 2.0)]);
        assert_eq!(prev(&never, "high_cholesterol"), Some(false));
        let skipped = person(2013, &[("LBXTC", 190.0)]);
        assert_eq!(prev(&skipped, "high_cholesterol"), None);
        assert_eq!(
            prev(&person(2013, &[("BPQ080", 1.0)]), "high_cholesterol"),
            Some(true)
        );
        assert_eq!(
            prev(&person(2013, &[("LBXTC", 250.0)]), "high_cholesterol"),
            Some(true)
        );
    }

    #[test]
    fn the_1999_kidney_question_has_another_name_and_unmeasured_cycles_stay_unknown() {
        let old = person(1999, &[("KIQ020", 1.0)]);
        assert_eq!(Raw::read(&old).told_kidney, Some(true));
        let new = person(2013, &[("KIQ022", 2.0)]);
        assert_eq!(Raw::read(&new).told_kidney, Some(false));
        // Osteoporosis is not asked in 2011 or 2015; depression and sleep start in 2005.
        let none = person(2011, &[]);
        assert_eq!(prev(&none, "osteoporosis"), None);
        assert_eq!(prev(&none, "depression"), None);
        assert_eq!(prev(&none, "sleep_problem"), None);
        let phq: Vec<(&'static str, f64)> = [
            "DPQ010", "DPQ020", "DPQ030", "DPQ040", "DPQ050", "DPQ060", "DPQ070", "DPQ080",
            "DPQ090",
        ]
        .into_iter()
        .map(|v| (v, 2.0))
        .collect();
        assert_eq!(
            prev(&person(2013, &phq), "depression"),
            Some(true),
            "18 points"
        );
        // One item missing means no score.
        assert_eq!(prev(&person(2013, &phq[..8]), "depression"), None);
    }

    #[test]
    fn undiagnosed_disease_is_the_measured_criterion_among_those_who_report_no_diagnosis() {
        let un = |p: &Fake, k: &str| undiagnosed(&Raw::read(p))[k];
        let high_a1c = person(2013, &[("DIQ010", 2.0), ("LBXGH", 6.8)]);
        assert_eq!(un(&high_a1c, "undiagnosed_diabetes"), Some(true));
        let told = person(2013, &[("DIQ010", 1.0), ("LBXGH", 6.8)]);
        assert_eq!(
            un(&told, "undiagnosed_diabetes"),
            None,
            "already diagnosed: not screened"
        );
        let treated = person(2013, &[("DIQ010", 2.0), ("DIQ050", 1.0), ("LBXGH", 6.8)]);
        assert_eq!(un(&treated, "undiagnosed_diabetes"), None);
        let bp = person(
            2013,
            &[("BPQ020", 2.0), ("BPXSY1", 150.0), ("BPXDI1", 85.0)],
        );
        assert_eq!(un(&bp, "undiagnosed_hypertension"), Some(true));
        let chol = person(2013, &[("BPQ060", 2.0), ("LBXTC", 250.0)]);
        assert_eq!(un(&chol, "undiagnosed_high_cholesterol"), Some(true));
        let kidney = person(
            2013,
            &[
                ("KIQ022", 2.0),
                ("RIDAGEYR", 60.0),
                ("RIAGENDR", 1.0),
                ("LBXSCR", 0.9),
                ("URXUMA", 5.0),
                ("URXUCR", 100.0),
            ],
        );
        assert_eq!(un(&kidney, "undiagnosed_kidney_markers"), Some(false));
    }
}
