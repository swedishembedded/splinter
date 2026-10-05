// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! NHANES participants as `timeline-v1` subjects.
//!
//! The prediction time is the examination; the subject's clock is age in
//! years. Only what was known at the examination is input:
//!
//! - every concept in [`crate::concepts`] measured or answered then,
//!   smoking as never / former / current, blood pressure as the mean of the
//!   readings taken;
//! - eating-time features from the 24-hour recalls ([`crate::diet`]);
//! - history: the ages at which conditions were first diagnosed and smoking
//!   started (events before entry), and recalled body weights a year ago, ten
//!   years ago, at 25 and at the heaviest (observations before entry).
//!
//! The outcome is death in the public-use linkage, through 2019, in three
//! cause groups that compete: cardiovascular (heart disease and
//! cerebrovascular, recodes 1 and 5), cancer (2) and every other cause.
//! The observation window runs from the examination to death or the end of
//! follow-up. Participants not eligible for linkage (under 18) are excluded
//! and counted, as are the eligible without a follow-up time.
//!
//! The sampling weight is the examination weight pooled over the ten cycles
//! by NCHS's rule: `2/10 * WTMEC4YR` for 1999-2002, `1/10 * WTMEC2YR` after.

use std::collections::BTreeMap;

use serde::Serialize;
use splinter_sdk::model::timeline::{AtRisk, Event, Observation, Subject, Value};

use crate::concepts::{AGE_RANGE, CATEGORIES, EXAM, ONSETS, POUNDS_RANGE, POUNDS_TO_KG};
use crate::diet::DietDay;
use crate::nhanes::{Cycle, Mortality};

/// The outcome codes, in the order the model is trained on them.
pub const CODES: [&str; 3] = ["death:cvd", "death:cancer", "death:other"];

/// Why a participant is not a subject.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Exclusion {
    /// Not in the mortality file at all.
    NoMortalityRecord,
    /// Not eligible for linkage (under 18 at screening, or ineligible).
    NotEligible,
    /// Eligible, but no follow-up time from the examination (not examined).
    NoFollowUp,
    /// No positive examination weight.
    NoWeight,
}

/// The design of one subject: what a design-based interval resamples.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Design {
    /// The subject's id.
    pub subject_id: String,
    /// Cycle start year.
    pub cycle: u16,
    /// Masked variance stratum.
    pub stratum: u64,
    /// Masked variance PSU within the stratum.
    pub psu: u64,
}

fn number(t: f64, var: &str, value: f64) -> Observation {
    Observation {
        t,
        var: var.into(),
        value: Value::Number(value),
    }
}

fn category(t: f64, var: &str, level: &str) -> Observation {
    Observation {
        t,
        var: var.into(),
        value: Value::Category(level.into()),
    }
}

/// The first of `vars` the cycle has a value for, scaled, inside `valid`.
fn first_valid(c: &Cycle, seqn: u64, vars: &[(&str, f64)], valid: (f64, f64)) -> Option<f64> {
    vars.iter()
        .find_map(|(v, f)| c.num(v, seqn).map(|x| x * f))
        .filter(|x| (valid.0..=valid.1).contains(x))
}

fn mean_of(c: &Cycle, seqn: u64, vars: &[&str], valid: (f64, f64)) -> Option<f64> {
    let vals: Vec<f64> = vars
        .iter()
        .filter_map(|v| c.num(v, seqn))
        .filter(|x| (valid.0..=valid.1).contains(x))
        .collect();
    (!vals.is_empty()).then(|| vals.iter().sum::<f64>() / vals.len() as f64)
}

/// The pooled ten-cycle examination weight.
pub fn pooled_weight(c: &Cycle, seqn: u64) -> Option<f64> {
    let w = if c.start <= 2001 {
        c.num("WTMEC4YR", seqn).map(|w| 0.2 * w)
    } else {
        c.num("WTMEC2YR", seqn).map(|w| 0.1 * w)
    };
    w.filter(|w| *w > 0.0)
}

/// One participant as a subject, or why not.
/// `extra` holds per-person values read from multi-row files, as
/// `(concept, value)` observations at the examination.
pub fn subject(
    c: &Cycle,
    seqn: u64,
    m: Option<&Mortality>,
    diet: &[DietDay],
    extra: &[(&str, f64)],
) -> Result<(Subject, Design), Exclusion> {
    let m = m.ok_or(Exclusion::NoMortalityRecord)?;
    if !m.eligible {
        return Err(Exclusion::NotEligible);
    }
    let months = m.months_from_exam.ok_or(Exclusion::NoFollowUp)?;
    let weight = pooled_weight(c, seqn).ok_or(Exclusion::NoWeight)?;
    let age_years = c.num("RIDAGEYR", seqn).ok_or(Exclusion::NotEligible)?;
    // Age is reported in whole years: the exam falls half a year in on average.
    let entry = age_years + 0.5;
    let exit = entry + months / 12.0;
    let mut obs = vec![number(entry, "age", age_years)];
    for n in EXAM {
        if let Some(v) = first_valid(c, seqn, n.vars, n.valid) {
            obs.push(number(entry, n.name, v));
        }
    }
    if let Some(v) = mean_of(
        c,
        seqn,
        &["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"],
        (60.0, 280.0),
    ) {
        obs.push(number(entry, "sbp", v));
    }
    // A diastolic reading of 0 is NHANES's "not detectable", not a pressure.
    if let Some(v) = mean_of(
        c,
        seqn,
        &["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"],
        (20.0, 160.0),
    ) {
        obs.push(number(entry, "dbp", v));
    }
    for k in CATEGORIES {
        let level = k
            .vars
            .iter()
            .find_map(|v| c.num(v, seqn))
            .and_then(|code| k.levels.iter().find(|(c, _)| *c == code))
            .map(|(_, l)| *l);
        if let Some(l) = level {
            obs.push(category(entry, k.name, l));
        }
    }
    let smoking = match (c.num("SMQ020", seqn), c.num("SMQ040", seqn)) {
        (Some(2.0), _) => Some("never"),
        (Some(1.0), Some(3.0)) => Some("former"),
        (Some(1.0), Some(1.0) | Some(2.0)) => Some("current"),
        _ => None,
    };
    if let Some(s) = smoking {
        obs.push(category(entry, "smoking", s));
    }
    for (name, v) in extra {
        obs.push(number(entry, name, *v));
    }
    if let Some(score) = phq9(c, seqn) {
        obs.push(number(entry, "phq9", score));
    }
    crate::diet::observations(diet, entry, &mut obs);
    // Recalled weights, placed at the time they held.
    let pounds = |v: &str| {
        c.num(v, seqn)
            .filter(|x| (POUNDS_RANGE.0..=POUNDS_RANGE.1).contains(x))
            .map(|x| x * POUNDS_TO_KG)
    };
    if let Some(w) = pounds("WHD050") {
        obs.push(number(entry - 1.0, "weight_kg", w));
    }
    if let Some(w) = pounds("WHD110").filter(|_| entry - 10.0 >= 16.0) {
        obs.push(number(entry - 10.0, "weight_kg", w));
    }
    if let Some(w) = pounds("WHD120").filter(|_| entry > 25.5) {
        obs.push(number(25.0, "weight_kg", w));
    }
    let heaviest_age = c.num("WHQ150", seqn).filter(|a| (16.0..entry).contains(a));
    if let (Some(w), Some(a)) = (pounds("WHD140"), heaviest_age) {
        obs.push(number(a, "weight_kg", w));
    }
    let mut events = Vec::new();
    for (code, vars) in ONSETS {
        let age = vars
            .iter()
            .find_map(|v| c.num(v, seqn))
            .filter(|a| (AGE_RANGE.0..=AGE_RANGE.1).contains(a));
        if let Some(a) = age.filter(|a| *a < entry) {
            events.push(Event {
                t: a,
                code: (*code).into(),
            });
        }
    }
    if m.died {
        let code = match m.cause {
            Some(1) | Some(5) => CODES[0],
            Some(2) => CODES[1],
            _ => CODES[2],
        };
        events.push(Event {
            t: exit,
            code: code.into(),
        });
    }
    let s = Subject {
        subject_id: format!("nhanes-{}-{seqn}", c.start),
        group_id: None,
        weight,
        source: format!("nhanes-{}", c.start),
        entry,
        calendar_at_entry: c.start as f64 + 1.0,
        observations: obs,
        events,
        at_risk: vec![AtRisk {
            code: "*".into(),
            from: entry,
            to: exit,
        }],
    };
    let design = Design {
        subject_id: s.subject_id.clone(),
        cycle: c.start,
        stratum: c.num("SDMVSTRA", seqn).map_or(0, |x| x as u64),
        psu: c.num("SDMVPSU", seqn).map_or(0, |x| x as u64),
    };
    Ok((s, design))
}

/// The PHQ-9 depression score, when all nine items were answered (0-3 each).
fn phq9(c: &Cycle, seqn: u64) -> Option<f64> {
    let items = [
        "DPQ010", "DPQ020", "DPQ030", "DPQ040", "DPQ050", "DPQ060", "DPQ070", "DPQ080", "DPQ090",
    ];
    items
        .iter()
        .map(|v| c.num(v, seqn).filter(|x| (0.0..=3.0).contains(x)))
        .sum()
}

/// Counts for the build report.
#[derive(Clone, Debug, Default, Serialize)]
pub struct CycleCounts {
    /// Participants in the demographics file.
    pub participants: usize,
    /// Subjects written.
    pub subjects: usize,
    /// Excluded, by reason.
    pub excluded: BTreeMap<String, usize>,
    /// Deaths by code.
    pub deaths: BTreeMap<String, usize>,
    /// Subjects with at least one recall day.
    pub with_diet: usize,
}
