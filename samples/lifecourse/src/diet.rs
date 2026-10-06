// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Eating-time features from the 24-hour dietary recalls.
//!
//! Every recalled food carries the clock time it was eaten (seconds since
//! midnight: `DRD020` in 1999-2002, `DR1_020`/`DR2_020` after) and its
//! energy. A recall covers midnight to midnight of the day before the
//! interview, so food eaten before 04:00 is counted as late on that day (24
//! hours added) rather than as the next morning's breakfast. Only items with
//! energy count as eating; a recall day whose status is not "reliable and
//! meets the minimum criteria" (`DR1DRSTZ`/`DR2DRSTZ`, `DRDDRSTZ` in
//! 2001-2002, `DRDDRSTS` in 1999-2000, = 1) is dropped. Two recall days (2003 onwards) are averaged.
//!
//! These features are the interpretable summaries; the model sees them as
//! observations at the examination like any other measurement.

use std::collections::HashMap;

use splinter_sdk::knowledge::tabular::Table;
use splinter_sdk::model::timeline::{Observation, Value};

use crate::nhanes::Cycle;

/// Hours added to food eaten in the small hours of the recall day.
const LATE_NIGHT_BEFORE_H: f64 = 4.0;
/// From this hour, energy counts as late.
const LATE_FROM_H: f64 = 20.0;

/// One recall day's eating times.
#[derive(Clone, Debug, PartialEq)]
pub struct DietDay {
    /// Hour of the first eating occasion.
    pub first_h: f64,
    /// Hour of the last.
    pub last_h: f64,
    /// Distinct clock times with energy.
    pub events: usize,
    /// Total energy.
    pub kcal: f64,
    /// Share of the energy eaten from [`LATE_FROM_H`].
    pub late_share: f64,
}

fn day_of(times_s: &[f64], kcal: &[f64]) -> Option<DietDay> {
    let mut items: Vec<(f64, f64)> = times_s
        .iter()
        .zip(kcal)
        .filter(|(_, k)| **k > 0.0)
        .map(|(t, k)| {
            let h = t / 3600.0;
            (if h < LATE_NIGHT_BEFORE_H { h + 24.0 } else { h }, *k)
        })
        .collect();
    if items.is_empty() {
        return None;
    }
    items.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: f64 = items.iter().map(|i| i.1).sum();
    let late: f64 = items
        .iter()
        .filter(|i| i.0 >= LATE_FROM_H)
        .map(|i| i.1)
        .sum();
    let mut times: Vec<f64> = items.iter().map(|i| i.0).collect();
    times.dedup();
    Some(DietDay {
        first_h: items[0].0,
        last_h: items[items.len() - 1].0,
        events: times.len(),
        kcal: total,
        late_share: late / total,
    })
}

/// Per participant, the recall days of `table` whose status is reliable.
fn days_from(
    c: &Cycle,
    table: &Table,
    time: &str,
    kcal: &str,
    status: &[&str],
    out: &mut HashMap<u64, Vec<DietDay>>,
) {
    let (Some(seqn), Some(t), Some(k)) = (
        table.numeric("SEQN"),
        table.numeric(time),
        table.numeric(kcal),
    ) else {
        return;
    };
    let mut by: HashMap<u64, (Vec<f64>, Vec<f64>)> = HashMap::new();
    for i in 0..table.rows {
        if let (Some(s), Some(ti), Some(ki)) = (seqn[i], t[i], k[i]) {
            let e = by.entry(s as u64).or_default();
            e.0.push(ti);
            e.1.push(ki);
        }
    }
    for (s, (ts, ks)) in by {
        if status.iter().find_map(|v| c.num(v, s)) != Some(1.0) {
            continue;
        }
        if let Some(d) = day_of(&ts, &ks) {
            out.entry(s).or_default().push(d);
        }
    }
}

/// Every participant's reliable recall days in the cycle.
pub fn days(c: &Cycle) -> HashMap<u64, Vec<DietDay>> {
    let mut out = HashMap::new();
    if let Some(t) = c.multi.get("DRXIFF") {
        // The status variable is DRDDRSTS in 1999-2000 and DRDDRSTZ in 2001-2002.
        days_from(
            c,
            t,
            "DRD020",
            "DRXIKCAL",
            &["DRDDRSTZ", "DRDDRSTS"],
            &mut out,
        );
    }
    if let Some(t) = c.multi.get("DR1IFF") {
        days_from(c, t, "DR1_020", "DR1IKCAL", &["DR1DRSTZ"], &mut out);
    }
    if let Some(t) = c.multi.get("DR2IFF") {
        days_from(c, t, "DR2_020", "DR2IKCAL", &["DR2DRSTZ"], &mut out);
    }
    out
}

/// The averaged features of `days` as observations at `entry`.
pub fn observations(days: &[DietDay], entry: f64, out: &mut Vec<Observation>) {
    if days.is_empty() {
        return;
    }
    let n = days.len() as f64;
    let mean = |f: &dyn Fn(&DietDay) -> f64| days.iter().map(f).sum::<f64>() / n;
    let mut push = |var: &str, v: f64| {
        out.push(Observation {
            t: entry,
            var: var.into(),
            value: Value::Number(v),
            unit: None,
        })
    };
    push("diet:first_meal_h", mean(&|d| d.first_h));
    push("diet:last_meal_h", mean(&|d| d.last_h));
    push("diet:eating_window_h", mean(&|d| d.last_h - d.first_h));
    push("diet:eating_events", mean(&|d| d.events as f64));
    push("diet:energy_kcal", mean(&|d| d.kcal));
    push("diet:late_energy_share", mean(&|d| d.late_share));
    push("diet:recall_days", n);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_day_counts_the_small_hours_as_late_and_ignores_zero_energy() {
        // 07:30 breakfast, 12:00 lunch, 21:00 dinner, 01:00 snack, water at 15:00.
        let t = [27000.0, 43200.0, 75600.0, 3600.0, 54000.0];
        let k = [400.0, 600.0, 800.0, 200.0, 0.0];
        let d = day_of(&t, &k).unwrap();
        assert_eq!(d.first_h, 7.5);
        assert_eq!(d.last_h, 25.0, "the 01:00 snack ends the day");
        assert_eq!(d.events, 4, "water is not eating");
        assert_eq!(d.kcal, 2000.0);
        assert!(
            (d.late_share - 0.5).abs() < 1e-12,
            "dinner and the snack are late"
        );
        assert!(day_of(&[1.0], &[0.0]).is_none());
    }
}
