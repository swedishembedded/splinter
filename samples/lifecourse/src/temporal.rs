// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A secondary analysis of calendar shift, not one of the pre-registered
//! criteria: an arm trained on the earlier cycles and scored on the later
//! ones, outside the locked test.

use std::path::Path;

use anyhow::{bail, Context, Result};
use splinter_sdk::model::timeline::Subject;

use crate::commands::{frozen, nhanes_terms, require_training, run_one, write_run};
use crate::experiment::Arm;

/// The cycle a subject was examined in (its `source`, `nhanes-<year>`).
fn cycle_year(s: &Subject) -> Option<u16> {
    s.source.strip_prefix("nhanes-")?.parse().ok()
}

/// A secondary analysis of calendar shift, outside the pre-registered
/// criteria: train `arm` on the non-locked subjects of the cycles examined
/// before `split`, score the non-locked subjects of `split` and later. The
/// locked test is never read. Later cycles are followed for less time, so
/// only the shorter horizons are scored; every metric is restricted to the
/// cycles that support it, as everywhere else.
pub fn temporal(data: &Path, arm: Arm, seed: u64, split: u16) -> Result<()> {
    require_training(&nhanes_terms())?;
    let f = frozen(data)?;
    let locked: std::collections::HashSet<&str> =
        f.partition.locked.iter().map(String::as_str).collect();
    let mut ids: Vec<(&str, u16)> = f
        .subjects
        .iter()
        .filter(|(id, _)| !locked.contains(id.as_str()))
        .map(|(id, s)| {
            cycle_year(s)
                .map(|y| (id.as_str(), y))
                .with_context(|| format!("{id}: no cycle in source {:?}", s.source))
        })
        .collect::<Result<_>>()?;
    ids.sort_unstable();
    let train: Vec<&str> = ids
        .iter()
        .filter(|(_, y)| *y < split)
        .map(|(id, _)| *id)
        .collect();
    let test: Vec<&str> = ids
        .iter()
        .filter(|(_, y)| *y >= split)
        .map(|(id, _)| *id)
        .collect();
    if train.is_empty() || test.is_empty() {
        bail!(
            "split {split} leaves {} training and {} test subjects",
            train.len(),
            test.len()
        );
    }
    let name = format!("{}-s{seed}-temporal-{split}.json", arm.name());
    let run = run_one(&f, arm, seed, &train, &test, false)?.run;
    write_run(data, &name, &run)?;
    println!(
        "{name}: trained on {} subjects before {split}, scored {}: {}",
        train.len(),
        test.len(),
        serde_json::to_string_pretty(&run.metrics)?
    );
    Ok(())
}
