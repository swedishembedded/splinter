// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement-learning runs whose
// training health is measured while they run, for its clients. If your team
// needs expertise in observable asynchronous policy optimisation, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The training health of every online training stage of a run: from its
//! product's report once it finished, else from the group log it keeps as it
//! runs (`work/<stage>/groups.jsonl`), so a running stage is observed live.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Serialize;
use serde_json::Value;
use splinter_core::factory::group::RolloutOutcome;

use super::records::RunRecords;
use crate::online::group_log::GROUPS_FILE;
use crate::online::health::VersionSeries;
use crate::run::online::ONLINE;

/// One online training stage's health.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OnlineStage {
    /// The stage.
    pub stage: String,
    /// Whether it finished (its report), else what its log holds so far.
    pub finished: bool,
    /// Each optimiser step's figures, once it finished.
    pub steps: Value,
    /// Each policy version's rollout signals, by version.
    pub versions: BTreeMap<u64, Value>,
    /// The run's own records of its log (a rollback's audit), in order.
    pub notes: Vec<Value>,
    /// The share of its scored groups whose rewards do not differ, over
    /// every version: what a task selector is meant to lower.
    pub zero_variance_share: Option<f64>,
}

/// The share of `versions`' groups whose rewards do not differ.
fn zero_variance_share(versions: &BTreeMap<u64, Value>) -> Option<f64> {
    let (zero, groups) = versions.values().fold((0.0, 0.0), |(z, g), v| {
        let n = v["groups"].as_f64().unwrap_or(0.0);
        (
            z + v["zero_variance_share"].as_f64().unwrap_or(0.0) * n,
            g + n,
        )
    });
    (groups > 0.0).then(|| zero / groups)
}

/// The per-version series and the run's own records the log at `path` holds.
fn from_log(path: &Path) -> Option<(VersionSeries, Vec<Value>)> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut series = VersionSeries::default();
    let mut seen = std::collections::HashSet::new();
    let mut notes = Vec::new();
    for line in text.lines() {
        let Ok(line) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let Some(group) = line["group"].as_str() else {
            notes.push(line);
            continue;
        };
        // A group's first line after it was scored holds its rollouts.
        if !seen.insert(group.to_string()) {
            continue;
        }
        let rollouts = line["rollouts"].as_array().cloned().unwrap_or_default();
        let outcomes: Vec<RolloutOutcome> = rollouts
            .iter()
            .filter_map(|r| serde_json::from_value(r["outcome"].clone()).ok())
            .collect();
        let failures = rollouts.iter().filter_map(|r| r["failures"].as_u64()).sum();
        series.add_group(&outcomes.iter().collect::<Vec<_>>(), failures);
    }
    Some((series, notes))
}

/// Every online training stage of the run `records` reads, in recipe order.
pub(super) fn observe(records: &RunRecords, run: &Path) -> Vec<OnlineStage> {
    records
        .stages()
        .iter()
        .filter(|s| s.uses == ONLINE)
        .filter_map(|s| {
            let log = from_log(&run.join("work").join(&s.id).join(GROUPS_FILE));
            let report = records
                .product(&s.id)
                .and_then(|p| p.json::<Value>("report.json"));
            let (series, notes) = log.unwrap_or_default();
            match report {
                Some(report) => {
                    let versions: BTreeMap<u64, Value> =
                        serde_json::from_value(report["health"]["versions"].clone())
                            .unwrap_or_default();
                    Some(OnlineStage {
                        stage: s.id.clone(),
                        finished: true,
                        steps: report["health"]["steps"].clone(),
                        zero_variance_share: zero_variance_share(&versions),
                        versions,
                        notes,
                    })
                }
                None if !series.versions.is_empty() || !notes.is_empty() => {
                    let versions = series.summary();
                    Some(OnlineStage {
                        stage: s.id.clone(),
                        finished: false,
                        steps: Value::Null,
                        zero_variance_share: zero_variance_share(&versions),
                        versions,
                        notes,
                    })
                }
                None => None,
            }
        })
        .collect()
}
