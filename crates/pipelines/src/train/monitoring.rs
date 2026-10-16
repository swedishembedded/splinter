// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning runs that watch their own
// held-out loss and say plainly when they cannot, for its clients. If your
// team needs expertise in overfit control for small-data fine-tunes, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Carving the monitoring set out of the records a run trains on.
//!
//! The monitoring set is whole groups of the training records. When what is
//! left to train on is one group - the held-out half took the other, or the
//! data was always one text - there is no whole group to carve, and dividing
//! one would put a print of the monitored text into training. The run then
//! trains without monitoring and says why, where a refusal here would throw
//! away the rollouts and teacher calls that produced the data.

use std::path::{Path, PathBuf};

use splinter_data::holdout::{monitor_split_named, MonitorSplit, FIT_FILE, MONITOR_FILE};
use splinter_data::ViewError;

use super::{concatenate, TrainPlan};
use splinter_orchestrator::error::OrchestratorError;

/// The monitoring set a run scores when it is made of several files: the
/// training families' monitoring records, the writer's text of those
/// families and the rehearsal's records together.
const MONITOR_ALL_FILE: &str = "monitor_all.jsonl";

/// What a supervised run fits and what it monitors on.
pub(super) struct WatchPlan {
    /// The records to fit instead of the training half; `None` fits all of it.
    pub(super) fit: Option<PathBuf>,
    /// The monitoring records; `None` monitors nothing.
    pub(super) monitor: Option<PathBuf>,
    /// Steps between evaluations: the plan's when there is a monitoring set,
    /// else 0.
    pub(super) eval_every: u32,
    /// Why monitoring that was asked for was not done, for the candidate's
    /// record.
    pub(super) skipped: Option<String>,
}

/// The monitoring of a run over `train`, the training half of its split.
///
/// The monitoring set is carved from the training half: the held-out half is
/// the gate's and the exam's evidence, and a step chosen on it would be
/// chosen on what it is then judged by. It is the training families'
/// monitoring records, the writer's text of those families and the
/// rehearsal's records together: the step selected is the one that
/// generalises best on the mixture the run trains - the writer's text
/// included, which is most of the tokens of a persona run - not on the
/// dialogues alone. When no whole group can be carved the run is unmonitored
/// and the rehearsal's monitoring records, with no set to join, are not
/// scored.
pub(super) fn plan_monitoring(
    plan: &TrainPlan,
    train: &Path,
) -> Result<WatchPlan, OrchestratorError> {
    let mut skipped: Vec<String> = Vec::new();
    let mut watch = WatchPlan {
        fit: None,
        monitor: None,
        eval_every: 0,
        skipped: None,
    };
    if plan.eval_every > 0 {
        match carve_monitor(
            train,
            &plan.dir,
            plan.monitor_share,
            (FIT_FILE, MONITOR_FILE),
        )? {
            Monitoring::Carved(m) => {
                let parts: Vec<&Path> = std::iter::once(m.monitor.as_path())
                    .chain(m.monitor_text.as_deref())
                    .chain(plan.rehearsal.as_ref().and_then(|r| r.monitor.as_deref()))
                    .collect();
                watch.monitor = Some(if parts.len() == 1 {
                    m.monitor.clone()
                } else {
                    concatenate(&parts, &plan.dir.join(MONITOR_ALL_FILE))?
                });
                watch.eval_every = plan.eval_every;
                watch.fit = Some(m.fit);
            }
            Monitoring::Skipped(why) => skipped.push(why),
        }
    }
    skipped.extend(
        plan.rehearsal
            .iter()
            .filter_map(|r| r.monitor_skipped.clone()),
    );
    watch.skipped = (!skipped.is_empty()).then(|| skipped.join("; "));
    Ok(watch)
}

/// The monitoring set carved from a file of training records, or the reason
/// none could be.
pub(super) enum Monitoring {
    /// The records split into the ones fitted and the ones monitored.
    Carved(MonitorSplit),
    /// No whole group could be set aside; the reason, as the candidate
    /// records it.
    Skipped(String),
}

/// `train`'s records split by [`monitor_split_named`] under `names`. Only
/// [`ViewError::NoGroupToHoldOut`] degrades to [`Monitoring::Skipped`]; every
/// other failure (too few records, an unreadable file) is the run's to refuse.
pub(super) fn carve_monitor(
    train: &Path,
    dir: &Path,
    share: f64,
    names: (&str, &str),
) -> Result<Monitoring, OrchestratorError> {
    match monitor_split_named(train, dir, share, names) {
        Ok(split) => Ok(Monitoring::Carved(split)),
        Err(cause @ ViewError::NoGroupToHoldOut { .. }) => Ok(Monitoring::Skipped(format!(
            "no monitoring set could be set aside, so that part of the run is not monitored: {cause}"
        ))),
        Err(other) => Err(other.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn records(groups: &[&str]) -> String {
        groups
            .iter()
            .enumerate()
            .map(|(n, group)| {
                format!(
                    r#"{{"messages":[],"metadata":{{"experiences":["blake3:{n:064x}"],"group":"{group}"}}}}"#
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn carve(name: &str, groups: &[&str]) -> Monitoring {
        let dir = std::env::temp_dir().join(format!("monitoring-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let train = dir.join("train.jsonl");
        std::fs::write(&train, records(groups)).unwrap();
        let outcome = carve_monitor(&train, &dir, 0.1, ("fit.jsonl", "monitor.jsonl")).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        outcome
    }

    #[test]
    fn what_is_left_of_one_group_is_trained_unmonitored_with_the_reason_recorded() {
        let Monitoring::Skipped(why) = carve("one", &["only"; 8]) else {
            panic!("one group has nothing to set aside whole");
        };
        assert!(
            why.contains("not monitored") && why.contains("1 independent source text"),
            "{why}"
        );
    }

    #[test]
    fn several_groups_are_carved_as_before() {
        let groups: Vec<String> = (0..12).map(|n| format!("g{n}")).collect();
        let refs: Vec<&str> = groups.iter().map(String::as_str).collect();
        assert!(matches!(carve("many", &refs), Monitoring::Carved(_)));
    }
}
