// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning loops whose every setting is proven
// to move what it is meant to move, for its clients. If your team needs
// expertise in observing and steering agent training loops, you can procure
// our services by sending an email to info@swedishembedded.com.

//! A probe of one knob: the run's recipe with that knob alone set to another
//! value, run as a fork of the run (so every stage upstream of the knob is
//! reused and only what reads it is computed), and both runs observed. The
//! report says whether the state variable the knob declares it moves did
//! move, in the generation the knob's stage feeds, and whether each of that
//! generation's evaluations followed: the candidate's pass rate in the probe
//! against the control's, with the paired bootstrap interval of the mean
//! per-task difference (probe minus control) over the tasks both evaluated.
//!
//! A variable *moved* when its absolute change exceeds the knob's
//! `min_change` (any change when unset); a knob whose variable did not move
//! is a defect of the knob. Capability *followed* when that interval lies
//! above zero.

use std::path::Path;

use serde::Serialize;
use serde_json::Value;
use splinter_eval::pilot::Share;
use splinter_model::stats::Interval;

use crate::knobs::{perturb, KnobSpec};

use super::capability::paired;
use super::data::Policy;
use super::records::RunRecords;
use super::{observe, Generation, Options, RunObservation};

/// Whether the knob's variable moved.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// It changed by more than the knob's least change.
    Moved,
    /// It did not: the knob does not control what it says it does.
    DidNotMove,
    /// One of the runs did not measure it.
    Unmeasured,
}

/// One evaluation of the generation, control against probe.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CapabilityChange {
    /// The evaluation stage.
    pub evaluation: String,
    /// The purpose its release gave it, `None` for a measurement.
    pub purpose: Option<String>,
    /// Its task set's label.
    pub task_set: String,
    /// The control candidate's pass rate.
    pub control: Option<Share>,
    /// The probed candidate's pass rate.
    pub probed: Option<Share>,
    /// The paired interval of the mean per-task difference, probe minus
    /// control.
    pub interval: Option<Interval>,
    /// Tasks the interval rests on.
    pub tasks: usize,
    /// Whether the interval lies above zero; `None` without one.
    pub followed: Option<bool>,
}

/// What a probe of one knob found.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ProbeReport {
    /// The knob.
    pub knob: String,
    /// The parameter it sets.
    pub set: String,
    /// The value the probe set it to.
    pub value: Value,
    /// The state variable it declares it moves.
    pub moves: String,
    /// The generation read, from 1.
    pub generation: usize,
    /// The variable in the control run.
    pub control: Option<f64>,
    /// The variable in the probe.
    pub probed: Option<f64>,
    /// Probe minus control.
    pub change: Option<f64>,
    /// The least change that counts.
    pub min_change: f64,
    /// Whether it moved.
    pub verdict: Verdict,
    /// The knob is a defect: its variable did not move.
    pub defect: bool,
    /// Each evaluation, control against probe.
    pub capability: Vec<CapabilityChange>,
}

/// The state variable `name` (one of [`crate::knobs::VARIABLES`]) of
/// `generation`'s first candidate.
#[must_use]
pub fn variable(generation: &Generation, name: &str) -> Option<f64> {
    let candidate = generation.candidates.first()?;
    let training = candidate.training.as_ref();
    let data = candidate.data.as_ref();
    let pooled = |policy: Policy| -> Option<f64> {
        let (k, n) = data?
            .sources
            .iter()
            .filter(|s| s.policy == policy)
            .filter_map(|s| s.pass_rate)
            .fold((0, 0), |a, p| (a.0 + p.successes, a.1 + p.total));
        (n > 0).then(|| k as f64 / n as f64)
    };
    match name {
        "training.epochs_at_selected" => training?.epochs_at_selected,
        "training.epochs_completed" => training?.epochs_completed,
        "training.steps" => training?.steps.map(|v| v as f64),
        "training.selected_step" => training?.selected_step.map(|v| v as f64),
        "training.draws_per_epoch" => training?.draws_per_epoch,
        "training.tokens" => training?.tokens.map(|v| v as f64),
        "data.train_records" => data.map(|d| d.train.records as f64),
        "data.own_share_records" => data?.own_share_records,
        "data.own_share_characters" => data?.own_share_characters,
        "data.validation_own_share_records" => data?.validation_own_share_records,
        "data.frontier_share" => data?.frontier.as_ref()?.frontier_share,
        "data.action_sequences" => data.map(|d| d.action_sequences as f64),
        "data.own_pass_rate" => pooled(Policy::Own),
        "data.other_pass_rate" => pooled(Policy::Other),
        _ => None,
    }
}

/// The generation (index) of `observation` that the stage `stage` feeds: the
/// first whose release, candidates, datasets, sources or evaluations read it.
fn generation_of(records: &RunRecords, observation: &RunObservation, stage: &str) -> Option<usize> {
    let downstream = records.downstream(stage);
    observation.generations.iter().position(|g| {
        let mut stages = vec![g.release.stage.as_str()];
        for c in &g.candidates {
            stages.push(&c.stage);
            if let Some(d) = &c.data {
                stages.push(&d.stage);
                stages.extend(d.sources.iter().map(|s| s.stage.as_str()));
            }
        }
        stages.extend(g.evaluations.iter().map(|e| e.stage.as_str()));
        stages.iter().any(|s| downstream.contains(*s))
    })
}

/// Why a probe could not be made.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct ProbeError(pub String);

/// The probe report of `knob` set to `value`, from the observation of the
/// control run (`control`, in `control_dir`) and of the probe (`probed`).
///
/// # Errors
/// The knob's stage feeds no generation of the control.
pub fn compare(
    knob: &str,
    spec: &KnobSpec,
    value: &Value,
    control_dir: &Path,
    control: &RunObservation,
    probed: &RunObservation,
    seed: u64,
) -> Result<ProbeReport, ProbeError> {
    let stage = spec
        .target()
        .map(|(s, _)| s.to_owned())
        .ok_or_else(|| ProbeError(format!("knob {knob}: set {:?} is not STAGE.PATH", spec.set)))?;
    let records = RunRecords::open(control_dir).map_err(|e| ProbeError(e.to_string()))?;
    let index = generation_of(&records, control, &stage).ok_or_else(|| {
        ProbeError(format!(
            "knob {knob}: stage {stage} feeds no generation of the run"
        ))
    })?;
    let (a, b) = (
        control.generations.get(index),
        probed.generations.get(index),
    );
    let before = a.and_then(|g| variable(g, &spec.moves));
    let after = b.and_then(|g| variable(g, &spec.moves));
    let change = before.zip(after).map(|(x, y)| y - x);
    let min_change = spec.min_change.unwrap_or(0.0);
    let verdict = match change {
        None => Verdict::Unmeasured,
        Some(d) if d.abs() > min_change => Verdict::Moved,
        Some(_) => Verdict::DidNotMove,
    };
    let capability = a
        .map(|g| &g.evaluations[..])
        .unwrap_or_default()
        .iter()
        .map(|e| {
            let other = b.and_then(|g| g.evaluations.iter().find(|o| o.stage == e.stage));
            let (interval, tasks) = other
                .map(|o| paired(&e.candidate_tasks, &o.candidate_tasks, seed))
                .unwrap_or((None, 0));
            CapabilityChange {
                evaluation: e.stage.clone(),
                purpose: e.purpose.clone(),
                task_set: e.task_set.label.clone(),
                control: e.candidate_arm.pass_rate,
                probed: other.and_then(|o| o.candidate_arm.pass_rate),
                interval,
                tasks,
                followed: interval.map(|i| i.above_zero()),
            }
        })
        .collect();
    Ok(ProbeReport {
        knob: knob.to_owned(),
        set: spec.set.clone(),
        value: value.clone(),
        moves: spec.moves.clone(),
        generation: index + 1,
        control: before,
        probed: after,
        change,
        min_change,
        verdict,
        defect: verdict == Verdict::DidNotMove,
        capability,
    })
}

/// Probes `knob` of the run `control` at `value` into `out`: its recipe with
/// that knob alone set, resolved by `resolve`, forked from the run and
/// carried out by `execute`, then both runs observed and compared.
///
/// # Errors
/// The run's recipe cannot be read, declares no such knob, or the probe
/// cannot be resolved, forked, run or observed.
pub fn probe<R>(
    control: &Path,
    out: &Path,
    knob: &str,
    value: &Value,
    options: &Options,
    resolve: impl FnOnce(&Path, &str) -> Result<R, String>,
    execute: impl FnOnce(&R, &Path) -> Result<(), String>,
) -> Result<ProbeReport, ProbeError> {
    let fail = |e: String| ProbeError(e);
    let (repository, text) = crate::run::started_with(control).map_err(|e| fail(e.to_string()))?;
    let recipe = crate::graph::parse(&text).map_err(|e| fail(e.0.join("; ")))?;
    let spec = recipe
        .knobs
        .get(knob)
        .cloned()
        .ok_or_else(|| fail(format!("the run's recipe declares no knob {knob:?}")))?;
    let perturbed = perturb(&text, knob, value).map_err(fail)?;
    let resolved = resolve(&repository, &perturbed).map_err(fail)?;
    crate::run::fork(control, out).map_err(|e| fail(e.to_string()))?;
    execute(&resolved, out).map_err(fail)?;
    let before = observe(control, options).map_err(|e| fail(e.to_string()))?;
    let after = observe(out, options).map_err(|e| fail(e.to_string()))?;
    compare(knob, &spec, value, control, &before, &after, options.seed)
}
