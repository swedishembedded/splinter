// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reinforcement learning on tasks with
// verifiable rewards, choosing only the tasks a policy can learn from, for
// its clients. If your team needs expertise in RL from verifiable rewards,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! `dataset.rlvr`: the tasks a reinforcement learner can learn from, as an
//! executable collection with their own verifiers.
//!
//! An online trainer rolls a task out again and grades every rollout by the
//! task's verifier; a task the policy always solves, or never solves, gives
//! each of its groups the same reward and so no gradient. The attempts an
//! attempt stage made measure that: a task is kept when its counted
//! attempts number at least `min_attempts` and its pass rate lies strictly
//! inside `(0, 1)`, or inside `[min_pass_rate, max_pass_rate]` when either
//! is given. The product holds each kept task's folder (instruction, tests,
//! environment, as the run pinned it) at its own path, which `tasks.pin`
//! pins as a collection, and `rlvr.json`, every attempted task's counts,
//! pass rate and whether it was kept.

use std::collections::BTreeMap;

use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use splinter_core::factory::execution::Isolation;
use splinter_core::factory::operation::{
    ArtifactKind, OperationDescriptor, Port, Reproducibility, OPERATION_SCHEMA,
};

use crate::execute::{Call, Execute, Outcome};
use crate::registry::OperationSpec;
use crate::run::rollout::trials_of;
use crate::run::tasks::tasks_of;
use crate::tree::link_tree;

/// The operation's name.
pub const RLVR: &str = "dataset.rlvr";

fn two() -> u32 {
    2
}

/// `dataset.rlvr`'s parameters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RlvrParams {
    /// The fewest counted attempts a task's pass rate may rest on.
    #[serde(default = "two")]
    pub min_attempts: u32,
    /// The lowest pass rate kept; above 0 when unset.
    #[serde(default)]
    pub min_pass_rate: Option<f64>,
    /// The highest pass rate kept; below 1 when unset.
    #[serde(default)]
    pub max_pass_rate: Option<f64>,
}

fn params(value: &Value) -> Result<RlvrParams, String> {
    serde_json::from_value(value.clone()).map_err(|e| e.to_string())
}

/// What the attempts measured of one task.
#[derive(Serialize)]
struct Measured {
    attempts: u32,
    passes: u32,
    pass_rate: f64,
    kept: bool,
}

impl RlvrParams {
    fn keeps(&self, attempts: u32, rate: f64) -> bool {
        let low = self.min_pass_rate.map_or(rate > 0.0, |low| rate >= low);
        let high = self.max_pass_rate.map_or(rate < 1.0, |high| rate <= high);
        attempts >= self.min_attempts && low && high
    }
}

/// The `dataset.rlvr` operation.
pub struct Rlvr;

fn port(name: &str, kind: ArtifactKind, many: bool) -> Port {
    Port {
        name: name.into(),
        kind,
        many,
        optional: false,
    }
}

impl OperationSpec for Rlvr {
    fn descriptor(&self) -> OperationDescriptor {
        OperationDescriptor {
            schema: OPERATION_SCHEMA.into(),
            name: RLVR.into(),
            revision: 1,
            summary: "the tasks attempts solve only sometimes, as an executable collection".into(),
            inputs: vec![
                port("tasks", ArtifactKind::Tasks, false),
                port("attempts", ArtifactKind::Attempts, true),
            ],
            outputs: vec![port("tasks", ArtifactKind::Files, false)],
            parameters: serde_json::to_value(schema_for!(RlvrParams)).unwrap_or(Value::Null),
            reproducibility: Reproducibility::Deterministic,
            isolation: Isolation::Any,
        }
    }

    fn check(&self, value: &Value) -> Result<(), String> {
        let p = params(value)?;
        let rate = |r: Option<f64>| r.is_none_or(|r| (0.0..=1.0).contains(&r));
        if p.min_attempts < 2 {
            return Err("a pass rate a learner can use rests on at least 2 attempts".into());
        }
        if !rate(p.min_pass_rate) || !rate(p.max_pass_rate) {
            return Err("pass rates lie between 0 and 1".into());
        }
        if let (Some(low), Some(high)) = (p.min_pass_rate, p.max_pass_rate) {
            if low > high {
                return Err("min_pass_rate is above max_pass_rate".into());
            }
        }
        Ok(())
    }

    fn executor(&self) -> Option<&dyn Execute> {
        Some(self)
    }
}

impl Execute for Rlvr {
    fn implementation(&self) -> String {
        format!("{RLVR} 1")
    }

    fn execute(&self, call: &Call<'_>) -> Result<Outcome, String> {
        let p = params(call.params)?;
        let tasks = call.inputs["tasks"].first().ok_or("no tasks are bound")?;
        let collection = tasks_of(tasks).map_err(|e| e.to_string())?;
        let mut counts: BTreeMap<String, (u32, u32)> = BTreeMap::new();
        for view in &call.inputs["attempts"] {
            for trial in trials_of(view).map_err(|e| e.to_string())? {
                // An attempt the infrastructure decided says nothing of the
                // policy.
                if !trial.grade.counted {
                    continue;
                }
                let entry = counts.entry(trial.task.clone()).or_default();
                entry.0 += 1;
                entry.1 += u32::from(trial.verified_success());
            }
        }
        let mut measured = BTreeMap::new();
        let mut kept = 0;
        for (task, (attempts, passes)) in counts {
            if collection.task(&task).is_none() {
                return Err(format!(
                    "task {task} was attempted but is not among the tasks"
                ));
            }
            let pass_rate = f64::from(passes) / f64::from(attempts);
            let keep = p.keeps(attempts, pass_rate);
            if keep {
                link_tree(&tasks.join(&task), &call.out.join(&task)).map_err(|e| e.to_string())?;
                kept += 1;
            }
            measured.insert(
                task,
                Measured {
                    attempts,
                    passes,
                    pass_rate,
                    kept: keep,
                },
            );
        }
        let text = serde_json::to_string_pretty(&measured).map_err(|e| e.to_string())?;
        std::fs::write(call.out.join("rlvr.json"), text + "\n").map_err(|e| e.to_string())?;
        if kept == 0 {
            return Ok(Outcome::Rejected(
                "no task's attempts show a spread a learner can use (rlvr.json says each one's pass rate)"
                    .into(),
            ));
        }
        Ok(Outcome::Produced)
    }
}
