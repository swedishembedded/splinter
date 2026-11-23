// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from a failed agent attempt to a
// verified corrected one, for its clients. If your team needs expertise in
// turning agent failures into training data you can procure our services by
// sending an email to info@swedishembedded.com.

//! `attempts.failures`: the tasks attempt stages left unsolved, as a
//! collection a teacher can work over.
//!
//! A task is a failure when some given stage attempted it and none solved
//! it. Each failure is an item named by its task ([`brief_item`]) holding
//! the task's `instruction.md` and `attempt.md`, the last failed attempt's
//! conversation condensed to what a reader needs: every turn's words under
//! its step id (what a fork point names), each tool call and what it
//! returned, each cut to a bound. An `agent` stage
//! over the collection writes a brief per task (knowledge passages, a plan
//! of subgoals, notes on the mistake) that an attempt stage with
//! `hint: brief` shows the student ([`crate::hint`]).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use splinter_core::factory::execution::Isolation;
use splinter_core::factory::items::{Item, ItemIndex, ITEMS_FILE};
use splinter_core::factory::operation::{
    ArtifactKind, OperationDescriptor, Port, Reproducibility, OPERATION_SCHEMA,
};
use splinter_core::factory::product::TrialResult;

use crate::execute::{Call, Execute, Outcome};
use crate::hint::{brief_item, INSTRUCTION};
use crate::registry::OperationSpec;
use crate::run::rollout::trials_of;
use crate::run::tasks::tasks_of;

/// The operation's name.
pub const FAILURES: &str = "attempts.failures";

/// The condensed attempt in a failure's item.
const ATTEMPT_FILE: &str = "attempt.md";

fn transcript_bytes() -> usize {
    24 * 1024
}

/// `attempts.failures`' parameters.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FailuresParams {
    /// The longest condensed attempt, in bytes.
    #[serde(default = "transcript_bytes")]
    pub transcript_bytes: usize,
}

/// The `attempts.failures` operation.
pub struct Failures;

fn params(value: &Value) -> Result<FailuresParams, String> {
    serde_json::from_value(value.clone()).map_err(|e| e.to_string())
}

fn port(name: &str, kind: ArtifactKind, many: bool) -> Port {
    Port {
        name: name.into(),
        kind,
        many,
        optional: false,
    }
}

impl OperationSpec for Failures {
    fn descriptor(&self) -> OperationDescriptor {
        OperationDescriptor {
            schema: OPERATION_SCHEMA.into(),
            name: FAILURES.into(),
            revision: 1,
            summary: "the tasks attempt stages left unsolved, with the last failed attempt".into(),
            inputs: vec![
                port("tasks", ArtifactKind::Tasks, false),
                port("attempts", ArtifactKind::Attempts, true),
            ],
            outputs: vec![port("failures", ArtifactKind::Files, false)],
            parameters: serde_json::to_value(schema_for!(FailuresParams)).unwrap_or(Value::Null),
            reproducibility: Reproducibility::Deterministic,
            isolation: Isolation::Any,
        }
    }

    fn check(&self, value: &Value) -> Result<(), String> {
        let p = params(value)?;
        if p.transcript_bytes == 0 {
            return Err("transcript_bytes is at least 1".into());
        }
        Ok(())
    }

    fn executor(&self) -> Option<&dyn Execute> {
        Some(self)
    }
}

impl Execute for Failures {
    fn implementation(&self) -> String {
        format!("{FAILURES} 1")
    }

    fn execute(&self, call: &Call<'_>) -> Result<Outcome, String> {
        let p = params(call.params)?;
        let tasks = call.inputs["tasks"].first().ok_or("no tasks are bound")?;
        let collection = tasks_of(tasks).map_err(|e| e.to_string())?;
        let mut solved = BTreeSet::new();
        // The last failed attempt of every task attempted, with the view it is in.
        let mut failed: BTreeMap<String, (TrialResult, &Path)> = BTreeMap::new();
        for view in &call.inputs["attempts"] {
            for trial in trials_of(view).map_err(|e| e.to_string())? {
                if trial.verified_success() {
                    solved.insert(trial.task.clone());
                } else {
                    failed.insert(trial.task.clone(), (trial, view.as_path()));
                }
            }
        }
        let mut items = Vec::new();
        for (task, (trial, view)) in failed.iter().filter(|(t, _)| !solved.contains(*t)) {
            if collection.task(task).is_none() {
                return Err(format!(
                    "task {task} was attempted but is not among the tasks"
                ));
            }
            let id = brief_item(task);
            let dir = call.out.join(&id);
            std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            let instruction = tasks.join(task).join(INSTRUCTION);
            std::fs::copy(&instruction, dir.join(INSTRUCTION))
                .map_err(|e| format!("{}: {e}", instruction.display()))?;
            let attempt = condensed(view, trial, p.transcript_bytes);
            std::fs::write(dir.join(ATTEMPT_FILE), attempt)
                .map_err(|e| format!("{}: {e}", dir.display()))?;
            items.push(Item {
                path: id.clone(),
                id,
            });
        }
        let index = ItemIndex::new(items, Vec::new()).map_err(|e| e.to_string())?;
        let text = serde_json::to_string_pretty(&index).map_err(|e| e.to_string())?;
        std::fs::write(call.out.join(ITEMS_FILE), text + "\n").map_err(|e| e.to_string())?;
        Ok(Outcome::Produced)
    }
}

/// The trial's conversation as a reader needs it, at most `bound` bytes; a
/// trial with no readable trajectory says so instead.
fn condensed(view: &Path, trial: &TrialResult, bound: usize) -> String {
    let mut out = format!(
        "# The failed attempt\n\nGraded {:?}, score {:?}.\n\n",
        trial.grade.status, trial.grade.score
    );
    let trajectory = trial
        .trace
        .as_ref()
        .and_then(|trace| std::fs::read(view.join(&trace.path)).ok())
        .and_then(|bytes| serde_json::from_slice::<atif::Trajectory>(&bytes).ok());
    let Some(trajectory) = trajectory else {
        out.push_str("(the attempt left no readable trajectory)\n");
        return out;
    };
    for step in &trajectory.steps {
        let source = serde_json::to_value(step.source)
            .ok()
            .and_then(|v| v.as_str().map(str::to_owned))
            .unwrap_or_else(|| "step".into());
        if let Some(text) = step.message.as_text().filter(|t| !t.trim().is_empty()) {
            out.push_str(&format!(
                "## step {} ({source})\n\n{}\n\n",
                step.step_id,
                cut(text, 2000)
            ));
        }
        for call in step.tool_calls.iter().flatten() {
            out.push_str(&format!(
                "-> {}({})\n",
                call.function_name,
                cut(&call.arguments.to_string(), 300)
            ));
        }
        for result in step.observation.iter().flat_map(|o| &o.results) {
            if let Some(text) = result.content.as_ref().and_then(|c| c.as_text()) {
                out.push_str(&format!("<- {}\n", cut(text, 600)));
            }
        }
        if out.len() > bound {
            break;
        }
    }
    if out.len() > bound {
        let mut end = bound;
        while !out.is_char_boundary(end) {
            end -= 1;
        }
        out.truncate(end);
        out.push_str("\n\n(cut at the bound)\n");
    }
    out
}

/// `text` cut to at most `chars` characters, marked when cut.
fn cut(text: &str, chars: usize) -> String {
    let mut cut: String = text.chars().take(chars).collect();
    if cut.len() < text.len() {
        cut.push_str(" [...]");
    }
    cut
}
