// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from a failed agent attempt to a
// verified corrected one, for its clients. If your team needs expertise in
// turning agent failures into training data you can procure our services by
// sending an email to info@swedishembedded.com.

//! The trajectory a repair is kept as, and the lineage it carries.
//!
//! The fork composes the student's exported steps with the teacher's. The
//! student's stored experience holds the same steps, led by the system prompt
//! the intake read from the trial's first request (and the tool schemas in
//! `agent.tool_definitions`). The repair is kept in the stored shape, so that
//! its first `prefix_steps` steps are exactly the student's stored ones: the
//! shared prefix a preference pair and a suffix-only fine-tune read.

use atif::{StepOrigin, Trajectory};
use serde::Serialize;
use serde_json::{json, Value};
use splinter_agent::correction::CORRECTION_KEY;
use splinter_core::diagnosis::DiagnosticResult;
use splinter_core::experience::ExperienceId;

use super::commands::Grade;

/// What a repair records about where it came from.
#[derive(Serialize)]
pub(super) struct Lineage<'a> {
    /// The student's stored attempt.
    pub student: &'a ExperienceId,
    /// Its trial.
    pub trial: &'a str,
    /// Why it failed.
    pub diagnosis: &'a DiagnosticResult,
    /// The last exported student step kept.
    pub keep_through: u64,
    /// The leading steps shared with the student's stored trajectory.
    pub prefix_steps: usize,
    /// The teacher.
    pub teacher: &'a str,
    /// The intervention.
    pub intervention: &'a str,
    /// What the fork held the replay to: `observations` (every replayed
    /// observation), or also the tree's digest.
    pub workspace_check: &'a str,
    /// The verifier's grade.
    pub grade: &'a Grade,
}

/// `composed` (the fork's student steps through `keep_through`, then the
/// teacher's) in the shape of the student's `stored` trajectory, and the
/// number of leading steps the two share.
///
/// # Errors
/// When the stored trajectory does not hold the kept steps.
pub(super) fn onto_stored(
    stored: &Trajectory,
    composed: Trajectory,
    keep_through: u64,
) -> Result<(Trajectory, usize), String> {
    let lead = usize::from(stored.steps.first().map(|s| s.source) == Some(StepOrigin::System));
    let kept_raw = composed
        .steps
        .iter()
        .take_while(|s| s.is_copied_context == Some(true))
        .count();
    let prefix_steps = lead + kept_raw;
    let offset = u64::try_from(lead).unwrap_or_default();
    if stored.steps.len() < prefix_steps
        || stored.steps[prefix_steps - 1].step_id != keep_through + offset
    {
        return Err(format!(
            "the student's stored trajectory does not hold its exported steps through step \
             {keep_through}"
        ));
    }
    let teacher_steps = composed.steps.into_iter().skip(kept_raw);
    let mut trajectory = Trajectory::new(composed.schema_version.clone(), composed.agent.clone());
    if trajectory.agent.tool_definitions.is_none() {
        trajectory.agent.tool_definitions = stored.agent.tool_definitions.clone();
    }
    trajectory.session_id = composed.session_id.clone();
    trajectory.final_metrics = composed.final_metrics.clone();
    trajectory.extra = composed.extra.clone();
    let mut id = 1;
    for step in stored.steps.iter().take(prefix_steps) {
        let mut step = step.clone();
        step.step_id = id;
        step.is_copied_context = Some(true);
        id += 1;
        trajectory.steps.push(step);
    }
    for mut step in teacher_steps {
        step.step_id = id;
        id += 1;
        trajectory.steps.push(step);
    }
    Ok((trajectory, prefix_steps))
}

/// `trajectory` with `lineage` beside what the fork recorded under
/// `extra.correction`.
///
/// # Errors
/// When the lineage cannot be written.
pub(super) fn with_lineage(
    mut trajectory: Trajectory,
    lineage: &Lineage<'_>,
) -> Result<Trajectory, String> {
    let lineage = serde_json::to_value(lineage).map_err(|e| e.to_string())?;
    let mut extra = match trajectory.extra.take() {
        Some(Value::Object(map)) => map,
        Some(other) => serde_json::Map::from_iter([("exported".to_string(), other)]),
        None => serde_json::Map::new(),
    };
    let mut correction = match extra.remove(CORRECTION_KEY) {
        Some(Value::Object(map)) => map,
        _ => serde_json::Map::new(),
    };
    if let Value::Object(fields) = lineage {
        correction.extend(fields);
    }
    extra.insert(CORRECTION_KEY.to_string(), json!(correction));
    trajectory.extra = Some(Value::Object(extra));
    Ok(trajectory)
}
