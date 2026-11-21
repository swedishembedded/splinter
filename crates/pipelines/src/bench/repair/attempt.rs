// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from a failed agent attempt to a
// verified corrected one, for its clients. If your team needs expertise in
// turning agent failures into training data you can procure our services by
// sending an email to info@swedishembedded.com.

//! One repair attempt: an intervention run on one case, and its record.

use std::path::Path;

use splinter_core::diagnosis::DiagnosticResult;
use splinter_core::experience::{Experience, ExperienceId};
use splinter_orchestrator::context::Context;

use super::super::diagnose::DiagnosedTrial;
use super::{Case, Intervention, Notes, RepairOutcome, RepairRecord, REPAIR_FORMAT};

/// Runs the intervention on one case and makes its record.
#[allow(clippy::too_many_arguments)]
pub(super) fn attempt_case(
    ctx: &Context,
    intervention: &dyn Intervention,
    trial: &DiagnosedTrial,
    result: &DiagnosticResult,
    run: &Path,
    student_id: &ExperienceId,
    key: &str,
    sample: usize,
) -> (RepairRecord, Option<Experience>) {
    let mut record = RepairRecord {
        format: REPAIR_FORMAT.into(),
        key: key.into(),
        student: student_id.clone(),
        trial: trial.trial.clone(),
        task: trial.task.clone(),
        domain: trial.domain.clone(),
        diagnosis: result.clone(),
        intervention: intervention.name().into(),
        sample,
        keep_through: result.last_good_step().unwrap_or_default(),
        teacher: intervention.actor().into(),
        outcome: RepairOutcome::Refused,
        reason: None,
        grade: None,
        experience: None,
        prefix_steps: None,
        teacher_actions: None,
        notes: Notes::default(),
    };
    let student = match ctx.experiences().get(student_id) {
        Ok(student) => student,
        Err(e) => {
            record.reason = Some(format!("the student's experience: {e}"));
            return (record, None);
        }
    };
    let case = Case {
        trial,
        diagnosis: result,
        run,
        student: &student,
        student_id,
        key,
        sample,
    };
    match intervention.attempt(ctx, &case) {
        Err(reason) => {
            record.reason = Some(reason);
            (record, None)
        }
        Ok(attempted) => {
            record.keep_through = attempted.keep_through;
            record.prefix_steps = Some(attempted.prefix_steps);
            record.teacher_actions = Some(attempted.teacher_actions);
            record.notes = attempted.notes;
            let passed = attempted.grade.passed();
            record.outcome = if passed {
                RepairOutcome::Accepted
            } else {
                RepairOutcome::Rejected
            };
            record.grade = Some(attempted.grade);
            (record, passed.then_some(attempted.experience))
        }
    }
}
