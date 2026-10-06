// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems whose held-out exam cannot
// leak, for its clients. If your team needs expertise in reserving an exam from
// a run's sources before anything is trained, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The stages of `learn` that reserve, write and run the exam: the exam's
//! families are taken out of the sources before anything is generated, the
//! exam is written from them and frozen, and after training the candidate is
//! put to it.

use splinter_orchestrator::pipeline::StageEnd;
use splinter_orchestrator::runs::Recorder;
use splinter_orchestrator::{Context, OrchestratorError};

use super::stages::{to_value, Done, LearnState};
use crate::exam::{examine, Exam, ExamineRequest};
use crate::exam_set::{create as create_exam_set, ExamSetRequest};
use crate::powered::{run as run_powered, PoweredExam, PoweredRequest};
use crate::reserve::{reserve, ReserveRequest};

/// Reserves the exam's families before anything is generated or built, and
/// swaps the run's sources for the sources without them.
pub(super) fn reserve_stage(ctx: &Context, _: &mut Recorder<'_>, st: &mut LearnState<'_>) -> Done {
    // Text the policy's own lineage was trained on is not examinable: the
    // base the candidate is compared with has seen it.
    let mut touched_by = Vec::new();
    if let Some(release) = &st.report.policy.release {
        for earlier in ctx.releases().lineage(release)? {
            touched_by.extend(earlier.manifest.datasets);
        }
    }
    let reserved = reserve(
        ctx,
        &ReserveRequest {
            sources: &st.source_ids,
            families: st.exam_families(),
            seed: 0,
            touched_by: &touched_by,
        },
    )?;
    st.source_ids = reserved.training.clone();
    let summary = to_value(&reserved)?;
    st.report.reserve = Some(reserved);
    Ok(StageEnd::done(summary))
}

/// Writes the exam from the reserved text alone, within its budget, and
/// freezes it, before any training.
pub(super) fn exam_set_stage(
    ctx: &Context,
    run: &mut Recorder<'_>,
    st: &mut LearnState<'_>,
) -> Done {
    let Some(reserved) = st.report.reserve.as_ref() else {
        unreachable!("the exam-set stage follows the reserve stage")
    };
    let kinds = st.kinds.clone();
    let exam = create_exam_set(
        ctx,
        &ExamSetRequest {
            reservation: reserved,
            kinds: &kinds,
            generator: st.learn.generator,
            goal: st.learn.goal,
            author: st.persona(),
            max_tasks: st
                .learn
                .exam
                .tasks
                .unwrap_or(crate::exam_set::DEFAULT_EXAM_TASKS),
            cancel: run.cancel_token(),
        },
    )?;
    let summary = to_value(&exam)?;
    st.exam_set = Some(exam.clone());
    st.report.exam_set = Some(exam);
    Ok(StageEnd::done(summary))
}

/// The candidate is trained and stored whatever the exam finds; an exam that
/// cannot run says why in the report and the release gate still decides.
pub(super) fn exam_stage(ctx: &Context, run: &mut Recorder<'_>, st: &mut LearnState<'_>) -> Done {
    let Some(id) = st.candidate.as_ref() else {
        unreachable!("the exam stage follows the train stage")
    };
    if let Some(exam) = st.exam_set.as_ref() {
        let outcome = match run_powered(
            ctx,
            &PoweredRequest {
                exam,
                candidate: id,
                base: Some(&st.learn.policy),
                judge: Some(st.learn.judge),
                goal: st.learn.goal,
                resamples: st
                    .learn
                    .exam
                    .resamples
                    .unwrap_or(crate::powered::DEFAULT_RESAMPLES),
                voice: true,
                cancel: &run.cancel_token(),
            },
        ) {
            Ok(powered) => PoweredExam::Ran(Box::new(powered)),
            Err(OrchestratorError::Cancelled) => return Err(OrchestratorError::Cancelled),
            Err(e) => PoweredExam::NotRun(format!("the exam failed: {e}")),
        };
        let summary = to_value(&outcome)?;
        st.report.powered = Some(outcome);
        return Ok(StageEnd::done(summary));
    }
    let examined = match examine(
        ctx,
        &ExamineRequest {
            candidate: id,
            base: Some(&st.learn.policy),
            judge: Some(st.learn.judge),
            prompted: st.learn.goal,
            retrieval: None,
        },
        &run.cancel_token(),
    ) {
        Ok(examined) => examined,
        Err(OrchestratorError::Cancelled) => return Err(OrchestratorError::Cancelled),
        Err(e) => Exam::NotRun(format!("the exam failed: {e}")),
    };
    let summary = to_value(&examined)?;
    st.report.exam = Some(examined);
    Ok(StageEnd::done(summary))
}
