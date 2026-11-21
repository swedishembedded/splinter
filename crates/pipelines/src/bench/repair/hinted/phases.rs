// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from a failed agent attempt to a
// verified corrected one, for its clients. If your team needs expertise in
// judging agent attempts before they become training data you can procure
// our services by sending an email to info@swedishembedded.com.

//! The phases of a rationalisation that judge attempts: the hinted trials
//! run and judged, the judgement of one attempt and what it settles to, and
//! the sufficiency trials.

use std::collections::BTreeMap;
use std::path::Path;

use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

use super::filters::{self, LeakCheck};
use super::keys::digest_of;
use super::material::TrainTask;
use super::record::{keep, Kept};
use super::trials::{read_attempt, run_job, stage_copy, staged_name, JobEnv};
use super::{
    refused, trial_result_sha, Case, Checks, Origin, Outcome, RationaliseRecord, RationaliseRequest,
};
use crate::bench::artifact::AgentContext;

/// Judges the graded attempt `trajectory` of `task`, given `instruction`
/// (the task's with its hint section) and graded `reward`, by the
/// screening filters the run applies: verified, the real-observations rule
/// when the run opts in, and no answer present.
pub(super) fn judge(
    record: &mut RationaliseRecord,
    reward: f64,
    trajectory: &atif::Trajectory,
    context: &AgentContext,
    instruction: &str,
    task: &TrainTask,
    request: &RationaliseRequest<'_>,
) {
    let section = instruction
        .find(filters::HINT_BEGIN)
        .map_or("", |at| &instruction[at..]);
    let shown = format!(
        "{}\n{}",
        context.system,
        serde_json::to_string(&context.tools).unwrap_or_default()
    );
    record.filters.verified = Some(filters::verified(Some(reward)));
    if request.checks.real_observations {
        record.filters.real_observations = Some(filters::read_before_write(
            trajectory,
            &task.files,
            &task.instruction,
        ));
    }
    record.filters.no_leak = Some(filters::no_leak(
        trajectory,
        &LeakCheck {
            terms: request.leak_terms,
            hint: section,
            instruction: &task.instruction,
            shown: &shown,
            files: &task.files,
        },
    ));
}

/// Stores the judged `record` of `case`: rejected when a screening filter
/// failed, a candidate when the sufficiency run is owed, else kept.
pub(super) fn settle(
    ctx: &Context,
    mut record: RationaliseRecord,
    case: &Case<'_>,
    attempt: (atif::Trajectory, AgentContext),
    checks: Checks,
) -> Result<(String, RationaliseRecord), OrchestratorError> {
    if !record.filters.screened(checks) {
        record.outcome = Outcome::Rejected;
        return stored(ctx, record, case);
    }
    if checks.sufficiency {
        record.outcome = Outcome::Candidate;
        return stored(ctx, record, case);
    }
    Ok((case.key.clone(), keep_attempt(ctx, record, case, attempt)?))
}

/// Keeps the attempt `record` judged: its trajectory with the task's own
/// instruction as the user turn; rejected when that cannot be made.
fn keep_attempt(
    ctx: &Context,
    mut record: RationaliseRecord,
    case: &Case<'_>,
    (trajectory, context): (atif::Trajectory, AgentContext),
) -> Result<RationaliseRecord, OrchestratorError> {
    let hinted_dir = record.hinted_trial.clone().unwrap_or_default();
    let kept =
        filters::kept_trajectory(&trajectory, &case.task.instruction).map(|trajectory| Kept {
            trajectory,
            context,
            trial: hinted_dir
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            result_sha256: trial_result_sha(&hinted_dir),
            solver: record.rationaliser.clone().unwrap_or_default(),
            helped: true,
        });
    match kept {
        Ok(kept) => {
            record.outcome = Outcome::Kept;
            keep(ctx, record, case.task, Some(&kept))
        }
        Err(why) => {
            record.outcome = Outcome::Rejected;
            record.reason = Some(format!("the kept trajectory cannot be made: {why}"));
            keep(ctx, record, case.task, None)
        }
    }
}

/// Runs the hinted attempts of `cases` as one Harbor job and judges each;
/// returns the stored records by key.
pub(super) fn hinted_phase(
    ctx: &Context,
    env: &JobEnv<'_>,
    dir: &Path,
    cases: &[&Case<'_>],
    request: &RationaliseRequest<'_>,
    identity: &str,
) -> Result<Vec<(String, RationaliseRecord)>, OrchestratorError> {
    let mut keys: Vec<&str> = cases.iter().map(|c| c.key.as_str()).collect();
    keys.sort_unstable();
    let name = format!("hinted-{}", &digest_of(&keys)?[..16]);
    let tasks_dir = dir.join(&name).join("tasks");
    let mut staged = BTreeMap::new();
    for case in cases {
        let folder = staged_name(&case.key, &folder_name(case.task));
        let instruction =
            filters::hinted_instruction(&case.task.instruction, &case.task.hint, request.framing);
        stage_copy(case.task, &instruction, &tasks_dir, &folder).map_err(refused)?;
        staged.insert(case.key.as_str(), (folder, instruction));
    }
    let attempts =
        run_job(env, &tasks_dir, &dir.join(&name).join("jobs"), &name).map_err(refused)?;
    let mut out = Vec::new();
    for case in cases {
        let (folder, instruction) = &staged[case.key.as_str()];
        let mut record = RationaliseRecord::new(
            &case.key,
            Origin::Rationalised,
            case.task,
            &case.student,
            Some(identity),
            case.sample,
        );
        let Some(attempt) = attempts.get(folder) else {
            out.push(stored(
                ctx,
                record.refuse("Harbor left no trial of the staged copy"),
                case,
            )?);
            continue;
        };
        record.hinted_trial = Some(attempt.dir.clone());
        let Some(reward) = attempt.reward else {
            out.push(stored(
                ctx,
                record.refuse("the hinted trial ended without a grade"),
                case,
            )?);
            continue;
        };
        let (trajectory, context) = match read_attempt(&attempt.dir) {
            Ok(read) => read,
            Err(why) => {
                out.push(stored(ctx, record.refuse(why), case)?);
                continue;
            }
        };
        judge(
            &mut record,
            reward,
            &trajectory,
            &context,
            instruction,
            case.task,
            request,
        );
        out.push(settle(
            ctx,
            record,
            case,
            (trajectory, context),
            request.checks,
        )?);
    }
    Ok(out)
}

/// Stores `record` of `case` with no experience.
pub(super) fn stored(
    ctx: &Context,
    record: RationaliseRecord,
    case: &Case<'_>,
) -> Result<(String, RationaliseRecord), OrchestratorError> {
    Ok((case.key.clone(), keep(ctx, record, case.task, None)?))
}

/// The task folder's own name.
fn folder_name(task: &TrainTask) -> String {
    task.folder
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| task.rel.replace('/', "-"))
}

/// Runs the sufficiency trial of every candidate as one Harbor job: the
/// task's instruction and the candidate's reasoning, no hint. A pass keeps
/// the candidate; returns the stored records by key.
pub(super) fn sufficiency_phase(
    ctx: &Context,
    env: &JobEnv<'_>,
    dir: &Path,
    candidates: &[(&Case<'_>, RationaliseRecord)],
) -> Result<Vec<(String, RationaliseRecord)>, OrchestratorError> {
    let mut keys: Vec<&str> = candidates.iter().map(|(c, _)| c.key.as_str()).collect();
    keys.sort_unstable();
    let name = format!("sufficiency-{}", &digest_of(&keys)?[..16]);
    let tasks_dir = dir.join(&name).join("tasks");
    let mut hinted = BTreeMap::new();
    let mut out = Vec::new();
    for (case, record) in candidates {
        let read = record
            .hinted_trial
            .as_deref()
            .ok_or_else(|| "the candidate names no hinted trial".to_string())
            .and_then(read_attempt);
        match read {
            Ok((trajectory, context)) => {
                let notes = filters::reasoning_of(&trajectory);
                let instruction = filters::notes_instruction(&case.task.instruction, &notes);
                let folder = staged_name(&case.key, &folder_name(case.task));
                stage_copy(case.task, &instruction, &tasks_dir, &folder).map_err(refused)?;
                hinted.insert(case.key.as_str(), (folder, trajectory, context));
            }
            Err(why) => out.push(stored(ctx, record.clone().refuse(why), case)?),
        }
    }
    if hinted.is_empty() {
        return Ok(out);
    }
    let attempts =
        run_job(env, &tasks_dir, &dir.join(&name).join("jobs"), &name).map_err(refused)?;
    for (case, record) in candidates {
        let Some((folder, trajectory, context)) = hinted.remove(case.key.as_str()) else {
            continue;
        };
        let mut record = record.clone();
        let Some(attempt) = attempts.get(&folder) else {
            record.reason = Some("Harbor left no sufficiency trial; it is run again".into());
            out.push(stored(ctx, record, case)?);
            continue;
        };
        record.sufficiency_trial = Some(attempt.dir.clone());
        let Some(reward) = attempt.reward else {
            record.reason =
                Some("the sufficiency trial ended without a grade; it is run again".into());
            out.push(stored(ctx, record, case)?);
            continue;
        };
        let sufficient = filters::verified(Some(reward));
        record.reason = None;
        record.filters.sufficient = Some(sufficient.clone());
        if !sufficient.passed {
            record.outcome = Outcome::Rejected;
            out.push(stored(ctx, record, case)?);
            continue;
        }
        let record = keep_attempt(ctx, record, case, (trajectory, context))?;
        out.push((case.key.clone(), record));
    }
    Ok(out)
}
