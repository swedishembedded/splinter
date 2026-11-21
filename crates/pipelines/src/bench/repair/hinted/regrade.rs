// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from a failed agent attempt to a
// verified corrected one, for its clients. If your team needs expertise in
// judging recorded agent attempts again under revised acceptance criteria,
// without paying for new model runs, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Re-grading: the graded attempts of earlier runs judged again with this
//! run's checks, without running a hinted trial.
//!
//! An earlier run is read from its `report.json`: its dataset's compile
//! record (it must be this run's), its hint framing (absent meaning the
//! worked answer, the only one before the option) and every rationalised
//! record with a graded hinted trial. A run that was itself a re-grade is
//! refused: its sources are re-graded instead. The hinted trial's job,
//! packed by `bench pack` or not, is unpacked in a copy in this run's
//! directory, never in place, so the earlier run's evidence is only read.
//! The attempt's hint section is rebuilt from the task and the earlier
//! run's framing, and must be the one its user turn holds.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde_json::Value;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

use super::filters::{self, HintFraming};
use super::material::{CompiledDataset, StudentTrial};
use super::phases::{judge, settle, stored};
use super::trials::{attempt_at, job_trial_dir, read_attempt};
use super::{
    refused, Case, Origin, RationaliseRecord, RationaliseRequest, RegradeOf, RATIONALISE_FORMAT,
};

/// One attempt of an earlier run to judge again.
#[derive(Clone, Debug)]
pub(super) struct Earlier {
    /// The earlier run's directory, resolved.
    pub(super) run: PathBuf,
    /// Its record of the attempt.
    pub(super) record: RationaliseRecord,
    /// How that run framed the hint.
    pub(super) framing: HintFraming,
}

impl Earlier {
    /// The student's trial the attempt answers, as its record names it.
    pub(super) fn student(&self) -> StudentTrial {
        StudentTrial {
            job: self.record.student_job.clone(),
            trial: self.record.student_trial.clone(),
            task: PathBuf::from(&self.record.folder),
            reward: None,
        }
    }

    /// The hinted trial the earlier run recorded.
    fn hinted_trial(&self) -> &Path {
        self.record.hinted_trial.as_deref().unwrap_or(Path::new(""))
    }
}

/// Every graded attempt of the earlier runs `runs`, each run once, in run
/// and record order.
///
/// # Errors
/// A run cannot be read, is not a rationalisation, is of another dataset,
/// or was itself a re-grade.
pub(super) fn earlier(
    runs: &[PathBuf],
    dataset: &CompiledDataset,
) -> Result<Vec<Earlier>, OrchestratorError> {
    let mut seen = BTreeSet::new();
    let mut attempts = Vec::new();
    for run in runs {
        let run = run
            .canonicalize()
            .map_err(|e| refused(format!("earlier run {}: {e}", run.display())))?;
        if !seen.insert(run.clone()) {
            continue;
        }
        let path = run.join("report.json");
        let bytes =
            std::fs::read(&path).map_err(|e| refused(format!("{}: {e}", path.display())))?;
        let report: Value = serde_json::from_slice(&bytes)
            .map_err(|e| refused(format!("{}: {e}", path.display())))?;
        let named = |why: &str| refused(format!("earlier run {}: {why}", run.display()));
        if report.get("format").and_then(Value::as_str) != Some(RATIONALISE_FORMAT) {
            return Err(named(&format!("its report is not {RATIONALISE_FORMAT}")));
        }
        if report
            .get("regraded")
            .and_then(Value::as_array)
            .is_some_and(|r| !r.is_empty())
        {
            return Err(named(
                "it is a re-grade itself; re-grade the runs it re-graded",
            ));
        }
        let sha = report
            .pointer("/dataset/compile_sha256")
            .and_then(Value::as_str);
        if sha != Some(dataset.sha256.as_str()) {
            return Err(named(&format!(
                "its dataset's compile record is {}, not this dataset's {}: its hints are not \
                 this dataset's",
                sha.unwrap_or("unrecorded"),
                dataset.sha256
            )));
        }
        let framing: HintFraming = match report.get("hint_framing") {
            None | Some(Value::Null) => HintFraming::default(),
            Some(value) => serde_json::from_value(value.clone())
                .map_err(|e| named(&format!("its hint framing: {e}")))?,
        };
        let records: Vec<RationaliseRecord> =
            serde_json::from_value(report.get("records").cloned().unwrap_or(Value::Null))
                .map_err(|e| named(&format!("its records: {e}")))?;
        attempts.extend(
            records
                .into_iter()
                .filter(|r| {
                    r.origin == Origin::Rationalised
                        && r.filters.verified.is_some()
                        && r.hinted_trial.is_some()
                })
                .map(|record| Earlier {
                    run: run.clone(),
                    record,
                    framing,
                }),
        );
    }
    Ok(attempts)
}

/// The earlier runs `cases` re-grade, in order.
pub(super) fn runs_of(cases: &[Case<'_>]) -> Vec<PathBuf> {
    cases
        .iter()
        .filter_map(|c| c.earlier.as_ref().map(|e| e.run.clone()))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// Judges every attempt of `cases` again, offline; returns the stored
/// records by key.
pub(super) fn phase(
    ctx: &Context,
    request: &RationaliseRequest<'_>,
    dir: &Path,
    cases: &[&Case<'_>],
) -> Result<Vec<(String, RationaliseRecord)>, OrchestratorError> {
    let copies = dir.join("regraded");
    let mut out = Vec::new();
    for case in cases {
        let Some(earlier) = &case.earlier else {
            continue;
        };
        let mut record = RationaliseRecord::new(
            &case.key,
            Origin::Rationalised,
            case.task,
            &case.student,
            earlier.record.rationaliser.as_deref(),
            case.sample,
        );
        record.regrade_of = Some(RegradeOf {
            run: earlier.run.clone(),
            key: earlier.record.key.clone(),
            hinted_trial: earlier.hinted_trial().to_path_buf(),
        });
        let read = trial_of(request, earlier, &copies).and_then(|trial| {
            record.hinted_trial = Some(trial.clone());
            let attempt = attempt_at(&trial)?;
            let reward = attempt
                .reward
                .ok_or("the earlier hinted trial ended without a grade")?;
            let (trajectory, context) = read_attempt(&trial)?;
            let instruction = filters::hinted_instruction(
                &case.task.instruction,
                &case.task.hint,
                earlier.framing,
            );
            let section = instruction
                .find(filters::HINT_BEGIN)
                .map_or("", |at| &instruction[at..])
                .trim_end();
            let user = filters::user_turn(&trajectory).unwrap_or_default();
            if !user.contains(section) {
                return Err(format!(
                    "the hint section rebuilt with the {:?} framing is not the one the earlier \
                     attempt's user turn holds: the task changed since, or the framing is not \
                     the run's",
                    earlier.framing
                ));
            }
            Ok((reward, trajectory, context, instruction))
        });
        match read {
            Ok((reward, trajectory, context, instruction)) => {
                judge(
                    &mut record,
                    reward,
                    &trajectory,
                    &context,
                    &instruction,
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
            Err(why) => out.push(stored(ctx, record.refuse(why), case)?),
        }
    }
    Ok(out)
}

/// The earlier attempt's trial directory, unpacked in a copy under `copies`
/// when its job is packed.
fn trial_of(
    request: &RationaliseRequest<'_>,
    earlier: &Earlier,
    copies: &Path,
) -> Result<PathBuf, String> {
    let recorded = earlier.hinted_trial();
    let (Some(job), Some(trial)) = (recorded.parent(), recorded.file_name()) else {
        return Err(format!("{} names no trial of a job", recorded.display()));
    };
    job_trial_dir(request.checkout, job, &trial.to_string_lossy(), copies)
}
