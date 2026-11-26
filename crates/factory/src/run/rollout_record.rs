// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running agent experiments that survive crashes and restarts for its
// clients. If your team needs expertise in resumable evaluation and training pipelines then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Recording what a batch of attempts returned: each claimed attempt's report becomes a trial and finishes or
//! returns its slot; an attempt without a report is lost, and whether that spends a retry depends on why
//! ([`Lost`]).

use splinter_core::clock::unix_now;
use splinter_core::factory::attempt::AttemptReport;
use splinter_store::factory::runstate::{AttemptRecord, AttemptState};

use super::rollout::{attempt_number, AttemptEntry, Running};
use super::{stage_error, Factory, RunError};

/// Why an attempt of a batch has no report.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Lost {
    /// The worker ran the batch to its end and reported nothing for it: the
    /// attempt counts against its slot's retries.
    Unreported,
    /// The run was stopped before the batch reported (found by a restarted
    /// controller): the slot is run again and no retry is spent.
    Interrupted,
}

pub(super) fn record(
    factory: &Factory<'_>,
    run: &Running<'_, '_>,
    batch: u64,
    attempts: &[AttemptRecord],
    reports: &[AttemptReport],
    lost: Lost,
) -> Result<Vec<AttemptEntry>, RunError> {
    let mut entries = Vec::new();
    let allowed = run.arm.pass.retries as usize;
    for attempt in attempts {
        let number = attempt_number(factory, attempt)?;
        let used = factory.state.attempts(&attempt.slot)?.len();
        let Some(report) = reports
            .iter()
            .find(|r| r.slot == attempt.slot && r.attempt == number)
        else {
            factory.state.record_attempt(
                &factory.fence,
                attempt.id,
                AttemptState::Lost,
                None,
                unix_now(),
            )?;
            if lost == Lost::Unreported && used > allowed {
                return Err(stage_error(
                    &run.arm.key,
                    format!(
                        "slot {} lost {used} attempts without a report",
                        attempt.slot
                    ),
                ));
            }
            continue;
        };
        let trial = run.trial(report, batch)?;
        let retry = trial.grade.status.is_retryable() && used <= allowed;
        let digest = trial.digest();
        if retry {
            factory.state.record_attempt(
                &factory.fence,
                attempt.id,
                AttemptState::Failed,
                None,
                unix_now(),
            )?;
        } else {
            factory.state.record_attempt(
                &factory.fence,
                attempt.id,
                AttemptState::Succeeded,
                Some(digest.clone()),
                unix_now(),
            )?;
            factory
                .state
                .finish_slot(&factory.fence, &attempt.slot, &digest, unix_now())?;
        }
        entries.push(AttemptEntry {
            batch,
            finished_slot: !retry,
            trial,
        });
    }
    Ok(entries)
}
