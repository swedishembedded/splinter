// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model releases proven to run on the
// plain serving stack before they ship, for its clients. If your team
// needs expertise in deploying fine-tuned coding agents behind a serving
// endpoint, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The serve check for a candidate that works in a workspace.
//!
//! The candidate is offered to `brain serve` as the release manifest a
//! release step publishes (`--adapter-manifest`), the server is waited for
//! until it reports the candidate's digest and is ready, and the sample of
//! the exam is attempted again through its OpenAI-compatible endpoint by the
//! same worker, under the same bounds and seeds, graded by the same
//! verifiers as the in-process attempts were ([`probe::grade_on`]). The
//! two runs of an agent over many steps share no text worth comparing, so
//! what must agree is the verdict: at least
//! [`gate::SERVE_AGREEMENT_PERCENT`] percent of the sampled tasks must be
//! graded as the in-process attempt at them was ([`Agreement::Verdicts`]).
//!
//! A sample task the in-process attempts did not grade has nothing to agree
//! with, so a sample holds only graded ones (the caller chooses it so). An
//! attempt the harness could not carry out through the server is the check's
//! failure to measure, not the served candidate's disagreement; so is a
//! server that stopped answering, with what it last wrote. An attempt that
//! was set aside as exposed has no verdict and disagrees with a graded one:
//! the served candidate did what the in-process one did not.

use std::path::Path;

use splinter_agent::CancelToken;
use splinter_core::digest::Digest;
use splinter_eval::gate::{self, Agreement, Check, Disagreement, Serve};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

use super::{finished, scratch, served_model, ServeCheck, NO_BINARY};
use crate::brain_server::{self as server, Failure, Launch, Offer};
use crate::release::probe::{self, Arm, Probe};

/// The most characters of a patch a disagreement keeps: the report is stored
/// with the release and a patch can be large.
const KEPT_PATCH_CHARS: usize = 2000;

/// Asks the sample of `request` through a server following a manifest for
/// the candidate; see the module documentation.
pub(super) fn check(
    ctx: &Context,
    request: &ServeCheck<'_>,
    cancel: &CancelToken,
) -> Result<Check<Serve>, OrchestratorError> {
    let Some(binary) = request.binary else {
        return Ok(Check::unmeasured(NO_BINARY));
    };
    // `brain serve` is another process loading its own copy of the base:
    // this one holds none while it runs.
    ctx.release_bases();
    let work = scratch(ctx)?;
    let result = serve_and_work(ctx, request, binary, &work, cancel);
    // The directory only held the keys, the ready marker and the manifest.
    let _ = std::fs::remove_dir_all(&work);
    finished(result)
}

fn serve_and_work(
    ctx: &Context,
    request: &ServeCheck<'_>,
    binary: &Path,
    work: &Path,
    cancel: &CancelToken,
) -> Result<Check<Serve>, Failure> {
    let (sample, in_process) = (request.sample, request.in_process);
    if let Some(task) = sample
        .tasks
        .iter()
        .zip(in_process)
        .find(|(_, probe)| probe.verdict.is_none())
        .map(|(task, _)| &task.task.id)
    {
        return Err(Failure::Unmeasured(format!(
            "the in-process attempt at task {task} was not graded, so the served one has nothing \
             to agree with"
        )));
    }
    let manifest = work.join("adapter-manifest.json");
    let offer = match request.adapter {
        Some(adapter) => {
            publish(&manifest, adapter, request.adapter_digest)?;
            Offer::Manifest(&manifest)
        }
        None => Offer::Nothing,
    };
    let mut started = server::start(
        &Launch {
            binary,
            work,
            models: &ctx.config().model_store,
            base: request.base,
            offer,
            expected_digest: request.adapter_digest,
            startup: request.startup,
            device: None,
            max_batch: None,
            kv_offload_gb: None,
            log: None,
        },
        cancel,
    )?;
    let mut measured = Serve {
        binary: binary.to_path_buf(),
        startup_line: started.startup_line.clone(),
        served_digest: started.served_digest.clone(),
        expected_digest: request.adapter_digest.to_string(),
        sampled: 0,
        agreed: 0,
        disagreed: Vec::new(),
        agreement: Agreement::Verdicts,
    };
    if !started.serves_candidate(request.adapter_digest) {
        return Ok(gate::serve(measured));
    }
    let served = served_model(&started, request.system)?;
    let probes =
        probe::grade_on(ctx, &Arm::handed(&served), sample, cancel).map_err(|e| match e {
            OrchestratorError::Cancelled => Failure::Cancelled,
            other => Failure::Unmeasured(format!(
                "asking the served candidate: {other}; brain serve said: {}",
                started.recent_output()
            )),
        })?;
    if let Ok(Some(status)) = started.running.0.try_wait() {
        return Err(Failure::Unmeasured(format!(
            "brain serve stopped answering ({status}): {}",
            started.recent_output()
        )));
    }
    measured.sampled = sample.tasks.len();
    for ((task, served), local) in sample.tasks.iter().zip(&probes).zip(in_process) {
        if served.verdict.is_none() && served.withheld.environment_failures > 0 {
            return Err(Failure::Unmeasured(format!(
                "the harness could not carry out the served attempt at task {}; brain serve \
                 said: {}",
                task.task.id,
                started.recent_output()
            )));
        }
        if served.verdict == local.verdict {
            measured.agreed += 1;
        } else {
            measured
                .disagreed
                .push(disagreement(&task.task.id, local, served));
        }
    }
    Ok(gate::serve(measured))
}

/// Publishes `adapter` (whose SHA-256 is `digest`) as the release a server
/// following `manifest` serves.
fn publish(manifest: &Path, adapter: &Path, digest: &str) -> Result<(), Failure> {
    let unmeasured = |why: String| Failure::Unmeasured(format!("the release manifest: {why}"));
    let digest = Digest::parse(digest).map_err(|e| unmeasured(e.to_string()))?;
    let adapter = std::path::absolute(adapter).map_err(|e| unmeasured(e.to_string()))?;
    splinter_model::manifest::write_adapter_manifest(manifest, &adapter, &digest)
        .map_err(|e| unmeasured(e.to_string()))
}

/// One task graded differently served than in-process.
fn disagreement(task: &Digest, in_process: &Probe, served: &Probe) -> Disagreement {
    Disagreement {
        task: task.to_string(),
        in_process: in_process.answer.as_deref().map(shortened),
        served: served.answer.as_deref().map(shortened),
        in_process_verdict: in_process.verdict,
        served_verdict: served.verdict,
    }
}

/// `patch`, cut to [`KEPT_PATCH_CHARS`].
fn shortened(patch: &str) -> String {
    match patch.char_indices().nth(KEPT_PATCH_CHARS) {
        Some((end, _)) => format!("{} ...", &patch[..end]),
        None => patch.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_long_patch_is_cut_for_the_report_and_a_short_one_kept() {
        assert_eq!(shortened("a small patch"), "a small patch");
        let long = "x".repeat(KEPT_PATCH_CHARS + 50);
        let cut = shortened(&long);
        assert!(cut.chars().count() < long.chars().count() && cut.ends_with(" ..."));
    }
}
