// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from a failed agent attempt to a
// verified corrected one, for its clients. If your team needs expertise in
// resumable, idempotent pipelines over agent experience you can procure our
// services by sending an email to info@swedishembedded.com.

//! The keys of rationalisation records: what makes an attempt the one it is,
//! so a rerun finds what it decided and a different run never does.
//!
//! An attempt is keyed by the student's trial, the rationaliser, the sample
//! and the run's variant: the profile, the endpoint's model id, the leak
//! terms, and the hint framing and checks when they are not what every run
//! before them used (the worked-answer framing, every check on), so those
//! runs keep their keys. A re-graded attempt is keyed by the attempt it
//! re-grades and the re-grade's variant.

use serde::Serialize;
use serde_json::json;
use splinter_core::digest::{canonical_json, Digest};
use splinter_orchestrator::error::OrchestratorError;

use super::filters::HintFraming;
use super::material::{StudentTrial, TrainTask};
use super::{refused, Checks, RATIONALISE_FORMAT};
use crate::bench::repair::route::InterventionKind;

/// The hex digest of `value`'s canonical JSON.
pub(super) fn digest_of<T: Serialize>(value: &T) -> Result<String, OrchestratorError> {
    let bytes = canonical_json(value).map_err(|e| refused(format!("a key: {e}")))?;
    Ok(Digest::of(&bytes).hex().to_string())
}

/// The key of the student's own pass `trial` of `task`.
pub(super) fn own_key(trial: &StudentTrial, task: &TrainTask) -> Result<String, OrchestratorError> {
    digest_of(&json!({
        "format": RATIONALISE_FORMAT, "origin": "own", "trial": trial.trial, "task": task.rel,
    }))
}

/// The variant of a run that draws and runs its attempts.
pub(super) fn draw_variant(
    profile_sha256: &str,
    served_as: &str,
    leak_terms: &[String],
    framing: HintFraming,
    checks: Checks,
) -> Result<String, OrchestratorError> {
    let mut variant = json!({
        "profile_sha256": profile_sha256,
        "served_as": served_as,
        "leak_terms": leak_terms,
    });
    // What every run made before the option used adds nothing, so their records keep their keys.
    if framing != HintFraming::default() {
        variant["framing"] = json!(framing);
    }
    if checks != Checks::ALL {
        variant["checks"] = json!(checks);
    }
    digest_of(&variant)
}

/// The key of sample `sample` of the attempt at the student's `trial` by
/// `rationaliser` under `variant`.
pub(super) fn attempt_key(
    trial: &str,
    rationaliser: &str,
    sample: usize,
    variant: &str,
) -> Result<String, OrchestratorError> {
    digest_of(&json!({
        "format": RATIONALISE_FORMAT,
        "intervention": InterventionKind::HintedRetry.as_str(),
        "student": trial,
        "rationaliser": rationaliser,
        "sample": sample,
        "variant": variant,
    }))
}

/// The variant of a re-grade: the leak terms and the checks, and, when the
/// sufficiency run is on, what runs it (the profile's sha256, the endpoint's
/// model id and the rationaliser's identity).
pub(super) fn regrade_variant(
    leak_terms: &[String],
    checks: Checks,
    runner: Option<(&str, &str, &str)>,
) -> Result<String, OrchestratorError> {
    let mut variant = json!({"regrade": true, "leak_terms": leak_terms, "checks": checks});
    if let Some((profile_sha256, served_as, rationaliser)) = runner {
        variant["profile_sha256"] = json!(profile_sha256);
        variant["served_as"] = json!(served_as);
        variant["rationaliser"] = json!(rationaliser);
    }
    digest_of(&variant)
}

/// The key of the re-grade of the attempt keyed `source` under `variant`.
pub(super) fn regrade_key(source: &str, variant: &str) -> Result<String, OrchestratorError> {
    digest_of(&json!({
        "format": RATIONALISE_FORMAT,
        "intervention": InterventionKind::HintedRetry.as_str(),
        "regrade_of": source,
        "variant": variant,
    }))
}

#[cfg(test)]
#[path = "keys_tests.rs"]
mod tests;
