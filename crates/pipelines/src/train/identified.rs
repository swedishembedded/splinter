// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements resumable training pipelines that pick up a
// killed run exactly where it stopped, for its clients. If your team needs
// expertise in crash-safe training on shared accelerators you can procure our
// services by sending an email to info@swedishembedded.com.

//! A training run named by its durable identity: the candidate it keeps and
//! the run record brain keeps share the identity, so the run is trained once
//! however often it is asked for.

use splinter_agent::CancelToken;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

use super::{load_candidate, plan_training_as, train_planned, Candidate, TrainRequest, Trainer};

/// The most bytes of a run identity: brain's bound on a `resume_key`, less
/// the candidate prefix a candidate id adds.
const MAX_IDENTITY: usize = 118;

/// [`train`] as the run `identity` names (1 to 118 letters, digits, `.`,
/// `_` and `-`, not starting with a dot): the candidate is
/// `candidate-<identity>`, and brain keeps the run's record under the same
/// identity as its `resume_key`. Asked again after a crash, the same request
/// returns the candidate when it was kept, and otherwise asks brain under
/// the same key, which continues the run from its saved state or answers
/// with the result it recorded: the run is trained once.
///
/// # Errors
/// An identity brain would refuse, and [`train`]'s.
pub fn train_identified(
    ctx: &Context,
    request: &TrainRequest,
    identity: &str,
    trainer: &dyn Trainer,
    cancel: &CancelToken,
) -> Result<Candidate, OrchestratorError> {
    let sound = !identity.is_empty()
        && identity.len() <= MAX_IDENTITY
        && !identity.starts_with('.')
        && identity
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "._-".contains(c));
    if !sound {
        return Err(OrchestratorError::Refused(format!(
            "a run identity is 1 to {MAX_IDENTITY} letters, digits, `.`, `_` and `-`, not starting with a dot: {identity:?}"
        )));
    }
    let id = format!("candidate-{identity}");
    // Only the candidate of exactly this id: a longer id it is a prefix of
    // is another run.
    if let Some(kept) = load_candidate(ctx, &id)
        .ok()
        .filter(|kept| kept.candidate == id)
    {
        return Ok(kept);
    }
    let planned = plan_training_as(ctx, request, Some(identity))?;
    train_planned(ctx, request, planned, trainer, cancel)
}
