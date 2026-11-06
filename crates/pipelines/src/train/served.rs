// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! Which training runs have a job on the brain service, and what a run does
//! that has none.
//!
//! With brain endpoints the service owns the cards: a supervised fine-tune and
//! a policy-gradient run are its jobs. A regime without a job (preference
//! optimisation, a full-parameter fine-tune, merging an adapter) would load a
//! model in this process on cards the service is using, which is the
//! contention endpoints exist to remove; it is refused by name rather than
//! run beside the service.

use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// Refuses `what` when the run has brain endpoints.
pub(super) fn refuse_unserved(ctx: &Context, what: &str) -> Result<(), OrchestratorError> {
    if ctx.config().brain.is_some() {
        return Err(OrchestratorError::Refused(format!(
            "{what} has no job on the brain service yet, and would load the model in this \
             process on cards the service is using; run it without --brain-endpoint"
        )));
    }
    Ok(())
}
