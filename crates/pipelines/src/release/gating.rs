// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! The release gate run on a candidate: the suites it is graded on, both
//! arms graded on each, and the four checks decided over the outcomes.

use splinter_agent::CancelToken;
use splinter_core::model_ref::ModelRef;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

use super::probe::{self, Arm, Probe, Suite};

/// One model's answer and outcome on each task of a suite.
pub(crate) type Outcomes = Vec<Probe>;

/// A suite's outcomes for one model, or why there are none.
pub(crate) type Graded = Result<Outcomes, String>;

/// `result`, with every failure but a cancel turned into its reason: a
/// check that cannot be measured fails the gate, it does not abort it.
pub(super) fn soft<T>(
    result: Result<T, OrchestratorError>,
) -> Result<Result<T, String>, OrchestratorError> {
    match result {
        Ok(value) => Ok(Ok(value)),
        Err(OrchestratorError::Cancelled) => Err(OrchestratorError::Cancelled),
        Err(e) => Ok(Err(e.to_string())),
    }
}

/// Which system prompt an arm is asked a suite under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Asked {
    /// The one the arm was trained under, as it is deployed.
    Deployed,
    /// The default one, whichever arm: for a suite that measures the weights
    /// and not the persona.
    Default,
}

/// The model `reference` names graded on each of `suites`, decoding
/// greedily ([`probe::greedy`]) where its sampling can be set here - a
/// verdict is then the weights', not a draw's, and the served candidate is
/// asked the same way. A suite is asked under the arm's `deployed` prompt
/// or the default one, as its [`Asked`] says. The arms differ only by
/// adapter, so the second one graded runs on the base the first one loaded;
/// the serve check releases it before the served candidate starts.
pub(crate) fn grade_arm(
    ctx: &Context,
    reference: &ModelRef,
    deployed: Option<&str>,
    suites: &[(&Suite, Asked)],
    cancel: &CancelToken,
) -> Result<Vec<Graded>, OrchestratorError> {
    let model = match soft(probe::greedy(ctx, reference))? {
        Ok(model) => model,
        Err(why) => return Ok(suites.iter().map(|_| Err(why.clone())).collect()),
    };
    let mut graded = Vec::with_capacity(suites.len());
    for (suite, asked) in suites {
        let asked_model = match (asked, deployed) {
            (Asked::Deployed, Some(system)) => model.clone().with_system(system),
            _ => model.clone(),
        };
        graded.push(soft(probe::grade_on(
            ctx,
            &Arm::named(reference, &asked_model),
            suite,
            cancel,
        ))?);
    }
    Ok(graded)
}
