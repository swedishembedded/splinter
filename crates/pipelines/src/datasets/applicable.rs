// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-set curation that projects
// verified agent experience into supervised datasets, for its clients. If
// your team needs expertise in dataset curation for fine-tuning, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Which options of a `dataset build` apply to its view: an option the view
//! cannot honour is refused, never silently ignored.

use splinter_core::chat::LossWeight;
use splinter_orchestrator::error::OrchestratorError;

use super::{BuildRequest, Control, HistoryCuration, ViewName};

/// Refuses an option `request`'s view does not take; the transition view's
/// loss weight, checked, when it is given.
pub(super) fn refuse_inapplicable(
    request: &BuildRequest,
) -> Result<Option<LossWeight>, OrchestratorError> {
    let view = request.view;
    if request.strip.is_some() && !view.takes_strip() {
        return Err(OrchestratorError::Refused(format!(
            "view {} shows the student no task, so --strip does not apply",
            view.as_str()
        )));
    }
    if request.pack && !matches!(view, ViewName::SftStep | ViewName::PolicyGradient) {
        return Err(OrchestratorError::Refused(format!(
            "view {} has no packed form, so --pack does not apply",
            view.as_str()
        )));
    }
    let judges_runs = matches!(view, ViewName::Preference | ViewName::Feedback);
    if request.history != HistoryCuration::default() && view != ViewName::SftStep && !judges_runs {
        return Err(OrchestratorError::Refused(format!(
            "view {} imitates no demonstration, so --repository-history does not apply",
            view.as_str()
        )));
    }
    let shuffled_transitions = view == ViewName::Transition
        && matches!(request.control, Control::ShuffledInstructions { .. });
    if request.control != Control::Verified && view != ViewName::SftStep && !shuffled_transitions {
        return Err(OrchestratorError::Refused(format!(
            "view {} has no negative controls, so --control does not apply",
            view.as_str()
        )));
    }
    if request.serialise_calls
        && !judges_runs
        && !matches!(
            view,
            ViewName::SftStep
                | ViewName::PolicyGradient
                | ViewName::SelfDistill
                | ViewName::Transition
        )
    {
        return Err(OrchestratorError::Refused(format!(
            "view {} has no run to take apart, so --serialise-calls does not apply",
            view.as_str()
        )));
    }
    let trains_actions = matches!(
        view,
        ViewName::SftStep | ViewName::PolicyGradient | ViewName::SelfDistill
    );
    if request.images && !trains_actions {
        return Err(OrchestratorError::Refused(format!(
            "view {} has no conversations that were shown images, so --images does not apply",
            view.as_str()
        )));
    }
    if !request.transition.is_default() && view != ViewName::Transition {
        return Err(OrchestratorError::Refused(format!(
            "view {} supervises no observation, so --loss-weight and --per-transition-cap do not \
             apply",
            view.as_str()
        )));
    }
    let transition_weight = request
        .transition
        .loss_weight
        .map(|weight| {
            LossWeight::new(weight).ok_or_else(|| {
                OrchestratorError::Refused(format!(
                    "--loss-weight {weight} is not a finite weight above zero"
                ))
            })
        })
        .transpose()?;
    if !request.rl.pairing_is_default() && view != ViewName::Preference {
        return Err(OrchestratorError::Refused(format!(
            "view {} makes no pairs, so --pairs-per-group and --pair-seed do not apply",
            view.as_str()
        )));
    }
    let rewards_attempts = matches!(view, ViewName::PolicyGradient | ViewName::SelfDistill);
    if !request.rl.rewards_are_default() && !rewards_attempts {
        return Err(OrchestratorError::Refused(format!(
            "view {} has no rewards or advantages, so --reward, --reward-metric, \
             --fail-ceiling, --advantage, --cliff-distill, --sampled-only and --overlong-max-tokens do \
             not apply",
            view.as_str()
        )));
    }
    if !request.practice.is_empty() && !rewards_attempts {
        return Err(OrchestratorError::Refused(format!(
            "view {} rewards no attempt, so --practice does not apply",
            view.as_str()
        )));
    }
    if request.rl.cliff_distill && view != ViewName::PolicyGradient {
        return Err(OrchestratorError::Refused(
            "--cliff-distill distils the all-fail groups of the policy-gradient view; the \
             self-distill view distils every attempt already"
                .into(),
        ));
    }
    if request.min_strength.is_some() && !view.takes_min_strength() {
        return Err(OrchestratorError::Refused(format!(
            "view {} reads no verdict, so --min-strength does not apply",
            view.as_str()
        )));
    }
    Ok(transition_weight)
}
