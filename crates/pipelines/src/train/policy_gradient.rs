// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reinforcement learning from the verified
// rewards of agent rollouts for its clients. If your team needs expertise in
// training agents against verifiers that do not trust them, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The policy-gradient regime of a training run: a dataset of rollout records
//! (each carrying the advantage of its attempt) trained by brain's clipped
//! policy gradient. It holds nothing out and monitors nothing, so the settings
//! of a plan that could not honour that are refused here, not dropped.

use splinter_agent::CancelToken;
use splinter_core::training::Regime;
use splinter_model::policy_gradient::{
    train_policy_gradient, PolicyGradientOptions, PolicyGradientTune, TrainedPolicyGradient,
};
use splinter_orchestrator::error::OrchestratorError;

use super::records::{carries_reasoning, combine};
use super::trainer_error;
use super::{default_alpha, TrainPlan, Tuning, DEFAULT_WEIGHT_DECAY};

/// Trains the policy-gradient adapter `plan` describes into `plan.dir`.
pub(super) fn train(
    plan: &TrainPlan,
    cancel: &CancelToken,
) -> Result<TrainedPolicyGradient, OrchestratorError> {
    let train = combine(plan)?;
    train_policy_gradient(&PolicyGradientTune {
        model_dir: &plan.base,
        train: &train,
        attempt_dir: &plan.dir,
        steps: plan.steps,
        rank: plan.rank,
        alpha: plan
            .tuning
            .alpha
            .unwrap_or_else(|| default_alpha(plan.rank)),
        weight_decay: plan.tuning.weight_decay.unwrap_or(DEFAULT_WEIGHT_DECAY),
        grad_accum: plan.tuning.records_per_step.unwrap_or(1),
        continue_from: plan.continue_from.as_deref(),
        cancel: Some(cancel),
        base: plan.tuning.base_precision.unwrap_or_else(|| {
            splinter_model::train::BasePrecision::bf16_if(plan.tuning.bf16_base)
        }),
        learning_rate: plan.tuning.learning_rate,
        seed: plan.seed,
        max_block: plan.tuning.max_block,
        thinking: carries_reasoning(&train)?,
        options: plan.tuning.policy_gradient,
    })
    .map_err(trainer_error)
}

/// Refuses settings a run of `regime` cannot honour, so none is dropped
/// silently: the objective's own settings belong to a policy-gradient run, and
/// that run holds nothing out and monitors nothing.
pub(super) fn check_settings(regime: Regime, tuning: &Tuning) -> Result<(), OrchestratorError> {
    let refused =
        |what: String, why: &str| Err(OrchestratorError::Refused(format!("{what} {why}")));
    if regime != Regime::PolicyGradient {
        let named = tuning.policy_gradient.named();
        if !named.is_empty() {
            return refused(
                named.join(", "),
                "set the policy-gradient objective, which only a dataset of rollout records \
                 (the policy-gradient view) trains",
            );
        }
        return Ok(());
    }
    let monitoring: Vec<&str> = [
        (tuning.eval_every.is_some_and(|n| n > 0), "--eval-every"),
        (tuning.patience.is_some_and(|n| n > 0), "--patience"),
        (tuning.monitor_share.is_some(), "--monitor-share"),
        (tuning.keep_evaluations, "--keep-evaluations"),
    ]
    .into_iter()
    .filter_map(|(named, flag)| named.then_some(flag))
    .collect();
    if monitoring.is_empty() {
        Ok(())
    } else {
        refused(
            monitoring.join(", "),
            "monitor a run, but a policy-gradient run holds nothing out and has no monitoring \
             loss; its exam is the judge",
        )
    }
}

/// The policy-gradient settings a plan records: every default resolved for a
/// run that trains the objective, none for any other.
pub(super) fn resolved(regime: Regime, named: PolicyGradientOptions) -> PolicyGradientOptions {
    if regime != Regime::PolicyGradient {
        return named;
    }
    let resolved = named.resolved();
    PolicyGradientOptions {
        clip_eps: Some(resolved.clip_eps),
        kl_beta: Some(resolved.kl_beta),
        reference: Some(resolved.reference),
        loss_agg: Some(resolved.loss_agg),
        temperature: Some(resolved.temperature),
    }
}
