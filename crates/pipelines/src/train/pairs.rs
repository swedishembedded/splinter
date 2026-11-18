// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements preference and feedback fine-tuning of agent
// policies on their own verified attempts for its clients. If your team needs
// expertise in DPO, SimPO, ORPO or KTO on agent experience then you can procure
// our services by sending an email to info@swedishembedded.com.

//! The runs on preference pairs and judged completions: brain's `preference`
//! objective by the pair loss the run names, and its `feedback` objective
//! (KTO) at the class weights the datasets' manifests record unless the run
//! names others; and the refusal of a setting of the other objective.

use splinter_agent::CancelToken;
use splinter_core::training::Regime;
use splinter_data::holdout::split_dataset_file;
use splinter_model::served_train::{train_feedback_served, train_preference_served};
use splinter_model::train::{
    FeedbackTune, PairLoss, PreferenceTune, TrainedFeedback, TrainedPreference,
};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

use super::records::combine;
use super::{default_alpha, trainer_error, PairSettings, TrainPlan};

/// Trains the preference adapter `plan` describes into `plan.dir` by the pair
/// loss it names, a tenth of the pairs held out for brain's preference score.
pub(super) fn train_preference(
    ctx: &Context,
    plan: &TrainPlan,
    cancel: &CancelToken,
) -> Result<TrainedPreference, OrchestratorError> {
    let split = split_dataset_file(&combine(plan)?, &plan.dir)?;
    train_preference_served(
        &ctx.brain()?,
        &PreferenceTune {
            model_dir: &plan.base,
            train: &split.train,
            held_out: &split.held_out,
            attempt_dir: &plan.dir,
            steps: plan.steps,
            rank: plan.rank,
            alpha: plan
                .tuning
                .alpha
                .unwrap_or_else(|| default_alpha(plan.rank)),
            beta: plan.beta,
            nll_weight: plan.tuning.pairs.nll_weight.unwrap_or(0.0),
            loss: plan.tuning.pairs.loss.unwrap_or_default(),
            gamma: plan.tuning.pairs.gamma,
            grad_accum: 1,
            learning_rate: plan.tuning.learning_rate,
            seed: plan.seed,
            continue_from: plan.continue_from.as_deref(),
            cancel: Some(cancel),
        },
    )
    .map_err(trainer_error)
}

/// Trains the KTO adapter `plan` describes into `plan.dir` on every judged
/// completion of its datasets: KTO takes no held-out set, and the release
/// gate and the exam judge what it learned.
pub(super) fn train_feedback(
    ctx: &Context,
    plan: &TrainPlan,
    cancel: &CancelToken,
) -> Result<TrainedFeedback, OrchestratorError> {
    let train = combine(plan)?;
    let (desirable_weight, undesirable_weight) = feedback_weights(plan);
    train_feedback_served(
        &ctx.brain()?,
        &FeedbackTune {
            model_dir: &plan.base,
            train: &train,
            attempt_dir: &plan.dir,
            steps: plan.steps,
            rank: plan.rank,
            alpha: plan
                .tuning
                .alpha
                .unwrap_or_else(|| default_alpha(plan.rank)),
            beta: Some(plan.beta),
            desirable_weight,
            undesirable_weight,
            grad_accum: plan.tuning.records_per_step.unwrap_or(1),
            learning_rate: plan.tuning.learning_rate,
            seed: plan.seed,
            continue_from: plan.continue_from.as_deref(),
            cancel: Some(cancel),
        },
    )
    .map_err(trainer_error)
}

/// The class weights a KTO run trains `plan`'s datasets at: the ones named,
/// else the classes of every dataset together balanced as a feedback view
/// balances one ([`splinter_data::FeedbackBalance`]).
#[must_use]
pub fn feedback_weights(plan: &TrainPlan) -> (f32, f32) {
    let (desirable, undesirable) = plan
        .datasets
        .iter()
        .filter_map(|d| d.manifest.feedback)
        .fold((0, 0), |(d, u), b| (d + b.desirable, u + b.undesirable));
    let balance = splinter_data::FeedbackBalance::of(desirable, undesirable);
    let pairs = &plan.tuning.pairs;
    (
        pairs
            .desirable_weight
            .unwrap_or(balance.desirable_weight as f32),
        pairs
            .undesirable_weight
            .unwrap_or(balance.undesirable_weight as f32),
    )
}

/// Refuses the settings of an objective other than `regime`'s, by flag, and
/// SimPO's margin without the SimPO loss: a setting that would be silently
/// ignored is never accepted.
pub(super) fn check_settings(
    regime: Regime,
    pairs: &PairSettings,
) -> Result<(), OrchestratorError> {
    let refused = |named: Vec<&str>, what: &str| {
        OrchestratorError::Refused(format!(
            "{} apply to {what} datasets only; these datasets are trained by {}",
            named.join(", "),
            regime.label()
        ))
    };
    let preference = pairs.preference_named();
    if regime != Regime::Dpo && !preference.is_empty() {
        return Err(refused(preference, "preference"));
    }
    let feedback = pairs.feedback_named();
    if regime != Regime::Feedback && !feedback.is_empty() {
        return Err(refused(feedback, "feedback"));
    }
    if pairs.gamma.is_some() && pairs.loss != Some(PairLoss::Simpo) {
        return Err(OrchestratorError::Refused(
            "--gamma is the target margin of the SimPO loss; name --loss simpo".into(),
        ));
    }
    Ok(())
}
