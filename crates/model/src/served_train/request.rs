// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems whose training runs as a
// job on a shared model service, for its clients. If your team needs
// expertise in turning a training configuration into a service request, you
// can procure our services by sending an email to info@swedishembedded.com.

//! A training configuration as the parameters of brain's `brain/train`
//! action: the same settings the in-process trainers take, with each dataset
//! published to the artifact store and named by digest.

use std::path::Path;

use serde_json::{json, Map, Value};

use crate::endpoint::BrainEndpoints;
use crate::error::PolicyError;
use crate::policy_gradient::{
    Behaviour, LossAggregation, PolicyGradientOptions, PolicyGradientTune, Reference,
};
use crate::selection::local_model_name;
use crate::train::{candidate_adapter_id, FineTune};
use crate::train_checks::check_chat_datasets;

/// A dataset published to the store, as the job's `data` document names it.
pub(super) fn source(brain: &BrainEndpoints, file: &Path) -> Result<Value, PolicyError> {
    Ok(json!({ "digest": brain.artifact_store()?.publish(file)?.hex() }))
}

pub(super) fn set(params: &mut Map<String, Value>, name: &str, value: impl Into<Value>) {
    params.insert(name.to_string(), value.into());
}

/// The parameters of a supervised fine-tune.
pub(super) fn fine_tune_params(
    brain: &BrainEndpoints,
    request: &FineTune<'_>,
    block: u32,
) -> Result<Value, PolicyError> {
    let model = local_model_name(request.model_dir);
    let mut data = Map::new();
    set(&mut data, "train", source(brain, request.train)?);
    set(&mut data, "held_out", source(brain, request.held_out)?);
    if let Some(text) = request.held_out_text {
        set(&mut data, "held_out_text", source(brain, text)?);
    }
    if let Some(monitor) = request.monitor {
        set(&mut data, "monitor", source(brain, monitor)?);
    }
    if let Some(adapter) = request.continue_from {
        set(&mut data, "start_adapter", source(brain, adapter)?);
    }
    if !request.replay.is_empty() {
        let replay: Result<Vec<Value>, PolicyError> =
            request.replay.iter().map(|p| source(brain, p)).collect();
        set(&mut data, "replay", replay?);
    }
    if !request.weighted_replay.is_empty() {
        let mut weighted = Vec::new();
        for (path, share) in request.weighted_replay {
            let mut entry = source(brain, path)?;
            entry["share"] = json!(share);
            weighted.push(entry);
        }
        set(&mut data, "weighted_replay", weighted);
    }

    let mut params = Map::new();
    set(&mut params, "base", model.as_str());
    set(&mut params, "data", Value::Object(data).to_string());
    set(&mut params, "objective", "sft");
    set(&mut params, "steps", request.steps);
    // A continued adapter keeps its own rank and alpha; brain refuses a
    // continuation asked for at another rank.
    if request.continue_from.is_none() {
        set(&mut params, "rank", request.rank);
        set(&mut params, "alpha", request.alpha);
    }
    set(&mut params, "weight_decay", request.weight_decay);
    set(&mut params, "grad_accum", request.grad_accum);
    set(&mut params, "seed", request.seed);
    set(&mut params, "max_block", block);
    set(&mut params, "thinking", request.thinking);
    set(&mut params, "pack_answers", request.pack_answers);
    set(&mut params, "eval_every", request.eval_every);
    set(&mut params, "patience", request.patience);
    set(
        &mut params,
        "keep_evaluations",
        request.keep_evaluations.is_some(),
    );
    set(
        &mut params,
        "adapter_id",
        candidate_adapter_id(&model, request.attempt_dir)?,
    );
    if let Some(format) = request.base.brain_format() {
        set(&mut params, "base_format", format);
    }
    if let Some(lr) = request.learning_rate {
        set(&mut params, "lr", lr);
    }
    if let Some(share) = request.replay_share {
        set(&mut params, "replay_share", share);
    }
    if let Some(key) = request.resume_key {
        set(&mut params, "resume_key", key);
    }
    Ok(Value::Object(params))
}

/// The parameters of a policy-gradient run, and the records its dataset holds.
pub(super) fn policy_gradient_params(
    brain: &BrainEndpoints,
    request: &PolicyGradientTune<'_>,
) -> Result<(Value, usize), PolicyError> {
    let checked = check_chat_datasets(
        brain,
        request.model_dir,
        &[request.train],
        1,
        (request.thinking, request.pack_answers),
        request.max_block,
    )?;
    let (records, block) = (checked.records, checked.block);
    let model = local_model_name(request.model_dir);
    let mut data = Map::new();
    set(&mut data, "train", source(brain, request.train)?);
    if let Some(adapter) = request.continue_from {
        set(&mut data, "start_adapter", source(brain, adapter)?);
    }
    let mut params = Map::new();
    set(&mut params, "base", model.as_str());
    set(&mut params, "data", Value::Object(data).to_string());
    set(&mut params, "objective", "policy_gradient");
    set(&mut params, "steps", request.steps);
    // A continued adapter keeps its own rank and alpha; brain refuses a
    // continuation asked for at another rank.
    if request.continue_from.is_none() {
        set(&mut params, "rank", request.rank);
        set(&mut params, "alpha", request.alpha);
    }
    set(&mut params, "weight_decay", request.weight_decay);
    set(&mut params, "grad_accum", request.grad_accum);
    set(&mut params, "seed", request.seed);
    set(&mut params, "max_block", block);
    set(&mut params, "thinking", request.thinking);
    set(&mut params, "pack_answers", request.pack_answers);
    set(
        &mut params,
        "adapter_id",
        candidate_adapter_id(&model, request.attempt_dir)?,
    );
    if let Some(format) = request.base.brain_format() {
        set(&mut params, "base_format", format);
    }
    if let Some(lr) = request.learning_rate {
        set(&mut params, "lr", lr);
    }
    options(&mut params, &request.options);
    Ok((Value::Object(params), records))
}

/// The objective's settings that were named; the rest are brain's defaults.
fn options(params: &mut Map<String, Value>, options: &PolicyGradientOptions) {
    let reference = |r: Reference| match r {
        Reference::Base => "base",
        Reference::StartOfRun => "start",
    };
    if let Some(v) = options.algorithm {
        set(params, "algorithm", v.name());
    }
    if let Some(v) = options.clip_eps {
        set(params, "clip_eps", v);
    }
    if let Some(v) = options.clip_eps_high {
        set(params, "clip_eps_high", v);
    }
    if let Some(v) = options.max_step_kl {
        set(params, "max_step_kl", v);
    }
    if let Some(v) = options.max_step_grad_norm {
        set(params, "max_step_grad_norm", v);
    }
    if let Some(v) = options.kl_beta {
        set(params, "kl_beta", v);
    }
    if let Some(v) = options.reference {
        set(params, "reference", reference(v));
    }
    if let Some(v) = options.behaviour {
        set(
            params,
            "behaviour",
            match v {
                Behaviour::Recorded => "recorded",
                Behaviour::StartOfRun => "start_of_run_proxy",
            },
        );
    }
    if let Some(v) = options.loss_agg {
        set(
            params,
            "loss_agg",
            match v {
                LossAggregation::SeqMean => "seq-mean",
                LossAggregation::TokenSum => "token-sum",
            },
        );
    }
    if let Some(v) = options.temperature {
        set(params, "temperature", v);
    }
    if let Some(v) = options.teacher {
        set(params, "teacher", reference(v));
    }
    if let Some(v) = options.advantage_bound {
        set(params, "advantage_bound", v);
    }
    if let Some(v) = options.sequence_advantage_cap {
        set(params, "sequence_advantage_cap", v);
    }
    if let Some(v) = options.advantage_norm {
        set(params, "advantage_norm", v.name());
    }
    if let Some(v) = options.advantage_floor {
        set(params, "advantage_floor", v.name());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_settings_that_were_named_are_sent_and_they_use_brains_spellings() {
        let mut params = Map::new();
        options(&mut params, &PolicyGradientOptions::default());
        assert!(params.is_empty(), "brain's defaults are brain's to apply");
        options(
            &mut params,
            &PolicyGradientOptions {
                algorithm: Some(crate::policy_gradient::Algorithm::Dapo),
                clip_eps: Some(0.3),
                clip_eps_high: Some(0.28),
                max_step_kl: Some(0.5),
                max_step_grad_norm: Some(10.0),
                kl_beta: Some(0.0),
                reference: Some(Reference::StartOfRun),
                behaviour: Some(Behaviour::StartOfRun),
                loss_agg: Some(LossAggregation::TokenSum),
                temperature: Some(0.6),
                teacher: Some(Reference::Base),
                advantage_bound: Some(4.0),
                sequence_advantage_cap: Some(30.0),
                advantage_norm: Some(splinter_data::AdvantageNorm::LeaveOneOut),
                advantage_floor: Some(splinter_data::AdvantageFloor::Zero),
            },
        );
        assert_eq!(params["reference"], "start");
        assert_eq!(params["loss_agg"], "token-sum");
        assert_eq!(params["behaviour"], "start_of_run_proxy");
        assert_eq!(params["teacher"], "base");
        assert_eq!(params["advantage_norm"], "leave-one-out");
        assert_eq!(params["advantage_floor"], "zero");
        assert_eq!(params["algorithm"], "dapo");
        assert_eq!(params.len(), 15);
        assert_eq!(
            params["clip_eps_high"]
                .as_f64()
                .map(|v| (v * 100.0).round()),
            Some(28.0)
        );
    }
}
