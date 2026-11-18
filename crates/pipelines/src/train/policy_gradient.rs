// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reinforcement learning from the verified
// rewards of agent rollouts for its clients. If your team needs expertise in
// training agents against verifiers that do not trust them, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The policy-gradient regime of a training run: a dataset of rollout records
//! (each carrying the reward of its attempt and its group) trained by brain's
//! clipped policy gradient, on the advantages brain computes from each group's
//! rewards by the normalisation the datasets were built for. It holds nothing out and monitors nothing, so the settings
//! of a plan that could not honour that are refused here, not dropped.

use splinter_agent::CancelToken;
use splinter_core::training::Regime;
use splinter_data::{AdvantageFloor, AdvantageKind, AdvantageNorm};
use splinter_model::policy_gradient::{
    PolicyGradientOptions, PolicyGradientTune, TrainedPolicyGradient,
};
use splinter_model::served_train::train_policy_gradient_served;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

use super::records::{carries_reasoning, combine};
use super::trainer_error;
use super::{default_alpha, TrainPlan, Tuning, DEFAULT_WEIGHT_DECAY};

/// Trains the policy-gradient adapter `plan` describes into `plan.dir`.
pub(super) fn train(
    ctx: &Context,
    plan: &TrainPlan,
    cancel: &CancelToken,
) -> Result<TrainedPolicyGradient, OrchestratorError> {
    let train = combine(plan)?;
    let advantage = datasets_advantage(plan)?;
    let request = PolicyGradientTune {
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
        pack_answers: plan.datasets.iter().all(|d| d.manifest.packed),
        options: PolicyGradientOptions {
            advantage_norm: advantage.map(|(norm, _)| norm),
            advantage_floor: advantage.map(|(_, floor)| floor),
            ..plan.tuning.policy_gradient
        },
    };
    train_policy_gradient_served(&ctx.brain()?, &request).map_err(trainer_error)
}

/// How the datasets of `plan` were built to have their advantages computed
/// (their manifests name it): the normalisation and the floor; `None` when none
/// of them carries rewards. Datasets built for two are refused: one run
/// computes its advantages one way.
fn datasets_advantage(
    plan: &TrainPlan,
) -> Result<Option<(AdvantageNorm, AdvantageFloor)>, OrchestratorError> {
    let kinds: Vec<(String, AdvantageKind)> = plan
        .datasets
        .iter()
        .filter_map(|d| d.manifest.advantage.map(|k| (d.id.to_string(), k)))
        .collect();
    let read = |k: &AdvantageKind| (k.norm(), k.floor());
    let first = kinds.first().map(|(_, k)| read(k));
    if let Some((id, other)) = kinds.iter().find(|(_, k)| Some(read(k)) != first) {
        let says = |k: &AdvantageKind| {
            format!(
                "the {} advantage normalisation with floor {}",
                k.norm().name(),
                k.floor().name()
            )
        };
        return Err(OrchestratorError::Refused(format!(
            "dataset {id} was built for {} and {} for {}; one run computes its advantages \
             one way, so build them alike",
            says(other),
            kinds[0].0,
            says(&kinds[0].1)
        )));
    }
    Ok(first)
}

/// Refuses settings a run of `regime` cannot honour, so none is dropped
/// silently: the objective's own settings belong to a policy-gradient run, and
/// that run holds nothing out and monitors nothing.
pub(crate) fn check_settings(regime: Regime, tuning: &Tuning) -> Result<(), OrchestratorError> {
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

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use splinter_core::dataset::DatasetId;
    use splinter_core::digest::Digest;
    use splinter_data::{
        Manifest, Objective, Projection, Record, RecordBody, RecordMetadata, RolloutOutcome,
        StoredDataset, Unchecked,
    };

    use super::*;

    /// A stored rollout dataset of two records in one group, built for `kind`.
    fn rollouts(dir: &Path, name: &str, kind: Option<AdvantageKind>) -> StoredDataset {
        let record = |reward: f64, at: &str| Record {
            body: RecordBody::chat(vec![splinter_core::chat::WireMessage {
                role: "assistant".into(),
                content: "a".into(),
                images: Vec::new(),
                tool_calls: Vec::new(),
                tool_call_id: None,
                reasoning_content: None,
                train: true,
            }])
            .with_outcome(RolloutOutcome::of(reward, &["task"], at)),
            metadata: RecordMetadata {
                experiences: Vec::new(),
                task: None,
                sources: Vec::new(),
                group: None,
                split: None,
                split_draw: None,
                view: "policy-gradient".into(),
                objective: Objective::PolicyGradient,
            },
        };
        let mut projection =
            Projection::new("policy-gradient", Objective::PolicyGradient, None, None);
        projection.records = vec![record(1.0, "0"), record(0.0, "1")];
        projection.advantage = kind;
        let path = dir.join(format!("{name}.jsonl"));
        splinter_data::write_dataset(&path, &projection, &Unchecked).unwrap();
        let manifest: Manifest = serde_json::from_str(
            &std::fs::read_to_string(splinter_data::manifest_path(&path)).unwrap(),
        )
        .unwrap();
        StoredDataset {
            id: DatasetId(Digest::of(name.as_bytes())),
            path,
            manifest,
        }
    }

    fn plan(dir: &Path, datasets: Vec<StoredDataset>) -> TrainPlan {
        TrainPlan {
            candidate: "candidate".into(),
            regime: Regime::PolicyGradient,
            dir: dir.to_path_buf(),
            datasets,
            base: PathBuf::from("base"),
            continue_from: None,
            parent: None,
            replay_file: None,
            replay_share: None,
            rehearsal: None,
            steps: 1,
            rank: 2,
            beta: 0.1,
            tuning: Tuning::default(),
            train_precision: None,
            seed: 1, // seed: no draw is made; any value
            eval_every: 0,
            patience: 0,
            monitor_share: 0.0,
            resume_key: None,
        }
    }

    /// The trainer is asked for the normalisation and floor the datasets were
    /// built for, and datasets built for two are refused rather than trained
    /// one way silently.
    #[test]
    fn the_run_normalises_as_its_datasets_were_built_and_refuses_a_mix() {
        let dir = tempfile::tempdir().unwrap();
        let loo = rollouts(dir.path(), "loo", Some(AdvantageKind::LeaveOneOut));
        let positive = rollouts(dir.path(), "positive", Some(AdvantageKind::Positive));
        let standardised = rollouts(
            dir.path(),
            "standardised",
            Some(AdvantageKind::Standardised),
        );
        let plain = rollouts(dir.path(), "plain", None);
        assert_eq!(
            datasets_advantage(&plan(dir.path(), vec![loo.clone(), plain.clone()])).unwrap(),
            Some((AdvantageNorm::LeaveOneOut, AdvantageFloor::None))
        );
        assert_eq!(
            datasets_advantage(&plan(dir.path(), vec![positive.clone()])).unwrap(),
            Some((AdvantageNorm::ZScore, AdvantageFloor::Zero)),
            "positive standardises and floors at zero"
        );
        let floors = datasets_advantage(&plan(dir.path(), vec![positive, standardised]));
        assert!(
            matches!(&floors, Err(OrchestratorError::Refused(why)) if why.contains("floor zero") && why.contains("floor none")),
            "{floors:?}"
        );
        assert_eq!(
            datasets_advantage(&plan(dir.path(), vec![plain])).unwrap(),
            None
        );
        let mixed = datasets_advantage(&plan(
            dir.path(),
            vec![
                loo,
                rollouts(dir.path(), "z", Some(AdvantageKind::Standardised)),
            ],
        ));
        assert!(
            matches!(&mixed, Err(OrchestratorError::Refused(why)) if why.contains("leave-one-out") && why.contains("z-score")),
            "{mixed:?}"
        );
    }
}
