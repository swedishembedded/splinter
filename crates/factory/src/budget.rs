// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training pipelines whose budgets keep their
// meaning as datasets grow, for its clients. If your team needs expertise in
// fine-tuning on cumulative datasets, you can procure our services by sending
// an email to info@swedishembedded.com.

//! A `model.train` budget stated in passes over the data instead of
//! optimizer steps.
//!
//! A loop that retrains from the base on a dataset that grows with replay
//! under a fixed step count trains fewer passes at every generation (1.89,
//! 0.97, then 0.63 epochs at 800 steps of 4 records). `params.epochs` states
//! the passes; the factory resolves it, against the records the stage draws
//! from, into the trainer's `steps` before the trainer sees the parameters:
//!
//! `steps = ceil(epochs * draws_per_epoch / records_per_step)`
//!
//! - Without `replay_share`, the dataset and the replayed records are one
//!   union drawn uniformly: `draws_per_epoch = dataset + replay`, so every
//!   record of either is seen `epochs` times.
//! - With `replay_share = s`, a share `s` of the draws comes from the replay
//!   and `1 - s` from the dataset: `draws_per_epoch = dataset / (1 - s)`, so
//!   every record of the dataset is seen `epochs` times (the trainer realises
//!   the share to within its own mixing tolerance); a replayed record is seen
//!   as often as its share and size make it.
//! - `records_per_step` is what one optimizer step of the trainer consumes
//!   ([`crate::training::Trainer::records_per_step`]).

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The parameter that states the budget in passes over the data.
pub const EPOCHS: &str = "epochs";

/// The trainer parameter an epoch budget resolves into.
pub const STEPS: &str = "steps";

/// The records a training stage draws from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Draws {
    /// The training records of the dataset trained.
    pub dataset: u64,
    /// The training records of every replayed dataset together.
    pub replay: u64,
    /// The share of the draws that comes from the replay, if one is set.
    pub replay_share: Option<f64>,
}

impl Draws {
    /// The draws in which every record of the dataset comes up once.
    #[must_use]
    pub fn per_epoch(&self) -> f64 {
        match self.replay_share {
            Some(share) => self.dataset as f64 / (1.0 - share),
            None => (self.dataset + self.replay) as f64,
        }
    }
}

/// An epoch budget as it was resolved, recorded in the stage's identity, its
/// report and the training record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EpochBudget {
    /// The passes over the dataset asked for.
    pub epochs: f64,
    /// The dataset's training records.
    pub dataset_records: u64,
    /// The replayed training records.
    pub replay_records: u64,
    /// The share of the draws from the replay, if one is set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replay_share: Option<f64>,
    /// The draws per epoch ([`Draws::per_epoch`]).
    pub draws_per_epoch: f64,
    /// The records one optimizer step consumes.
    pub records_per_step: u64,
    /// The optimizer steps the budget resolved to.
    pub steps: u32,
}

/// What is wrong with the budget of `params` before any data is counted:
/// `epochs` beside `steps`, or `epochs` that is not a positive number.
#[must_use]
pub fn problems(params: &Value) -> Vec<String> {
    let Some(epochs) = params.get(EPOCHS) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    if params.get(STEPS).is_some() {
        out.push(
            "params: epochs and steps both state the budget; give one (epochs scales with the records drawn, steps does not)"
                .to_owned(),
        );
    }
    if !epochs.as_f64().is_some_and(|e| e.is_finite() && e > 0.0) {
        out.push(format!(
            "params: epochs must be a positive number, got {epochs}"
        ));
    }
    out
}

/// Whether `params` states its budget in epochs.
#[must_use]
pub fn in_epochs(params: &Value) -> bool {
    params.get(EPOCHS).is_some()
}

/// `params` as a trainer can check them before the records exist: an epoch
/// budget stands in as one step, since what a trainer accepts does not depend
/// on how many steps run; `params` unchanged when they state steps.
#[must_use]
pub fn for_trainer_check(params: &Value) -> Value {
    let mut checked = params.clone();
    if in_epochs(params) {
        if let Some(object) = checked.as_object_mut() {
            object.remove(EPOCHS);
            object.insert(STEPS.into(), Value::from(1));
        }
    }
    checked
}

/// `params` with an epoch budget resolved into `steps` over `draws`, and the
/// budget as resolved; `params` unchanged and no budget when it states none.
/// `records_per_step` is asked with the parameters the trainer will get,
/// before they hold a step count.
///
/// # Errors
/// The budget is refused ([`problems`]), the dataset holds no records, or the
/// steps do not fit.
pub fn resolve(
    params: &Value,
    draws: Draws,
    records_per_step: impl FnOnce(&Value) -> Result<u64, String>,
) -> Result<(Value, Option<EpochBudget>), String> {
    if !in_epochs(params) {
        return Ok((params.clone(), None));
    }
    let refused = problems(params);
    if !refused.is_empty() {
        return Err(refused.join("; "));
    }
    let epochs = params[EPOCHS].as_f64().unwrap_or_default();
    if draws.dataset == 0 {
        return Err("params: epochs over a dataset with no training records".to_owned());
    }
    let mut trainer_params = params.clone();
    if let Some(map) = trainer_params.as_object_mut() {
        map.remove(EPOCHS);
    }
    let per_step = records_per_step(&trainer_params)?;
    if per_step == 0 {
        return Err("the trainer consumes no records per step".to_owned());
    }
    let per_epoch = draws.per_epoch();
    let exact = epochs * per_epoch / per_step as f64;
    // A product that is a whole number in exact arithmetic (2 epochs of 100
    // records at 4 a step) must not round up past it on a last-bit error.
    let steps = (exact - exact.max(1.0) * 1e-9).ceil().max(1.0);
    if steps > f64::from(u32::MAX) {
        return Err(format!(
            "params: {epochs} epochs is {steps} steps, more than a run takes"
        ));
    }
    let steps = steps as u32;
    trainer_params[STEPS] = Value::from(steps);
    Ok((
        trainer_params,
        Some(EpochBudget {
            epochs,
            dataset_records: draws.dataset,
            replay_records: draws.replay,
            replay_share: draws.replay_share,
            draws_per_epoch: per_epoch,
            records_per_step: per_step,
            steps,
        }),
    ))
}

#[cfg(test)]
mod tests {

    #[test]
    fn a_trainer_checks_an_epoch_budget_as_one_step() {
        let checked = super::for_trainer_check(&serde_json::json!({"rank": 8, "epochs": 1.9}));
        assert_eq!(checked, serde_json::json!({"rank": 8, "steps": 1}));
        let steps = serde_json::json!({"rank": 8, "steps": 800});
        assert_eq!(super::for_trainer_check(&steps), steps);
    }

    use serde_json::json;

    use super::*;

    fn four(_: &Value) -> Result<u64, String> {
        Ok(4)
    }

    #[test]
    fn steps_are_the_epochs_over_the_union_of_dataset_and_replay() {
        // Knowledge 235's generation 3: 5045 records in all, 1.891 epochs, 4 a step.
        let draws = Draws {
            dataset: 1911,
            replay: 3134,
            replay_share: None,
        };
        let (params, budget) = resolve(&json!({"rank": 8, "epochs": 1.891}), draws, four).unwrap();
        // 1.891 * 5045 / 4 = 2385.02...
        assert_eq!(params, json!({"rank": 8, "steps": 2386}));
        let budget = budget.unwrap();
        assert_eq!((budget.steps, budget.records_per_step), (2386, 4));
        assert_eq!(budget.draws_per_epoch, 5045.0);
    }

    #[test]
    fn with_a_replay_share_every_dataset_record_is_seen_epochs_times() {
        // A quarter of the draws are replay: 300 dataset records are 400 draws an epoch,
        // whatever the replay holds.
        for replay in [10, 3000] {
            let draws = Draws {
                dataset: 300,
                replay,
                replay_share: Some(0.25),
            };
            let (params, _) = resolve(&json!({"epochs": 2}), draws, four).unwrap();
            assert_eq!(params["steps"], 200, "replay {replay}");
        }
    }

    #[test]
    fn a_whole_product_is_not_rounded_up_and_a_fraction_is() {
        let draws = Draws {
            dataset: 30,
            replay: 0,
            replay_share: None,
        };
        let one = |_: &Value| Ok(1);
        assert_eq!(
            resolve(&json!({"epochs": 0.1}), draws, one).unwrap().0["steps"],
            3
        );
        assert_eq!(
            resolve(&json!({"epochs": 0.11}), draws, one).unwrap().0["steps"],
            4
        );
        // Never less than one step.
        assert_eq!(
            resolve(&json!({"epochs": 0.001}), draws, one).unwrap().0["steps"],
            1
        );
    }

    #[test]
    fn the_trainer_is_asked_with_the_parameters_it_will_get() {
        let draws = Draws {
            dataset: 64,
            replay: 0,
            replay_share: None,
        };
        let (params, _) = resolve(&json!({"epochs": 1, "grad_accum": 8}), draws, |p| {
            assert!(p.get(EPOCHS).is_none(), "{p}");
            Ok(p["grad_accum"].as_u64().unwrap())
        })
        .unwrap();
        assert_eq!(params["steps"], 8);
    }

    #[test]
    fn epochs_and_steps_together_are_refused() {
        let draws = Draws {
            dataset: 10,
            replay: 0,
            replay_share: None,
        };
        let refused = resolve(&json!({"epochs": 1, "steps": 5}), draws, four).unwrap_err();
        assert!(refused.contains("epochs and steps"), "{refused}");
        for bad in [json!(0), json!(-1), json!("two")] {
            assert!(!problems(&json!({ "epochs": bad })).is_empty(), "{bad}");
        }
        assert!(resolve(
            &json!({"epochs": 1}),
            Draws {
                dataset: 0,
                ..draws
            },
            four
        )
        .is_err());
    }

    #[test]
    fn a_budget_in_steps_is_left_as_it_is() {
        let draws = Draws {
            dataset: 10,
            replay: 0,
            replay_share: None,
        };
        let (params, budget) = resolve(&json!({"steps": 5}), draws, |_| unreachable!()).unwrap();
        assert_eq!((params, budget), (json!({"steps": 5}), None));
    }
}
