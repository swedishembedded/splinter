// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning loops whose every generation can be
// explained from its own records, for its clients. If your team needs
// expertise in observing and steering agent training loops, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The budget a candidate was actually trained with: the passes over its
//! data at the step its checkpoint was selected, the steps it ran, the tokens
//! it read, its validation curve's ends and whether it stopped early.
//!
//! The effective epochs at the selected step are
//! `selected_step * records_per_step / draws_per_epoch`, the inverse of the
//! epoch budget's resolution: `draws_per_epoch` is the dataset's training
//! records plus the replayed ones (or the dataset's over `1 - replay_share`),
//! and `records_per_step` the budget's own when the stage recorded one, else
//! `grad_accum` (1 when unnamed) on each data-parallel rank the trainer
//! reported.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::budget::Draws;

use super::data::train_records;
use super::records::RunRecords;

/// The training a candidate had.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Training {
    /// The training stage.
    pub stage: String,
    /// The dataset trained.
    pub dataset: String,
    /// The datasets replayed beside it.
    pub replay: Vec<String>,
    /// The epoch budget asked for, when the stage stated one.
    pub epochs_asked: Option<f64>,
    /// The steps the trainer was given.
    pub steps: Option<u64>,
    /// The steps it ran.
    pub steps_completed: Option<u64>,
    /// The step whose checkpoint is the candidate.
    pub selected_step: Option<u64>,
    /// Whether the trainer stopped before its steps (validation plateau).
    pub stopped_early: Option<bool>,
    /// The records one optimizer step consumed.
    pub records_per_step: Option<u64>,
    /// Whether `records_per_step` is the recorded budget's or derived from
    /// the parameters.
    pub records_per_step_recorded: bool,
    /// The draws of one epoch.
    pub draws_per_epoch: Option<f64>,
    /// The passes over the data at the selected step.
    pub epochs_at_selected: Option<f64>,
    /// The passes over the data at the last step run.
    pub epochs_completed: Option<f64>,
    /// Tokens the trainer read (every row it trained on).
    pub tokens: Option<u64>,
    /// Training throughput the trainer measured.
    pub tokens_per_second: Option<f64>,
    /// Validation loss at the first evaluation.
    pub validation_loss_first: Option<f64>,
    /// Validation loss at the selected step.
    pub validation_loss_selected: Option<f64>,
    /// Validation loss at the last evaluation.
    pub validation_loss_last: Option<f64>,
    /// The base model's loss on the validation records.
    pub validation_loss_base: Option<f64>,
}

/// The training of the stage `id`.
pub(super) fn observe(records: &RunRecords, id: &str) -> Option<Training> {
    let product = records.product(id)?;
    let report: Value = product.json("report.json").unwrap_or(Value::Null);
    let stage: Value = product.json("stage.json").unwrap_or(Value::Null);
    let params = &stage["parameters"]["params"];
    let curve = &report["curve"];
    let dataset = records.upstream(id, "dataset").into_iter().next()?;
    let replay = records.upstream(id, "replay");
    let budget = &report["budget"];

    let points = curve["points"].as_array().map_or(&[][..], Vec::as_slice);
    let selected_step = curve["selected_step"].as_u64();
    let loss_at = |p: Option<&Value>| p.and_then(|p| p["monitor_loss"].as_f64());
    let selected_point =
        selected_step.and_then(|s| points.iter().find(|p| p["step"].as_u64() == Some(s)));

    let world = curve["parallel"]["reported"]["world_size"]
        .as_u64()
        .unwrap_or(1);
    let derived = params["grad_accum"].as_u64().unwrap_or(1) * world;
    let recorded = budget["records_per_step"].as_u64();
    let records_per_step = recorded.or(Some(derived));
    let draws_per_epoch = budget["draws_per_epoch"].as_f64().or_else(|| {
        let dataset = train_records(records, &dataset)? as u64;
        let replay: Option<u64> = replay
            .iter()
            .map(|r| train_records(records, r).map(|n| n as u64))
            .sum();
        let share = stage["parameters"]["replay_share"].as_f64();
        Some(
            Draws {
                dataset,
                replay: replay?,
                replay_share: share,
            }
            .per_epoch(),
        )
    });
    let epochs = |step: Option<u64>| -> Option<f64> {
        let (step, per, draws) = (step?, records_per_step?, draws_per_epoch?);
        (draws > 0.0).then(|| (step * per) as f64 / draws)
    };
    let steps_completed = curve["steps_completed"].as_u64();
    Some(Training {
        stage: id.to_owned(),
        dataset,
        replay,
        epochs_asked: budget["epochs"]
            .as_f64()
            .or_else(|| params["epochs"].as_f64()),
        steps: params["steps"].as_u64().or_else(|| curve["steps"].as_u64()),
        steps_completed,
        selected_step,
        stopped_early: curve["stopped_early"].as_bool(),
        records_per_step,
        records_per_step_recorded: recorded.is_some(),
        draws_per_epoch,
        epochs_at_selected: epochs(selected_step),
        epochs_completed: epochs(steps_completed),
        tokens: curve["speed"]["row_tokens"].as_u64(),
        tokens_per_second: curve["speed"]["tokens_per_second"].as_f64(),
        validation_loss_first: loss_at(points.first()),
        validation_loss_selected: loss_at(selected_point),
        validation_loss_last: loss_at(points.last()),
        validation_loss_base: report["base_score"]["loss"].as_f64(),
    })
}
