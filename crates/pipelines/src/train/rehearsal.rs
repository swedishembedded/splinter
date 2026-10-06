// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning that rehearses a model's own
// answers so new training does not cost it what it could already do, for
// its clients. If your team needs expertise in continual learning without
// catastrophic forgetting, you can procure our services by sending an email
// to info@swedishembedded.com.

//! The rehearsal of a training run: a chat dataset of the base's own
//! answers, resolved and split into the records mixed into training at
//! their share and the records that join the monitoring set.

use std::path::{Path, PathBuf};

use serde::Serialize;
use splinter_core::training::Regime;
use splinter_data::holdout::monitor_split_named;
use splinter_data::StoredDataset;

use super::{concatenate, trainable};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::{io, OrchestratorError};

/// The rehearsed records mixed into training, inside a candidate's
/// directory.
pub const REHEARSAL_FILE: &str = "rehearsal.jsonl";
/// The rehearsed records set aside for the monitoring set.
const REHEARSAL_MONITOR_FILE: &str = "rehearsal_monitor.jsonl";

/// A chat dataset rehearsed beside the new records: mixed into the
/// training draws at `share`, never held out, and monitored on.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Rehearse {
    /// The dataset, by id or unique prefix.
    pub dataset: String,
    /// The share of the training draws its records take, in (0, 1).
    pub share: f64,
}

/// The rehearsal of a plan, split for training and monitoring.
#[derive(Clone, Debug)]
pub struct RehearsalPlan {
    /// The dataset rehearsed.
    pub dataset: StoredDataset,
    /// The records mixed into training.
    pub fit: PathBuf,
    /// The records that join the monitoring set; `None` when the run
    /// monitors nothing.
    pub monitor: Option<PathBuf>,
    /// The share of the training draws `fit` takes.
    pub share: f32,
    /// Records in `fit`.
    pub trained: usize,
    /// Records in `monitor`.
    pub monitored: usize,
}

/// How a rehearsal is split for a run: where its files go, whether the run
/// monitors and at what share, and the share the replay already takes.
pub(super) struct RehearsalSplit<'a> {
    pub(super) dir: &'a Path,
    pub(super) monitored: bool,
    pub(super) monitor_share: f64,
    pub(super) replay_share: Option<f32>,
}

/// The rehearsal `rehearse` asks for, resolved and split: refused for a
/// share outside (0, 1), for shares that with the replay's leave the new
/// records no draws, and for a dataset brain does not train by supervised
/// fine-tuning. A monitored run sets `split.monitor_share` of the records
/// aside for the monitoring set, whole records, chosen as the training
/// families' monitoring records are.
pub(super) fn prepare_rehearsal(
    ctx: &Context,
    rehearse: &Rehearse,
    split: &RehearsalSplit<'_>,
) -> Result<RehearsalPlan, OrchestratorError> {
    if !(rehearse.share > 0.0 && rehearse.share < 1.0) {
        return Err(OrchestratorError::Refused(format!(
            "the rehearsal share {} is not in (0, 1): the rehearsed records take some of the \
             training draws and the new records the rest",
            rehearse.share
        )));
    }
    let taken = rehearse.share + f64::from(split.replay_share.unwrap_or(0.0));
    if taken >= 1.0 {
        return Err(OrchestratorError::Refused(format!(
            "the rehearsal share {} and the replay share {} leave the new records no draws",
            rehearse.share,
            split.replay_share.unwrap_or(0.0)
        )));
    }
    let (dataset, regime) = trainable(ctx, &rehearse.dataset)?;
    if regime != Regime::Sft {
        return Err(OrchestratorError::Refused(format!(
            "dataset {} holds {:?} records; only chat records are rehearsed",
            dataset.id, dataset.manifest.objective
        )));
    }
    let count = |path: &Path| -> Result<usize, OrchestratorError> {
        Ok(std::fs::read_to_string(path)
            .map_err(io(path))?
            .lines()
            .filter(|l| !l.trim().is_empty())
            .count())
    };
    let (fit, monitor) = if split.monitored {
        let halves = monitor_split_named(
            &dataset.path,
            split.dir,
            split.monitor_share,
            (REHEARSAL_FILE, REHEARSAL_MONITOR_FILE),
        )?;
        (halves.fit, Some(halves.monitor))
    } else {
        (
            concatenate(&[&dataset.path], &split.dir.join(REHEARSAL_FILE))?,
            None,
        )
    };
    Ok(RehearsalPlan {
        trained: count(&fit)?,
        monitored: monitor.as_deref().map_or(Ok(0), count)?,
        dataset,
        fit,
        monitor,
        share: rehearse.share as f32,
    })
}
