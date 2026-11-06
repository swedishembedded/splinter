// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! What a fine-tune's datasets must satisfy before a device is claimed or a
//! byte is sent to a training service: every record parses and renders under
//! the base's chat template, monitoring has records, and none is longer than
//! the row the run trains. Shared by the in-process trainer
//! ([`crate::train::fine_tune`]) and the served one
//! ([`crate::served_train::fine_tune_served`]).

use std::path::{Path, PathBuf};

use crate::capabilities::overlong_records;
use crate::error::PolicyError;
use crate::pictures::validate_dataset_on;
use crate::train::{validate_dataset, FineTune};

/// Every chat dataset the run trains or scores: the training and held-out
/// records, the writer's own text, the monitoring records and every replay.
pub(crate) fn datasets_of<'a>(request: &FineTune<'a>) -> Vec<&'a Path> {
    [
        Some(request.train),
        Some(request.held_out),
        request.held_out_text,
        request.monitor,
    ]
    .into_iter()
    .flatten()
    .chain(request.replay.iter().map(PathBuf::as_path))
    .chain(request.weighted_replay.iter().map(|(p, _)| p.as_path()))
    .collect()
}

/// Names every record of every chat dataset the run trains or scores that
/// is longer than `max_block`, as the failure reason; `Ok` when none is.
fn refuse_overlong(request: &FineTune<'_>, max_block: u32) -> Result<(), String> {
    let datasets = datasets_of(request);
    let mut named = Vec::new();
    for dataset in datasets {
        let found = overlong_records(
            dataset,
            request.model_dir,
            request.thinking,
            request.pack_answers,
            max_block,
        )
        .map_err(|e| e.to_string())?;
        named.extend(found.iter().map(ToString::to_string));
    }
    if named.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} record(s) longer than max_block {max_block}; shorten or drop them: {}",
            named.len(),
            named.join("; ")
        ))
    }
}

/// Checks every dataset of `request` before a device is claimed or a byte
/// is sent anywhere: each parses under the trainer's own parser and renders
/// under the base's chat template, monitoring has records to monitor, and no
/// record is longer than `max_block`. Returns the records of the training
/// and held-out sets together.
pub(crate) fn check_request(request: &FineTune<'_>) -> Result<usize, PolicyError> {
    let failed = |reason: String| PolicyError::Train {
        dir: request.attempt_dir.to_path_buf(),
        reason,
    };
    let trained_on = validate_dataset_on(request.train, request.model_dir)?;
    let held_out_summary = validate_dataset_on(request.held_out, request.model_dir)?;
    for replayed in request.replay {
        validate_dataset_on(replayed, request.model_dir)?;
    }
    for (weighted, _) in request.weighted_replay {
        validate_dataset_on(weighted, request.model_dir)?;
    }
    if let Some(monitor) = request.monitor {
        validate_dataset_on(monitor, request.model_dir)?;
    } else if request.eval_every > 0 {
        return Err(failed(
            "the run was asked to evaluate as it trains but given no monitoring records".into(),
        ));
    }
    if let Some(text) = request.held_out_text {
        validate_dataset(text)?;
    }
    if let Some(max_block) = request.max_block {
        refuse_overlong(request, max_block).map_err(failed)?;
    }
    Ok(trained_on.records + held_out_summary.records)
}
