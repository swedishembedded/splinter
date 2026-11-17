// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! What a fine-tune's datasets must satisfy before a run is queued: every
//! record parses and renders under the base's chat template, monitoring has
//! records, and none is longer than the row the run trains. The service's own
//! trainer reads each dataset ([`check_dataset`]), so a refused record is
//! named by its line and the row is sized to the longest example it found.

use std::path::{Path, PathBuf};

use crate::endpoint::BrainEndpoints;
use crate::error::PolicyError;
use crate::selection::local_model_name;
use crate::served_train::{check_dataset, CheckSpec};
use crate::train::FineTune;

/// brain refuses a training row shorter than this.
pub(crate) const MIN_BLOCK: u32 = 64;

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

/// What the checks found: the valid records of the datasets counted, and the
/// row the run needs.
pub(crate) struct Checked {
    pub records: usize,
    pub block: u32,
}

/// Checks `datasets` against the base `model_dir` as a run in the mode
/// `(thinking, pack_answers)` reads them: each is read by the trainer, and
/// none holds a record longer than `max_block` when the caller bounded the
/// row. Counts the valid records of the first `counted` datasets; the row is
/// the bound, else the longest example.
pub(crate) fn check_chat_datasets(
    brain: &BrainEndpoints,
    model_dir: &Path,
    datasets: &[&Path],
    counted: usize,
    (thinking, pack_answers): (bool, bool),
    max_block: Option<u32>,
) -> Result<Checked, PolicyError> {
    let base = local_model_name(model_dir);
    let mut spec = CheckSpec::chat(&base, thinking, pack_answers);
    spec.per_record = false;
    if let Some(bound) = max_block {
        spec.max_block = bound;
    }
    let (mut records, mut longest, mut overlong) = (0usize, 0u32, Vec::new());
    for (index, dataset) in datasets.iter().enumerate() {
        let report = check_dataset(brain, dataset, &spec)?;
        report.refuse_invalid(dataset)?;
        if index < counted {
            records += report.valid_records;
        }
        longest = longest.max(report.longest());
        overlong.extend(
            report
                .over_max_block_lines
                .iter()
                .map(|line| format!("{}:{line}", dataset.display())),
        );
        let unlisted = report
            .over_max_block
            .saturating_sub(report.over_max_block_lines.len());
        if unlisted > 0 {
            overlong.push(format!("{unlisted} more in {}", dataset.display()));
        }
    }
    if let (Some(bound), false) = (max_block, overlong.is_empty()) {
        return Err(PolicyError::Dataset {
            path: datasets
                .first()
                .map_or_else(PathBuf::new, |p| p.to_path_buf()),
            reason: format!(
                "record(s) longer than max_block {bound}; shorten or drop them: {}",
                overlong.join("; ")
            ),
        });
    }
    Ok(Checked {
        records,
        block: max_block.unwrap_or(longest).max(MIN_BLOCK),
    })
}

/// Checks every dataset of `request` before a card is claimed: each is read
/// by the trainer, monitoring has records to monitor, and no record is longer
/// than `max_block`. Counts the training and held-out records.
pub(crate) fn check_request(
    brain: &BrainEndpoints,
    request: &FineTune<'_>,
) -> Result<Checked, PolicyError> {
    if request.monitor.is_none() && request.eval_every > 0 {
        return Err(PolicyError::Train {
            dir: request.attempt_dir.to_path_buf(),
            reason: "the run was asked to evaluate as it trains but given no monitoring records"
                .into(),
        });
    }
    check_chat_datasets(
        brain,
        request.model_dir,
        &datasets_of(request),
        2,
        (request.thinking, request.pack_answers),
        request.max_block,
    )
}
