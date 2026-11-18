// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements curated fine-tuning datasets that are
// checked by the trainer before a GPU is claimed, for its clients. If your
// team needs expertise in training data for agent models, you can procure
// our services by sending an email to info@swedishembedded.com.

//! A stored dataset read by the trainer that will train it, before any
//! device is claimed: brain's `check_dataset` over every record, rendered
//! for the base and in the mode a supervised run would use (reasoning when
//! the records carry it, one example per record when the dataset was built
//! packed), every record listed.

use serde::Serialize;
use splinter_core::dataset::DatasetId;
use splinter_core::model_ref::ModelRef;
use splinter_model::served_train::{
    check_dataset, CheckSpec, Distribution, RecordLength, Refusal, MAX_BLOCK,
};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

use super::resolve_dataset;

/// How many of the longest records a check names.
const LONGEST_NAMED: usize = 5;

/// What the trainer found in a stored dataset.
#[derive(Clone, Debug, Serialize)]
pub struct DatasetCheck {
    /// The dataset.
    pub dataset: DatasetId,
    /// The base whose tokenizer and chat template read it.
    pub base: String,
    /// Whether it was read as a run that trains reasoning reads it.
    pub thinking: bool,
    /// Whether a record supervising several turns is one example.
    pub pack_answers: bool,
    /// The row it was checked against, in tokens.
    pub max_block: u32,
    /// Records in the file.
    pub records: usize,
    /// Records the trainer reads.
    pub valid_records: usize,
    /// What they train as.
    pub examples: usize,
    /// Per example, in the base's tokens.
    pub tokens: Distribution,
    /// Per example, the supervised tokens.
    pub supervised_tokens: Distribution,
    /// Examples that would train on nothing.
    pub examples_without_supervision: usize,
    /// Examples longer than `max_block`.
    pub over_max_block: usize,
    /// Every record the trainer refuses, with why.
    pub refused: Vec<Refusal>,
    /// The longest records, longest first.
    pub longest: Vec<RecordLength>,
}

/// Checks the stored dataset `id` (or a unique prefix of it) against the
/// base `from` trains (the policy's when `None`), against `max_block` when
/// given.
///
/// # Errors
/// The dataset or the base cannot be resolved, no brain service can be
/// asked, or the service refused the check.
pub fn check(
    ctx: &Context,
    id: &str,
    from: Option<&ModelRef>,
    max_block: Option<u32>,
) -> Result<DatasetCheck, OrchestratorError> {
    let stored = resolve_dataset(ctx, id)?;
    let base = crate::train::trained_base(ctx, from.unwrap_or(&ModelRef::policy_default()))?;
    let base = splinter_model::selection::local_model_name(&base);
    let thinking = crate::train::carries_reasoning(&stored.path)?;
    let pack_answers = stored.manifest.packed;
    let mut spec = CheckSpec::chat(&base, thinking, pack_answers);
    spec.max_block = max_block.unwrap_or(MAX_BLOCK);
    let report = check_dataset(&ctx.brain()?, &stored.path, &spec).map_err(|e| {
        OrchestratorError::Model {
            model: base.clone(),
            detail: format!("checking dataset {}: {e}", stored.id),
        }
    })?;
    let max_block = spec.max_block;
    let mut longest = report.per_record.clone();
    longest.sort_by(|a, b| b.tokens.cmp(&a.tokens).then(a.line.cmp(&b.line)));
    longest.truncate(LONGEST_NAMED);
    Ok(DatasetCheck {
        dataset: stored.id,
        base,
        thinking,
        pack_answers,
        max_block,
        records: report.records,
        valid_records: report.valid_records,
        examples: report.examples,
        tokens: report.tokens,
        supervised_tokens: report.supervised_tokens,
        examples_without_supervision: report.examples_without_supervision,
        over_max_block: report.over_max_block,
        refused: report.refused,
        longest,
    })
}
