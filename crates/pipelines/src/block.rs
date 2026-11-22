// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-set curation that sizes every
// record exactly as the trainer will, for its clients. If your team needs
// expertise in fitting conversations to a fine-tuning block, you can procure
// our services by sending an email to info@swedishembedded.com.

//! A dataset's records read by the trainer they are built for (`dataset build
//! --from`), and bounded to its training block (`--max-block`).

use std::path::Path;

use splinter_data::{BlockMeasure, Exclusion, Projection};
use splinter_model::measure::RecordMeter;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// `projection` read by the trainer of `base`, the model a run on it starts
/// from: the records that trainer refuses left out, and with `max_block` the
/// records longer than it too (with `split`, a longer chat record is split
/// into windows that fit, [`Projection::split_over_block`], instead).
///
/// A record is read as the trainer will read it: written as the dataset line
/// it will be and checked for `base` by the brain service's own trainer
/// ([`RecordMeter`], one check of every record: its chat template, its record
/// contract and its count; pictures count the rows its vision configuration
/// gives them). A record the trainer refuses is left out as
/// [`Exclusion::TrainerRefused`], so a run on the dataset never meets it. The
/// trainer's check is the only one: nothing here renders a template.
///
/// When the trainer cannot read the records - the objective's records are not
/// chat lines, or no brain service can be asked - a bound is applied by
/// characters at the fallback ratio plus image rows, a floor that leaves out
/// the chat template's markup, and nothing is checked; which measure was used,
/// and why, is recorded on the projection ([`BlockMeasure`]) for the build
/// report. A build that names its base (`required`) is refused instead when no
/// service can be asked: it claims a dataset that base's trainer reads.
///
/// # Errors
/// `required`, and no brain service can be asked.
pub(crate) fn trainer_checked(
    ctx: &Context,
    base: &Path,
    projection: Projection,
    max_block: Option<usize>,
    split: bool,
    required: bool,
) -> Result<Projection, OrchestratorError> {
    let brain = match ctx.brain() {
        Ok(brain) => brain,
        Err(e) if required => {
            return Err(OrchestratorError::Refused(format!(
                "a build for the base {} has every record read by that base's trainer, and no \
                 brain service can be asked to: {e}; give --brain-endpoint (or a brain binary), \
                 or build without --from",
                base.display()
            )))
        }
        Err(e) => return Ok(floor(projection, max_block, split, e.to_string())),
    };
    let meter = match trainer_measure(&brain, base, &projection) {
        Ok(meter) => meter,
        Err(reason) => return Ok(floor(projection, max_block, split, reason)),
    };
    let objective = projection.objective;
    let line = |record: &splinter_data::Record| {
        splinter_data::dataset_line(objective, record).map_err(|e| e.to_string())
    };
    let lines: Vec<String> = projection
        .records
        .iter()
        .filter_map(|record| line(record).ok())
        .collect();
    if let Err(e) = meter.measure(&lines.iter().map(String::as_str).collect::<Vec<_>>()) {
        return Ok(floor(projection, max_block, split, e.to_string()));
    }
    let tokens = |record: &splinter_data::Record| line(record).and_then(|l| meter.tokens(&l));
    let mut projection = projection;
    let mut first_refusal = None;
    let before = projection.records.len();
    projection.records.retain(|record| match tokens(record) {
        Ok(_) => true,
        Err(reason) => {
            first_refusal.get_or_insert(reason);
            false
        }
    });
    let refused = before - projection.records.len();
    let mut projection = projection.with_excluded(Exclusion::TrainerRefused, refused);
    if let Some(max_block) = max_block {
        let size = |record: &splinter_data::Record| tokens(record).unwrap_or(usize::MAX);
        projection = bounded(projection, max_block, split, &size);
    }
    projection.block_measure = Some(BlockMeasure::Trainer {
        base: base.to_path_buf(),
        first_refusal,
    });
    Ok(projection)
}

/// `projection` bounded to `max_block` by characters at the fallback ratio
/// plus image rows (a floor on the trainer's count), the reason recorded;
/// unchanged without a bound.
fn floor(
    projection: Projection,
    max_block: Option<usize>,
    split: bool,
    reason: String,
) -> Projection {
    let Some(max_block) = max_block else {
        return projection;
    };
    // A picture is the rows it adds to the sequence, not its base64.
    let size = |record: &splinter_data::Record| {
        let (text, image_rows) = record.body.text_and_image_rows();
        splinter_data::chars_as_tokens(&text) + image_rows
    };
    let mut projection = bounded(projection, max_block, split, &size);
    projection.block_measure = Some(BlockMeasure::Floor { reason });
    projection
}

fn bounded(
    projection: Projection,
    max_block: usize,
    split: bool,
    size: &dyn Fn(&splinter_data::Record) -> usize,
) -> Projection {
    if split {
        projection.split_over_block(max_block, size)
    } else {
        projection.within_block(max_block, size)
    }
}

/// The trainer's count of a record of `projection` for the base `base`, in the
/// mode its fine-tune would run in; the reason it cannot be had otherwise.
fn trainer_measure<'b>(
    brain: &'b splinter_model::BrainEndpoints,
    base: &Path,
    projection: &Projection,
) -> Result<RecordMeter<'b>, String> {
    use splinter_data::Format;
    let format = projection.objective.line_format();
    if format != Format::GenericMessagesV2 {
        return Err(format!(
            "the {format:?} records of this objective are not read by the trainer's chat reader"
        ));
    }
    // The mode `train` runs the dataset in: reasoning records train for
    // thinking mode, and a packed dataset keeps each record as one sequence.
    let thinking = projection.records.iter().any(|record| {
        splinter_data::dataset_line(projection.objective, record)
            .is_ok_and(|line| splinter_model::measure::carries_reasoning(&line))
    });
    Ok(RecordMeter::for_model(
        brain,
        base,
        thinking,
        projection.packed,
    ))
}
