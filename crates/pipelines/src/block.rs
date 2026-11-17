// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-set curation that sizes every
// record exactly as the trainer will, for its clients. If your team needs
// expertise in fitting conversations to a fine-tuning block, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Bounding a dataset's records to the training block (`dataset build
//! --max-block`).

use std::path::Path;

use splinter_data::{BlockMeasure, Projection};
use splinter_model::measure::RecordMeter;
use splinter_orchestrator::context::Context;

/// `projection` without the records longer than `max_block` tokens; with
/// `split`, a longer chat record is split into windows that fit
/// ([`Projection::split_over_block`]) instead of left out.
///
/// A record is measured as the trainer will measure it: written as the
/// dataset line it will be and read for the policy base `base` by the brain
/// service's own trainer ([`RecordMeter`], one check of every record, the
/// count a run's length check refuses by; pictures count the rows its vision
/// configuration gives them), so every record kept passes that check at the
/// same bound. Only where the trainer's count is unavailable - the objective's
/// records are not chat lines, or no brain service can be asked - is a record
/// measured by characters at the fallback ratio plus image rows, a floor that
/// leaves out the chat template's markup. Which measure was used, and why, is
/// recorded on the projection ([`BlockMeasure`]) for the build report.
pub(crate) fn within_policy_block(
    ctx: &Context,
    base: &Path,
    projection: Projection,
    max_block: usize,
    split: bool,
) -> Projection {
    let brain = ctx.brain().map_err(|e| e.to_string());
    let measured = brain.and_then(|brain| {
        let meter = trainer_measure(&brain, base, &projection)?;
        let objective = projection.objective;
        let lines: Vec<String> = projection
            .records
            .iter()
            .filter_map(|record| splinter_data::dataset_line(objective, record).ok())
            .collect();
        meter
            .measure(&lines.iter().map(String::as_str).collect::<Vec<_>>())
            .map_err(|e| e.to_string())?;
        let refusal = std::cell::RefCell::new(None::<String>);
        let size = |record: &splinter_data::Record| {
            splinter_data::dataset_line(objective, record)
                .map_err(|e| e.to_string())
                .and_then(|line| meter.tokens(&line))
                // A record the trainer's reader refuses is refused by the
                // trainer too: it counts as longer than any bound.
                .unwrap_or_else(|reason| {
                    refusal.borrow_mut().get_or_insert(reason);
                    usize::MAX
                })
        };
        let bounded = bounded(projection.clone(), max_block, split, &size);
        Ok((bounded, refusal.into_inner()))
    });
    let (mut projection, measure) = match measured {
        Ok((bounded, first_refusal)) => (
            bounded,
            BlockMeasure::Trainer {
                base: base.to_path_buf(),
                first_refusal,
            },
        ),
        Err(reason) => {
            // A picture is the rows it adds to the sequence, not its base64.
            let size = |record: &splinter_data::Record| {
                let (text, image_rows) = record.body.text_and_image_rows();
                splinter_data::chars_as_tokens(&text) + image_rows
            };
            (
                bounded(projection, max_block, split, &size),
                BlockMeasure::Floor { reason },
            )
        }
    };
    projection.block_measure = Some(measure);
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
