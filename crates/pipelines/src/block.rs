// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-set curation that sizes every
// record exactly as the trainer will, for its clients. If your team needs
// expertise in fitting conversations to a fine-tuning block, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Bounding a dataset's records to the training block (`dataset build
//! --max-block`).

use splinter_data::{BlockMeasure, Projection};
use splinter_model::measure::RecordMeter;
use splinter_model::TokenCounter;
use splinter_orchestrator::context::Context;

/// `projection` without the records longer than `max_block` tokens; with
/// `split`, a longer chat record is split into windows that fit
/// ([`Projection::split_over_block`]) instead of left out.
///
/// A record is measured as the trainer will measure it: written as the
/// dataset line it will be, read and encoded for the policy base by brain's
/// chat reader ([`RecordMeter`], the count `train`'s length check refuses by),
/// so every record kept passes that check at the same bound. Only where the
/// trainer's count is unavailable - the policy base has no readable chat
/// tokenizer, or the objective's records are not chat lines - is a record
/// measured by its JSON text through a tokenizer (else a character ratio) plus
/// image rows, a floor that leaves out the chat template's markup. Which
/// measure was used, and why, is recorded on the projection
/// ([`BlockMeasure`]) for the build report.
pub(crate) fn within_policy_block(
    ctx: &Context,
    projection: Projection,
    max_block: usize,
    split: bool,
) -> Projection {
    let (mut projection, measure) = match trainer_measure(ctx, &projection) {
        Ok(meter) => {
            let refusal = std::cell::RefCell::new(None::<String>);
            let objective = projection.objective;
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
            let bounded = bounded(projection, max_block, split, &size);
            let first_refusal = refusal.into_inner();
            (bounded, BlockMeasure::Trainer { first_refusal })
        }
        Err(reason) => {
            let counter = TokenCounter::for_model(&ctx.config().policy_base).ok();
            let tokens = |text: &str| {
                counter
                    .as_ref()
                    .map_or_else(|| splinter_data::chars_as_tokens(text), |c| c.count(text))
            };
            // A picture is the rows it adds to the sequence, not its base64.
            let size = |record: &splinter_data::Record| {
                let (text, image_rows) = record.body.text_and_image_rows();
                tokens(&text) + image_rows
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

/// The trainer's count of a record of `projection` for the policy base, in the
/// mode its fine-tune would run in; the reason it cannot be had otherwise.
fn trainer_measure(ctx: &Context, projection: &Projection) -> Result<RecordMeter, String> {
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
    RecordMeter::for_model(&ctx.config().policy_base, thinking, projection.packed)
        .map_err(|e| e.to_string())
}
