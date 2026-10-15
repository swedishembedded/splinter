// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible, auditable fine-tuning
// pipelines for its clients. If your team needs expertise in seeding and
// experiment provenance for model training, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The seed of a training run: initialisation of a fresh adapter, order of
//! the batches and draw of the records replayed. A seed the caller gave is
//! used as given; one left out is drawn, so no two unseeded runs share a
//! draw, and it is reported so that passing it back (`--seed`) reproduces the
//! run. A run recorded as a command's work draws through its recorder, which
//! writes the seed into the run record as well ([`train_recorded`]).

use splinter_core::rng::Seed;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::runs::Recorder;

use super::{train, Candidate, TrainRequest, Trainer, Tuning};

/// The name a training seed is recorded and reported under: the flag that
/// sets it.
pub const SEED: &str = "seed";

/// The seed of a run: `given`, else a fresh one reported to the context's
/// progress receiver as the stage `seed`.
pub(super) fn resolve(ctx: &Context, given: Option<u64>) -> u64 {
    given.unwrap_or_else(|| {
        let seed = splinter_model::seed::draw();
        ctx.report_stage(
            "seed",
            &serde_json::json!({ "name": SEED, "seed": seed, "drawn": true }),
        );
        seed.value()
    })
}

/// `tuning` with its seed settled as the recorded draw `name` of `run`: a
/// seed it leaves out is drawn through the recorder, so the run record states
/// it, and is then the seed the training is handed.
pub fn record_seed(
    run: &mut Recorder<'_>,
    name: &str,
    tuning: Tuning,
) -> Result<Tuning, OrchestratorError> {
    let seed = run.seed(name, tuning.seed.map(Seed::from))?;
    Ok(Tuning {
        seed: Some(seed.value()),
        ..tuning
    })
}

/// [`train`] as the work of the recorded run `run`: the training seed is
/// recorded under the name `seed`.
pub fn train_recorded(
    ctx: &Context,
    run: &mut Recorder<'_>,
    request: &TrainRequest,
    trainer: &dyn Trainer,
) -> Result<Candidate, OrchestratorError> {
    train_recorded_as(ctx, run, SEED, request, trainer)
}

/// [`train_recorded`] with the seed recorded under `name`, for a run that
/// trains more than once and so holds one seed per training.
pub fn train_recorded_as(
    ctx: &Context,
    run: &mut Recorder<'_>,
    name: &str,
    request: &TrainRequest,
    trainer: &dyn Trainer,
) -> Result<Candidate, OrchestratorError> {
    let seeded = TrainRequest {
        tuning: record_seed(run, name, request.tuning)?,
        ..request.clone()
    };
    train(ctx, &seeded, trainer, &run.cancel_token())
}
