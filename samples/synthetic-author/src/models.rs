// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in proving that a trained model has
// learned what a corpus implies, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The models a command line run writes and reads with: sampled at the
//! writer's temperature when they run here, at their server's own when they
//! do not, and, when they run here, seeded for every request from the run's
//! seed.

use std::str::FromStr;

use anyhow::Context as _;
use splinter_sdk::agent::blocking::BlockingModel;
use splinter_sdk::model::local::{Sampling, GREEDY_SAMPLING};
use splinter_sdk::vocabulary::model_ref::ModelRef;
use splinter_sdk::Context;

use crate::writer::{Lent, Models, RequestKey};

/// The temperature the writer draws at, so that no two articles are alike.
pub const WRITER_TEMPERATURE: f32 = 0.8;

/// How the models of a corpus decode, for its manifest.
pub const WRITER_DECODING: &str =
    "sampled at temperature 0.8 with reasoning off when run here, the provider's own when remote";

fn reference(name: &str) -> anyhow::Result<ModelRef> {
    ModelRef::from_str(name).with_context(|| format!("model {name:?}"))
}

/// Whether the model `name` names is reached over a network, where its
/// sampling and seed are its provider's.
///
/// # Errors
/// `name` is not a model reference.
pub fn is_remote(name: &str) -> anyhow::Result<bool> {
    Ok(reference(name)?.is_remote())
}

fn sampling(seed: Option<u64>) -> Sampling {
    Sampling {
        temperature: WRITER_TEMPERATURE,
        thinking: false,
        seed,
        ..GREEDY_SAMPLING
    }
}

/// The model a run asks: one model for every request when it is reached over a
/// network, and for a model that runs here a model made for each request with
/// that request's own seed.
pub enum Source<'a> {
    /// The same model answers every request.
    Shared(BlockingModel),
    /// A model is made for every request, seeded from the run seed.
    Seeded {
        /// Where models are had.
        ctx: &'a Context,
        /// The model.
        reference: ModelRef,
        /// The seed every request's own seed is derived from.
        run_seed: u64,
    },
}

impl<'a> Source<'a> {
    /// The source of the model `name` names, seeded from `run_seed` when the
    /// model runs here.
    ///
    /// # Errors
    /// `name` is not a model reference, the model cannot be had, or it runs
    /// here and there is no run seed.
    pub fn open(ctx: &'a Context, name: &str, run_seed: Option<u64>) -> anyhow::Result<Self> {
        let reference = reference(name)?;
        if reference.is_remote() {
            return Ok(Self::Shared(BlockingModel::new(ctx.model(&reference)?)?));
        }
        let run_seed = run_seed.with_context(|| format!("{name}: a run seed is needed"))?;
        Ok(Self::Seeded {
            ctx,
            reference,
            run_seed,
        })
    }
}

impl Models for Source<'_> {
    fn model(&self, request: RequestKey) -> anyhow::Result<Lent<'_>> {
        match self {
            Self::Shared(model) => Ok(Lent::Borrowed(model)),
            Self::Seeded {
                ctx,
                reference,
                run_seed,
            } => {
                let model = ctx
                    .resampled(reference, sampling(Some(request.seed(*run_seed))))?
                    .with_context(|| format!("{reference} cannot be sampled at a temperature"))?;
                Ok(Lent::Owned(Box::new(BlockingModel::new(model)?)))
            }
        }
    }
}
