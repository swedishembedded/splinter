// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements experiment runs whose every model is pinned
// by what it is, not where it is served, for its clients. If your team needs
// expertise in reproducible machine-learning experiments, you can procure our
// services by sending an email to info@swedishembedded.com.

//! A stage's model roles as a run reaches them: the binding an identity sees
//! (a champion by its content), and the binding a call is made to (a
//! champion served through the trainer).

use splinter_core::factory::execution::ModelBinding;

use super::{stage_error, Factory, RunError};
use crate::champion::{Champion, CHAMPION};
use crate::recipe::ModelRef;

/// The binding a stage's identity sees for `model`, and the champion it is:
/// a connection's model as named; a champion as its base's connection and
/// its digest, since the id it is served under is the server's to choose.
pub(crate) fn binding_of(
    factory: &Factory<'_>,
    key: &str,
    model: &ModelRef,
) -> Result<(ModelBinding, Option<Champion>), RunError> {
    let Some(champion) = factory.resolved.champions.get(&model.to_string()) else {
        let binding = factory
            .resolved
            .connections
            .bind(model)
            .map_err(|e| stage_error(key, e))?;
        return Ok((binding, None));
    };
    let mut binding = factory
        .resolved
        .connections
        .bind(&champion.record.base)
        .map_err(|e| stage_error(key, e))?;
    // By content, not by where its run is: the same champion is the same input.
    binding.model = format!("{CHAMPION}@{}", champion.digest.hex());
    Ok((binding, Some(champion.clone())))
}

/// `binding` as a model is reached: itself, or the champion it is, served
/// from its base and adapter through the trainer.
pub(crate) fn serve(
    factory: &Factory<'_>,
    key: &str,
    binding: &ModelBinding,
    champion: Option<&Champion>,
) -> Result<ModelBinding, RunError> {
    let Some(champion) = champion else {
        return Ok(binding.clone());
    };
    let trainer = factory
        .services
        .trainer
        .ok_or_else(|| stage_error(key, "a champion is served through the trainer"))?;
    let id = trainer
        .serve(&champion.record.base, champion.adapter.as_deref())
        .map_err(|e| stage_error(key, e))?;
    Ok(ModelBinding {
        model: id,
        ..binding.clone()
    })
}
