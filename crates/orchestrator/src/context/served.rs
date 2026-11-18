// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that reach their model
// only through a serving API, for its clients. If your team needs expertise
// in routing agent traffic across inference servers, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Where a local model is served, for a process outside this one to ask it.

use std::path::PathBuf;

use splinter_core::model_ref::ModelRef;

use super::Context;
use crate::error::OrchestratorError;

/// A local model as a brain endpoint serves it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServedEndpoint {
    /// The endpoint's OpenAI-compatible base URL (ending in `/v1`).
    pub base_url: String,
    /// The key requests to it are made with; empty when it takes none.
    pub key: String,
    /// The id the endpoint lists the weights under.
    pub model: String,
    /// The identity records give the model.
    pub identity: String,
    /// The adapter file it serves on the base, when it serves one.
    pub adapter: Option<PathBuf>,
}

impl Context {
    /// Where the brain server serves the local model `reference` names, for
    /// a process outside this one to ask it (an examiner's gateway): the
    /// first endpoint's URL and key, the id it lists the weights under, the
    /// identity records give the model and the adapter file it serves.
    ///
    /// # Errors
    /// `reference` is not a local model, no endpoint serves it, or the
    /// brain server cannot be reached or started.
    pub fn served_endpoint(
        &self,
        reference: &ModelRef,
    ) -> Result<ServedEndpoint, OrchestratorError> {
        let (_, selection) = self.resolve(reference)?;
        let weights = selection.local().ok_or_else(|| {
            OrchestratorError::Refused(format!(
                "{reference} is not a local model: only a model brain serves is reached \
                 through an endpoint"
            ))
        })?;
        let brain = self.brain()?;
        let model = brain
            .locate(weights)
            .map_err(|e| OrchestratorError::Model {
                model: reference.to_string(),
                detail: format!("{e:#}"),
            })?
            .into_iter()
            .next()
            .ok_or_else(|| OrchestratorError::Refused("no brain endpoint is configured".into()))?;
        Ok(ServedEndpoint {
            base_url: brain.urls()[0].clone(),
            key: brain.key().to_string(),
            model,
            identity: selection.identity(),
            adapter: weights.adapter.clone(),
        })
    }
}
