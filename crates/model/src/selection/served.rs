// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that reach their model
// only through a serving API, so that many agent attempts share one
// continuously batched inference service, for its clients. If your team
// needs expertise in routing agent traffic across inference servers, you
// can procure our services by sending an email to info@swedishembedded.com.

//! A local model reference requested from brain endpoints
//! ([`ModelSelection::load_served`]): the provider of each endpoint, built for
//! the sampling and the pin of the attempt that asks.

use std::sync::Arc;

use sven_sdk::model::ModelProvider;

use super::{
    probed_provider, served_config_capped, LoadedModel, ModelSelection, ServedSampling,
    LOCAL_SERVER_PROVIDER,
};
use crate::endpoint::BrainEndpoints;
use crate::error::PolicyError;
use crate::local::{Sampling, AGENT_SAMPLING, REASONING_REPLY_TOKENS};

/// A local reference requested from brain endpoints: where, and under which
/// id each endpoint serves it.
pub(super) struct Served {
    pub(super) endpoints: BrainEndpoints,
    /// The id each endpoint serves the model under, in the order of its URLs.
    pub(super) ids: Vec<String>,
}

impl Served {
    /// A provider for the endpoint the conversation `pin` names is pinned
    /// to (the first id's, without one), sampling as `sampling` says and
    /// capped as an in-process reply is.
    pub(super) fn provider(
        &self,
        sampling: &Sampling,
        pin: Option<Vec<u8>>,
    ) -> Result<Arc<dyn ModelProvider>, PolicyError> {
        let index = self
            .endpoints
            .pin(&pin.unwrap_or_else(|| self.ids[0].as_bytes().to_vec()));
        // Asked through sven's brain driver, not a generic OpenAI one: a
        // traced request then gets brain's policy trace (the policy's own
        // log-probabilities over the whole vocabulary) rather than the
        // sampler's truncated figure, which is 0 under greedy decoding.
        let spec = format!("{LOCAL_SERVER_PROVIDER}/{}", self.ids[index]);
        let cap = if sampling.thinking {
            sampling.max_new_tokens.max(REASONING_REPLY_TOKENS)
        } else {
            sampling.max_new_tokens
        };
        let config = served_config_capped(
            &spec,
            &self.endpoints.urls()[index],
            self.endpoints.key(),
            &ServedSampling {
                temperature: Some(sampling.temperature),
                top_p: sampling.top_p,
                top_k: Some(sampling.top_k),
                seed: sampling.seed,
                thinking: sampling.thinking,
                thinking_budget: None,
            },
            Some(cap),
        )?;
        let driver = probed_provider(&config).map_err(|e| PolicyError::Remote {
            spec: spec.clone(),
            reason: format!("{e:#}"),
        })?;
        Ok(Arc::from(driver))
    }
}

impl ModelSelection {
    /// This local selection requested from the brain `endpoints` instead of
    /// loaded in this process: a provider that sends each request to the
    /// endpoint the conversation is pinned to, under the id that endpoint
    /// serves the weights as, known by the same [`Self::identity`] an
    /// in-process load has. Refused by name when an endpoint does not serve
    /// them ([`BrainEndpoints::locate`]); never loaded in-process instead.
    pub fn load_served(&self, endpoints: &BrainEndpoints) -> Result<LoadedModel, PolicyError> {
        let Self::Local(weights) = self else {
            return Err(PolicyError::Endpoint {
                url: String::new(),
                reason: format!("{} is not a local model", self.identity()),
            });
        };
        let served = Served {
            endpoints: endpoints.clone(),
            ids: endpoints.locate(weights)?,
        };
        let provider = served.provider(&AGENT_SAMPLING, None)?;
        Ok(LoadedModel {
            provider,
            identity: self.identity(),
            served: Some(served),
        })
    }
}
