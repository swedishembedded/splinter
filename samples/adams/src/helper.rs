// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent pipelines that put a large language
// model to work behind checks that do not trust it, for its clients. If your
// team needs expertise in schema-checked model calls with bounded cost, you
// can procure our services by sending an email to info@swedishembedded.com.

//! A helper model: a larger model, served by brain, asked for values of a
//! given shape. Every helper role of this sample is a typed call on one of
//! these, so the bounds (deadline, output tokens, repairs) are set where the
//! call is made and the reply is read as its shape or sent back for correction.

use std::time::Duration;

use serde::de::DeserializeOwned;
use serde::Serialize;
use splinter_sdk::agent::schemars::JsonSchema;
use splinter_sdk::agent::solve::Model;
use splinter_sdk::agent::typed::TypedCall;

/// A model that answers typed calls.
pub struct Helper {
    model: Model,
    runtime: tokio::runtime::Runtime,
}

/// Where a helper is served: a brain server's OpenAI-compatible address, the
/// key it printed, and the model's name.
pub struct Served<'a> {
    pub base_url: &'a str,
    pub api_key: &'a str,
    pub model: &'a str,
}

/// How long one helper call may take, repairs included.
pub const CALL_DEADLINE: Duration = Duration::from_secs(900);

impl Helper {
    /// A helper on a served model.
    ///
    /// # Errors
    /// The model cannot be reached or configured.
    pub fn served(served: &Served<'_>) -> anyhow::Result<Self> {
        let loaded = splinter_sdk::model::selection::served_model(
            served.base_url,
            served.api_key,
            served.model,
            Some(0.2),
        )?;
        Ok(Self::with_model(Model::new(
            loaded.provider(),
            loaded.identity(),
        )))
    }

    /// A helper on any model: what a specification uses.
    ///
    /// # Panics
    /// The async runtime cannot be created.
    #[must_use]
    pub fn with_model(model: Model) -> Self {
        #[allow(clippy::expect_used)]
        let runtime = tokio::runtime::Runtime::new().expect("a tokio runtime");
        Self { model, runtime }
    }

    /// The reply to `input` as a `T`, within the call's own bounds.
    ///
    /// # Errors
    /// The model could not be reached, or its reply was not a `T` after the
    /// permitted corrections.
    pub fn call<T, I>(&self, call: TypedCall<T>, input: &I) -> anyhow::Result<T>
    where
        T: DeserializeOwned + JsonSchema + Send + 'static,
        I: Serialize + ?Sized + Sync,
    {
        self.runtime
            .block_on(call.run(&self.model, input))
            .map_err(|e| anyhow::anyhow!("helper call failed: {e}"))
    }
}
