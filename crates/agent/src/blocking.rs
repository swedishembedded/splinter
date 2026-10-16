// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A model asked typed questions by a program that is not itself async.
//!
//! A sample or a command-line tool that only needs "the reply to this, as a
//! value of this shape" should not each carry a runtime and the plumbing to
//! block on it. The bounds - deadline, output tokens, corrections - are set
//! on the [`TypedCall`] where the call is made.

use serde::de::DeserializeOwned;
use serde::Serialize;
use sven_sdk::schemars::JsonSchema;
use sven_sdk::CallError;

use crate::solve::Model;
use crate::typed::{Measured, TypedCall};

/// A model and the runtime its calls are blocked on.
///
/// Calls must not be made from inside an async runtime: blocking one on
/// another panics.
pub struct BlockingModel {
    model: Model,
    runtime: tokio::runtime::Runtime,
}

impl BlockingModel {
    /// `model`, with a runtime of its own.
    ///
    /// # Errors
    /// The runtime could not be created.
    pub fn new(model: Model) -> std::io::Result<Self> {
        Ok(Self {
            model,
            runtime: tokio::runtime::Runtime::new()?,
        })
    }

    /// The model the calls are put to.
    #[must_use]
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// The reply to `input` as a `T`, within the call's own bounds.
    ///
    /// # Errors
    /// The model could not be reached, or its reply was not a `T` after the
    /// permitted corrections.
    pub fn call<T, I>(&self, call: TypedCall<T>, input: &I) -> Result<T, CallError>
    where
        T: DeserializeOwned + JsonSchema + Send + 'static,
        I: Serialize + ?Sized + Sync,
    {
        self.runtime.block_on(call.run(&self.model, input))
    }

    /// [`Self::call`] with the tokens it used, on a failed call too.
    pub fn call_measured<T, I>(&self, call: TypedCall<T>, input: &I) -> Measured<T>
    where
        T: DeserializeOwned + JsonSchema + Send + 'static,
        I: Serialize + ?Sized + Sync,
    {
        self.runtime.block_on(async {
            match call.start(&self.model) {
                Ok(session) => session.call_measured(input).await,
                Err(e) => Measured {
                    result: Err(e),
                    spent: crate::typed::Spent::default(),
                },
            }
        })
    }
}
