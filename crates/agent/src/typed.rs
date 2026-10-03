// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent runtimes that run a model under
// bounded, replayable conditions, for its clients. If your team needs
// expertise in agent execution or typed model calls, you can procure our
// services by sending an email to info@swedishembedded.com.

//! A typed model call: a model asked for a value of a given shape.
//!
//! The call's task and role brief the model, the reply's schema is its
//! shape, and sven reads a reply as that shape and sends one that is not
//! back for correction (up to `repairs` times) or one that breaks the
//! caller's postcondition. It runs with no tools, bounded by a deadline and
//! an optional output-token budget that cover every attempt. Every typed
//! call Splinter makes is described by a [`TypedCall`] and run here, so the
//! bounds, the empty toolset and the stream idle limit are set in one place.

use std::time::Duration;

use serde::de::DeserializeOwned;
use serde::Serialize;
use sven_sdk::schemars::JsonSchema;
use sven_sdk::{CallError, CancelToken, Engine, Method, Toolset};

use crate::solve::{Model, SolveOptions};

/// The check a reply must pass beyond its shape, with the correction sent
/// back to the model when it fails.
pub type Postcondition<T> = Box<dyn Fn(&T) -> Result<(), String> + Send + Sync>;

/// What a typed call asks and how it is bounded.
pub struct TypedCall<T> {
    /// The call's name.
    pub method: String,
    /// The brief the model is given.
    pub task: String,
    /// The role the model is given.
    pub role: String,
    /// How long the call may take, every attempt included.
    pub deadline: Duration,
    /// Output tokens the call may generate; `None` leaves it to the model's
    /// own default.
    pub max_output_tokens: Option<u64>,
    /// How many times a reply that is not the shape is sent back for
    /// correction; `None` leaves it to sven's default.
    pub repairs: Option<u32>,
    /// Stops the call from outside.
    pub cancel: Option<CancelToken>,
    /// What else a reply must satisfy.
    pub postcondition: Option<Postcondition<T>>,
}

impl<T> TypedCall<T> {
    /// A call named `method` briefed with `task` under `role`, bounded by
    /// `deadline` alone.
    #[must_use]
    pub fn new(
        method: impl Into<String>,
        task: impl Into<String>,
        role: impl Into<String>,
        deadline: Duration,
    ) -> Self {
        Self {
            method: method.into(),
            task: task.into(),
            role: role.into(),
            deadline,
            max_output_tokens: None,
            repairs: None,
            cancel: None,
            postcondition: None,
        }
    }

    /// Bounds the call's output to `tokens`.
    #[must_use]
    pub fn max_output_tokens(mut self, tokens: u64) -> Self {
        self.max_output_tokens = Some(tokens);
        self
    }

    /// Allows `repairs` corrections of a malformed reply.
    #[must_use]
    pub fn repairs(mut self, repairs: u32) -> Self {
        self.repairs = Some(repairs);
        self
    }

    /// Stops the call once `cancel` is cancelled.
    #[must_use]
    pub fn cancel(mut self, cancel: CancelToken) -> Self {
        self.cancel = Some(cancel);
        self
    }

    /// Requires `check` of every reply, beyond its shape.
    #[must_use]
    pub fn postcondition(
        mut self,
        check: impl Fn(&T) -> Result<(), String> + Send + Sync + 'static,
    ) -> Self {
        self.postcondition = Some(Box::new(check));
        self
    }
}

impl<T: DeserializeOwned + JsonSchema + Send + 'static> TypedCall<T> {
    /// The call set up on `model`, to be run once or many times.
    pub fn start(self, model: &Model) -> Result<TypedSession<T>, CallError> {
        let mut bounds = SolveOptions::new(self.deadline);
        bounds.max_output_tokens = self.max_output_tokens;
        bounds.cancel = self.cancel;
        bounds.stream_idle = model.stream_idle;
        let mut method = Method::<T>::new(self.method)
            .task(self.task)
            .role(self.role);
        if let Some(repairs) = self.repairs {
            method = method.max_repairs(repairs);
        }
        if let Some(check) = self.postcondition {
            method = method.postcondition(check);
        }
        let engine = Engine::builder()
            .config(bounds.engine_config())
            .model_provider(bounds.provider(model.provider.clone()))
            .toolset(Toolset::none())
            .build()?;
        Ok(TypedSession {
            engine,
            method,
            bounds,
        })
    }

    /// The call run once on `model` over `input`.
    pub async fn run<I: Serialize + ?Sized + Sync>(
        self,
        model: &Model,
        input: &I,
    ) -> Result<T, CallError> {
        self.start(model)?.call(input).await
    }
}

/// A typed call ready to run, on an engine built once.
pub struct TypedSession<T> {
    engine: Engine,
    method: Method<T>,
    bounds: SolveOptions,
}

impl<T: DeserializeOwned + JsonSchema + Send + 'static> TypedSession<T> {
    /// The reply to `input`, within the call's own bounds.
    pub async fn call<I: Serialize + ?Sized + Sync>(&self, input: &I) -> Result<T, CallError> {
        self.engine
            .call_with(&self.method, input, self.bounds.run_options())
            .await
    }

    /// The reply to `input`, within `deadline` alone: one turn of a longer
    /// exchange has its own limit.
    pub async fn call_within<I: Serialize + ?Sized + Sync>(
        &self,
        input: &I,
        deadline: Duration,
    ) -> Result<T, CallError> {
        self.engine
            .call_with(
                &self.method,
                input,
                SolveOptions::new(deadline).run_options(),
            )
            .await
    }
}
