// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that acquire a capability
// from a document or a tool and prove it with evidence. If your team needs
// expertise in agent infrastructure or small-model training loops, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Bounded concurrency for work a pipeline fans out.
//!
//! A local model owns one device and is asked one thing at a time; a model
//! reached over an API can be asked several. [`fan_out`] runs work items
//! concurrently up to a limit and returns their results in the order the
//! items were given, so what a stage derives from them does not depend on
//! which answer came first.

use std::future::Future;

use futures::stream::{self, StreamExt};

use crate::context::Context;
use splinter_core::model_ref::ModelRef;

/// Runs `work` over `items`, at most `limit` at a time (zero is one), and
/// returns the results in the order of `items`.
///
/// Items start in order and one starts as soon as a slot is free, whichever
/// item freed it: a finished item never keeps its slot while an earlier,
/// slower one is still running, or one runaway episode would hold back every
/// item behind it. A work item that checks a stop - a cancel, a spent budget -
/// when it begins sees what the items that had finished by then raised. Items
/// already running when a stop is raised are drained, never dropped.
pub async fn fan_out<I, T, F, Fut>(items: I, limit: usize, work: F) -> Vec<T>
where
    I: IntoIterator,
    F: FnMut(I::Item) -> Fut,
    Fut: Future<Output = T>,
{
    fan_out_each(items, limit, work, |_| {}).await
}

/// [`fan_out`], calling `each` with every result as it arrives, in the order
/// the results finish: for a caller that makes what is done so far durable
/// while the rest runs.
pub async fn fan_out_each<I, T, F, Fut>(
    items: I,
    limit: usize,
    mut work: F,
    mut each: impl FnMut(&T),
) -> Vec<T>
where
    I: IntoIterator,
    F: FnMut(I::Item) -> Fut,
    Fut: Future<Output = T>,
{
    let mut finished = stream::iter(items.into_iter().enumerate())
        .map(|(at, item)| {
            let running = work(item);
            async move { (at, running.await) }
        })
        .buffer_unordered(limit.max(1));
    let mut done: Vec<(usize, T)> = Vec::new();
    while let Some((at, result)) = finished.next().await {
        each(&result);
        done.push((at, result));
    }
    done.sort_by_key(|(at, _)| *at);
    done.into_iter().map(|(_, result)| result).collect()
}

impl Context {
    /// How many requests to `model` may be in flight at once: one for a model
    /// run on this machine's device, the configured width for one reached
    /// over an API.
    #[must_use]
    pub fn concurrency_for(&self, model: &ModelRef) -> usize {
        if model.is_remote() {
            self.config().remote_concurrency.max(1)
        } else {
            1
        }
    }
}
